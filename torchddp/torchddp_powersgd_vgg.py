import torch
import argparse
import logging
import time
import numpy as np
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import powerSGD_hook
import os
from torch.profiler import profile, record_function, ProfilerActivity, schedule


class SyntheticDataIter():
    def __init__(self, num_classes, data_shape, max_iter, rank):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        label = np.random.randint(0, num_classes, [self.batch_size, ])
        data = np.random.uniform(-1, 1, data_shape).astype(np.float32)
        self.data = torch.from_numpy(data).to(rank)
        self.label = torch.from_numpy(label).to(rank)

    def __iter__(self):
        return self

    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return self.data, self.label
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (default: 128)')
    parser.add_argument('--model', type=str, default='vgg19',
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of NN output classes, default is 1000')
    parser.add_argument('--num-iterations', type=int, default=200,
                    help='number of iterations trained per epoch')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='number of batches to wait before logging (default: 20)')
    parser.add_argument('--powersgd-rank', type=int, default=1,
                        help='rank of powersgd low-rank vectors')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='generate profiling results')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    dist.init_process_group('nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    log_interval = args.log_interval
    epoch_size = args.num_iterations
    image_shapes = {
        'vgg19': (3, 224, 224)
    }
    image_shape = image_shapes[args.model]
    data_shape = (batch_size,) + image_shape

    models = {
        'vgg19': torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=False)
    }
    model = models[args.model].to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # PowerSGD configs here
    low_rank = args.powersgd_rank
    state = PowerSGDState(
        process_group=dist.distributed_c10d.group.WORLD,
        matrix_approximation_rank=low_rank,
        use_error_feedback=True,
        warm_start=False,
        start_powerSGD_iter=2)
    model.register_comm_hook(state, powerSGD_hook)

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.SGD(model.parameters(), lr=0.001)

    train_data = SyntheticDataIter(num_classes, data_shape, epoch_size, local_rank)

    if args.profile:
        btic = time.time()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=schedule(skip_first=100, wait=0, warmup=80, active=10, repeat=1),
            with_stack=True) as prof:
            with record_function("model_training"):
                for epoch in range(num_epochs):
                    for nbatch, batch in enumerate(train_data, start=1):
                        data, label = batch
                        with record_function("forward"):
                            output = model(data)
                        with record_function("backward"):
                            loss_fn(output, label).backward()
                        with record_function("update"):
                            opt.step()
                        if nbatch % log_interval == 0:
                            if rank == 0:
                                batch_speed = world_size * batch_size * log_interval / (time.time() - btic)
                                logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec',
                                                epoch, nbatch, batch_speed)
                                step_time = 1000 / (batch_speed / world_size / batch_size)
                                logging.info('step time: %.2f ms', step_time)
                            btic = time.time()
                        prof.step()
        prof.export_chrome_trace("torchddp_vgg.json")
    else:
        btic = time.time()
        for epoch in range(num_epochs):
            for nbatch, batch in enumerate(train_data, start=1):
                data, label = batch
                output = model(data)
                loss_fn(output, label).backward()
                opt.step()
                if nbatch % log_interval == 0:
                    if rank == 0:
                        batch_speed = world_size * batch_size * log_interval / (time.time() - btic)
                        logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec',
                                        epoch, nbatch, batch_speed)
                        step_time = 1000 / (batch_speed / world_size / batch_size)
                        logging.info('step time: %.2f ms', step_time)
                    btic = time.time()


if __name__ == '__main__':
    main()
