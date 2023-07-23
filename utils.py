import os
import logging
import csv

import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import ignite.distributed as idist


class Logger(object):

    def __init__(self, logdir, prefix='', resume=None):
        self.logdir = logdir
        self.rank = idist.get_rank()
        self.csv_msg = None
        self.prefix = prefix

        handlers = [logging.StreamHandler(os.sys.stdout)]
        if logdir is not None and self.rank == 0:
            if resume is None:
                try:
                    os.makedirs(logdir)
                except FileExistsError:
                    print('Warning: log file already exists!')
            handlers.append(logging.FileHandler(os.path.join(logdir, f'log_{prefix}.txt')))
            self.writer = SummaryWriter(log_dir=logdir)
        else:
            self.writer = None

        logging.basicConfig(format=f"[%(asctime)s ({self.rank})] %(message)s",
                            level=logging.INFO,
                            handlers=handlers)
        logging.info(' '.join(os.sys.argv))

    def log_msg(self, msg):
        if idist.get_rank() > 0:
            return
        logging.info(msg)

    def log(self, engine, global_step, print_msg=True, **kwargs):
        if idist.get_rank() > 0:
            return
        msg = f'[epoch {engine.state.epoch}] [iter {engine.state.iteration}]'
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is float:
                msg += f' [{k} {v:.3f}]'
            else:
                msg += f' [{k} {v}]'

            if self.writer is not None:
                self.writer.add_scalar(k, v, global_step)

        if print_msg:
            logging.info(msg)

    def log_csv(self, vals):
        self.csv_msg = vals

    def save_csv(self, prefix=''):
        csv_file = os.path.join(self.logdir, 'result.csv')
        with open(csv_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([prefix] + self.csv_msg)

    def save(self, engine, record_epoch=True, **kwargs):
        if idist.get_rank() > 0:
            return

        state = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.nn.parallel.DistributedDataParallel):
                v = v.module

            if hasattr(v, 'state_dict'):
                state[k] = v.state_dict()
                continue

            if type(v) is list and hasattr(v[0], 'state_dict'):
                state[k] = [x.state_dict() for x in v]
                continue

            state[k] = v  # record other info

        if record_epoch:
            filename = f'ckpt-{self.prefix}-{engine.state.epoch}.pth'
        else:
            filename = f'ckpt-{self.prefix}-best.pth'
        torch.save(state, os.path.join(self.logdir, filename))
        print(f'Checkpoint saved to {os.path.join(self.logdir, filename)}')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1) # top-k index: size (B, k)
        pred = pred.t() # size (k, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k * 100.0 / batch_size)

        if len(acc) == 1:
            return acc[0]
        else:
            return acc


class LambdaLR:
    """https://github.com/eriklindernoren/PyTorch-GAN/blob/36d3c77e5ff20ebe0aeefd322326a134a279b93e/implementations/unit/models.py"""
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)