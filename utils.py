import sys
import torch
import numpy
import random


class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_padding_mask(x, x_lens):
    """
    :param x: (seq_len, batch_size, feature_dim)
    :param x_lens: sequence lengths within a batch with size (batch_size,)
    :return: padding_mask with size (batch_size, seq_len)
    """
    seq_len, batch_size, _ = x.size()
    mask = torch.ones(batch_size, seq_len, device=x.device)
    for seq, seq_len in enumerate(x_lens):
        mask[seq, :seq_len] = 0
    mask = mask.bool()
    return mask

