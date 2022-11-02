from dataclasses import dataclass

import numpy as np


def shape(exp):
    if type(exp) is np.ndarray:
        return list(exp.shape)
    else:
        return []


def type_of(exp):
    if type(exp) is np.ndarray:
        return exp.dtype
    else:
        return type(exp)


@dataclass
class ReplayMemoryParams:
    size: int = 50_000
    cer: bool = True


class ReplayMemory:
    """
    Replay memory class for RL
    """

    def __init__(self, params: ReplayMemoryParams):
        self.k = 0
        self.head = -1
        self.full = False
        self.size = params.size
        self.memory = None
        self.cer = params.cer

    def initialize(self, experience):
        self.memory = [np.zeros(shape=[self.size] + shape(exp), dtype=type_of(exp)) for exp in experience]
        print(f"Experience replay size {sum([n.size * n.itemsize for n in self.memory]) / 1e6} MB")

    def store(self, experience):
        if self.memory is None:
            self.initialize(experience)
        if len(experience) != len(self.memory):
            raise Exception('Experience not the same size as memory', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.k] = e

        self.head = self.k
        self.k += 1
        if self.k >= self.size:
            self.k = 0
            self.full = True

    def sample(self, batch_size):
        r = self.size
        if not self.full:
            r = self.k
        random_idx = np.random.choice(r, size=batch_size, replace=False)

        if self.cer:
            random_idx[0] = self.head  # Always add the latest one

        return [mem[random_idx] for mem in self.memory]

    def get(self, start, length):
        return [mem[start:start + length] for mem in self.memory]

    def get_size(self):
        if self.full:
            return self.size
        return self.k

    def get_max_size(self):
        return self.size

    def reset(self):
        self.k = 0
        self.head = -1
        self.full = False

    def shuffle(self):
        self.memory = self.sample(self.get_size())
