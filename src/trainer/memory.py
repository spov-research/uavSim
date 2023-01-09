from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from src.trainer.utils import shape, type_of
from utils import Factory


class ReplayMemory:
    """
    Replay memory class for RL
    """
    @dataclass
    class Params:
        size: int = 50_000
        cer: bool = True

    def __init__(self, params: Params):
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

        return [mem[random_idx] for mem in self.memory], np.ones_like(random_idx), random_idx

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

    def update_td_error(self, index, td_error):
        pass

    def get_logs(self):
        return {}




class PrioritizedReplayMemory(ReplayMemory):
    @dataclass
    class Params(ReplayMemory.Params):
        alpha: float = 1.0
        beta: float = 1.0
        min_prio: float = 1e-3
    def __init__(self, params: Params):
        super().__init__(params)
        self.params = params
        self.priorities = np.ones(params.size)

    def store(self, experience):
        self.priorities[self.head] = np.max(self.priorities[:max(self.get_size(), 1)])
        super().store(experience)

    def sample(self, batch_size) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        r = self.size
        if not self.full:
            r = self.k
        rank = r - np.argsort(np.argsort(self.priorities[:r]))
        p = (1 / rank) ** self.params.alpha
        p = p / np.sum(p)
        random_idx = np.random.choice(r, size=batch_size, replace=False, p=p)

        w = (1.0 / (r * p[random_idx])) ** self.params.beta

        return [mem[random_idx] for mem in self.memory], w, random_idx

    def update_td_error(self, index, td_error):
        self.priorities[index] = np.abs(td_error.numpy()) + self.params.min_prio

    def get_logs(self):
        prios = self.priorities[:self.get_size()]
        return {"prio_min": np.min(prios), "prio_max": np.max(prios), "prio_avg": np.mean(prios)}


class ReplayMemoryFactory(Factory):

    @classmethod
    def registry(cls):
        return {
            "normal": ReplayMemory,
            "prio": PrioritizedReplayMemory
        }

    @classmethod
    def defaults(cls) -> Tuple[str, type]:
        return "normal", ReplayMemory
