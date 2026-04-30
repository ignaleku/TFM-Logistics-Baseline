from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000):
        self.capacity = int(capacity)
        self.data: List[Transition] = []
        self.pos = 0

    def add(self, t: Transition) -> int:
        """Añade transición y devuelve su índice (para poder actualizar reward luego)."""
        if len(self.data) < self.capacity:
            self.data.append(t)
            idx = len(self.data) - 1
        else:
            idx = self.pos
            self.data[idx] = t
            self.pos = (self.pos + 1) % self.capacity
        return idx

    def set_reward(self, idx: int, reward: float) -> None:
        t = self.data[idx]
        self.data[idx] = Transition(t.state, t.action, t.next_state, float(reward), t.done)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, ...]:
        n = len(self.data)
        idxs = rng.integers(0, n, size=batch_size)
        states = np.stack([self.data[i].state for i in idxs])
        actions = np.array([self.data[i].action for i in idxs], dtype=np.int64)
        next_states = np.stack([self.data[i].next_state for i in idxs])
        rewards = np.array([self.data[i].reward for i in idxs], dtype=np.float32)
        dones = np.array([self.data[i].done for i in idxs], dtype=np.float32)
        return states, actions, next_states, rewards, dones

    def __len__(self) -> int:
        return len(self.data)