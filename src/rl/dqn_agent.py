from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    input_dim: int = 8
    hidden_dim: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 256
    target_update_steps: int = 2_000
    train_start_size: int = 5_000
    train_every_steps: int = 4
    max_grad_norm: float = 5.0

    # epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000


class DQNAgent:
    def __init__(self, cfg: DQNConfig, seed: int = 0, device: Optional[str] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q = QNetwork(cfg.input_dim, cfg.hidden_dim, 2).to(self.device)
        self.q_target = QNetwork(cfg.input_dim, cfg.hidden_dim, 2).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.steps = 0

    def epsilon(self) -> float:
        # decaimiento lineal (simple y estable)
        frac = min(1.0, self.steps / max(1, self.cfg.eps_decay_steps))
        return float(self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start))

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Devuelve acción {0,1}. Si greedy=True, usa argmax sin explorar y sin consumir steps."""
        if greedy:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                qvals = self.q(s).squeeze(0)
                return int(torch.argmax(qvals).item())

        eps = self.epsilon()
        self.steps += 1

        if self.rng.random() < eps:
            return int(self.rng.integers(0, 2))

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q(s).squeeze(0)
            return int(torch.argmax(qvals).item())

    def train_step(self, batch, global_step: int) -> float:
        """Un paso de entrenamiento a partir de un batch del replay buffer."""
        states, actions, next_states, rewards, dones = batch

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(states_t).gather(1, actions_t)

        # target: r + gamma * max_a' Q_target(s', a') * (1-done)
        with torch.no_grad():
            q_next = self.q_target(next_states_t).max(dim=1, keepdim=True).values
            target = rewards_t + self.cfg.gamma * q_next * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        # actualizar target network cada N steps
        if global_step % self.cfg.target_update_steps == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "q_state_dict": self.q.state_dict(),
                "q_target_state_dict": self.q_target.state_dict(),
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["q_state_dict"])
        self.q_target.load_state_dict(ckpt["q_target_state_dict"])
        self.steps = int(ckpt.get("steps", 0))