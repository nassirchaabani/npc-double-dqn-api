from pathlib import Path

import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, observation_size: int = 6, action_count: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_count),
        )

    def forward(self, observation):
        return self.network(observation)


def encode_observation(observation, grid_size: int = 5) -> np.ndarray:

    raw = np.asarray(observation, dtype=np.float32)
    agent, target, obstacle = raw[:2], raw[2:4], raw[4:6]
    scale = float(grid_size - 1)
    return np.concatenate(((target - agent) / scale, (obstacle - agent) / scale, agent / scale)).astype(np.float32)


def load_model(path: str | Path, device: str = "cpu") -> DQN:
    model = DQN()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
