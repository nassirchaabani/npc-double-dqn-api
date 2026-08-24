import argparse
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .environment import NPCGridWorld
from .model import DQN, encode_observation


def train(
    episodes: int = 2000,
    seed: int = 42,
    output: str = "artifacts/npc_dqn.pt",
    algorithm: str = "double_dqn",
):
    if algorithm not in {"dqn", "double_dqn"}:
        raise ValueError("algorithm must be 'dqn' or 'double_dqn'")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = NPCGridWorld()
    env.action_space.seed(seed)
    policy, target = DQN(), DQN()
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    loss_fn = nn.SmoothL1Loss()
    replay = deque(maxlen=10_000)
    gamma, batch_size = 0.99, 128
    learning_steps = 0

    for episode in range(episodes):
        raw_state, _ = env.reset(seed=seed + episode)
        state = encode_observation(raw_state, env.size)
        done = False
        total_reward = 0.0
        while not done:
            epsilon = 0.05 + 0.95 * math.exp(-3.5 * episode / max(1, episodes))
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(policy(torch.tensor(state).unsqueeze(0)).argmax(1).item())
            raw_next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_observation(raw_next_state, env.size)
            done = terminated or truncated
            replay.append((state, action, reward, next_state, done))
            state, total_reward = next_state, total_reward + reward

            if len(replay) >= 1_000:
                batch = random.sample(replay, batch_size)
                states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
                actions = torch.tensor([x[1] for x in batch]).unsqueeze(1)
                rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32)
                dones = torch.tensor([x[4] for x in batch], dtype=torch.float32)
                current = policy(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    if algorithm == "double_dqn":
                        # Double DQN separates action selection from evaluation.
                        next_actions = policy(next_states).argmax(1, keepdim=True)
                        next_values = target(next_states).gather(1, next_actions).squeeze(1)
                    else:
                        # Classical DQN selects and evaluates with the target network.
                        next_values = target(next_states).max(1).values
                    expected = rewards + gamma * next_values * (1 - dones)
                loss = loss_fn(current, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()
                learning_steps += 1
                if learning_steps % 250 == 0:
                    target.load_state_dict(policy.state_dict())
        if episode % 50 == 0:
            print(f"episode={episode:04d} reward={total_reward:.1f} epsilon={epsilon:.3f}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output", default="artifacts/npc_dqn.pt")
    parser.add_argument("--algorithm", choices=["dqn", "double_dqn"], default="double_dqn")
    args = parser.parse_args()
    train(args.episodes, output=args.output, algorithm=args.algorithm)
