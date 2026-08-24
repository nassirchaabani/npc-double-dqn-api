import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .environment import NPCGridWorld
from .model import encode_observation, load_model


def evaluate_policy(policy, episodes: int = 500, seed_offset: int = 10_000):
    env = NPCGridWorld()
    successes = 0
    rewards = []
    for episode in range(episodes):
        raw_state, _ = env.reset(seed=seed_offset + episode)
        state = encode_observation(raw_state, env.size)
        done = False
        total_reward = 0.0
        while not done:
            action = int(policy(state, env))
            raw_state, reward, terminated, truncated, info = env.step(action)
            state = encode_observation(raw_state, env.size)
            done = terminated or truncated
            total_reward += reward
            if done:
                successes += int(info["success"])
        rewards.append(total_reward)
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_reward": float(np.mean(rewards)),
    }


def random_policy(seed: int = 42):
    rng = np.random.default_rng(seed)
    return lambda state, env: rng.integers(env.action_space.n)


def greedy_model_policy(model_path: str):
    model = load_model(model_path)

    def policy(state, env):
        with torch.no_grad():
            return model(torch.tensor(state).unsqueeze(0)).argmax(1).item()

    return policy


def compare(double_dqn_model: str, dqn_model: str, episodes: int = 500):
    results = {
        "random": evaluate_policy(random_policy(), episodes),
        "dqn": evaluate_policy(greedy_model_policy(dqn_model), episodes),
        "double_dqn": evaluate_policy(greedy_model_policy(double_dqn_model), episodes),
    }
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--double-dqn-model", default="artifacts/npc_dqn.pt")
    parser.add_argument("--dqn-model", default="artifacts/npc_classic_dqn.pt")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", default="artifacts/baseline_results.json")
    args = parser.parse_args()
    result = compare(args.double_dqn_model, args.dqn_model, args.episodes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
