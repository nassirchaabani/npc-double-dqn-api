import argparse

import torch

from .environment import NPCGridWorld
from .model import encode_observation, load_model


def evaluate(model_path: str, episodes: int = 100):
    env, model = NPCGridWorld(), load_model(model_path)
    successes, rewards = 0, []
    for episode in range(episodes):
        raw_state, _ = env.reset(seed=10_000 + episode)
        state = encode_observation(raw_state, env.size)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                action = int(model(torch.tensor(state).unsqueeze(0)).argmax(1).item())
            raw_state, reward, terminated, truncated, info = env.step(action)
            state = encode_observation(raw_state, env.size)
            done, total = terminated or truncated, total + reward
            if done:
                successes += int(info["success"])
        rewards.append(total)
    result = {"episodes": episodes, "success_rate": successes / episodes, "mean_reward": sum(rewards) / episodes}
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/npc_dqn.pt")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
