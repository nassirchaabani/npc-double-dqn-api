import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NPCGridWorld(gym.Env):
    

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size: int = 5, max_steps: int = 50):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, size - 1, shape=(6,), dtype=np.float32)
        self.agent = np.zeros(2, dtype=np.int64)
        self.target = np.zeros(2, dtype=np.int64)
        self.obstacle = np.zeros(2, dtype=np.int64)
        self.steps = 0

    def _observation(self) -> np.ndarray:
        return np.concatenate((self.agent, self.target, self.obstacle)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cells = self.np_random.choice(self.size * self.size, size=3, replace=False)
        self.agent = np.array(divmod(int(cells[0]), self.size), dtype=np.int64)
        self.target = np.array(divmod(int(cells[1]), self.size), dtype=np.int64)
        self.obstacle = np.array(divmod(int(cells[2]), self.size), dtype=np.int64)
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
        previous_distance = int(np.abs(self.agent - self.target).sum())
        candidate = np.clip(self.agent + moves[int(action)], 0, self.size - 1)
        self.agent = candidate
        new_distance = int(np.abs(self.agent - self.target).sum())
        self.steps += 1
        reached_target = bool(np.array_equal(self.agent, self.target))
        hit_obstacle = bool(np.array_equal(self.agent, self.obstacle))
        terminated = reached_target or hit_obstacle
        truncated = self.steps >= self.max_steps and not terminated
        if reached_target:
            reward = 10.0
        elif hit_obstacle:
            reward = -10.0
        else:
            
            reward = -0.1 + 0.2 * (previous_distance - new_distance)
        return self._observation(), reward, terminated, truncated, {
            "success": reached_target
        }

    def render(self):
        grid = np.full((self.size, self.size), ".", dtype="<U1")
        grid[tuple(self.target)] = "T"
        grid[tuple(self.obstacle)] = "X"
        grid[tuple(self.agent)] = "A"
        return "\n".join(" ".join(row) for row in grid)
