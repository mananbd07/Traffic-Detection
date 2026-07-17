import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from simulation.city_sim import CitySimulation

class TrafficEnv(gym.Env):
def **init**(self, grid_size=3):
super().**init**()

```
    self.grid_size = grid_size
    self.sim = CitySimulation(grid_size=grid_size)

    self.num_intersections = grid_size * grid_size

    self.action_space = spaces.MultiDiscrete([self.num_intersections, 10])

    self.observation_space = spaces.Box(
        low=0,
        high=255,
        shape=(1, 84, 84),
        dtype=np.uint8
    )

    self.steps = 0
    self.max_steps = 300
    self.last_action = None

def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.steps = 0
    self.sim.reset()
    self.last_action = None
    return self._get_image(), {}

def step(self, action):
    self.steps += 1

    idx = int(action[0])
    duration = int(action[1]) + 1

    r = idx // self.grid_size
    c = idx % self.grid_size

    switching_penalty = 0
    if self.last_action is not None and self.last_action != idx:
        switching_penalty = 1

    self.last_action = idx

    current = self.sim.signals[r][c].state
    new_state = "EW" if current == "NS" else "NS"
    self.sim.signals[r][c].set(new_state, duration)

    _, moved = self.sim.step()

    grid = self.sim.get_state()
    total_cars = np.sum(grid)
    avg_cars = total_cars / self.num_intersections
    max_lane = np.max(grid)

    reward = (
        -0.3 * avg_cars
        -0.7 * max_lane
        +0.8 * moved
        -0.2 * duration
        -1.0 * switching_penalty
    ) / 50.0

    done = self.steps >= self.max_steps

    return self._get_image(), reward, done, False, {}

def _get_image(self):
    size = self.grid_size * 20
    img = np.zeros((size, size), dtype=np.uint8)

    grid = self.sim.get_state()

    for r in range(self.grid_size):
        for c in range(self.grid_size):
            val = int(grid[r][c])
            img[r*20:(r+1)*20, c*20:(c+1)*20] = min(val * 40, 255)

    img = cv2.resize(img, (84, 84))
    return img[np.newaxis, :, :]
```
