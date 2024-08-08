import time
import gymnasium as gym
import numpy as np

class TrainingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.time = time.perf_counter()
        self.frames = 0

    def step(self, action):
        self.frames += 1
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self):
        print(f"INFO Episode length {self.frames}, FPS {self.frames/(time.perf_counter()-self.time)}.")
        self.time = time.perf_counter()
        self.frames = 0
        # TODO: make the seed sampling deterministic
        seed = np.random.randint(0, high=2**31-1)
        print(f"INFO: episode seed: {seed}")
        self.env.seed(seed)
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()
