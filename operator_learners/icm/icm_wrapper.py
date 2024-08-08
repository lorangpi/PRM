import gymnasium as gym
import torch

class ICMWrapper(gym.Wrapper):
    def __init__(self, env, icm, intrinsic_reward_weight=0.5):
        super().__init__(env)
        self.icm = icm
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.current_observation = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.current_observation = observation['observation'] if type(observation) is dict else observation
        return observation, info

    def step(self, action):
        next_observation, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        
        next_observation_icm = next_observation['observation'] if type(next_observation) is dict else next_observation

        # Compute the intrinsic reward
        _, _, intrinsic_reward = self.icm(
            torch.tensor(self.current_observation, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(next_observation_icm, dtype=torch.float32)
        )

        # Combine the extrinsic and intrinsic rewards
        #print("Extrinsic reward: ", extrinsic_reward, " Intrinsic reward: ", self.intrinsic_reward_weight * intrinsic_reward.item(), " Total reward: ", extrinsic_reward + self.intrinsic_reward_weight * intrinsic_reward.item())
        total_reward = extrinsic_reward + self.intrinsic_reward_weight * intrinsic_reward.item()

        self.current_observation = next_observation['observation'] if type(next_observation) is dict else next_observation
        return next_observation, total_reward, terminated, truncated, info