'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu
'''
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

class HRL(gym.Env):
	def __init__(self, env, teacher_policy, icm_network, student_policy=None, high_env=None, verbose=False, horizon=1000):
		self.horizon = horizon
		self.low_env = env
		self.high_env = high_env
		self.teacher_policy = teacher_policy
		self.icm_network = icm_network
		self.buffer = []
		if student_policy is None:
			# Initialize replay buffer
			self.student_policy_buffer = ReplayBuffer(100000, self.teacher_policy.observation_space, self.teacher_policy.action_space)
			self.student_policy = SAC("MlpPolicy", env=env, learning_rate=5e-3, buffer_size=10000, replay_buffer_class=ReplayBuffer, verbose=1,
                tensorboard_log="./data/student/",)

		else:
			self.student_policy = student_policy
			self.student_policy_buffer = None

		self.step_executor = 0

		self.verbose = verbose
		self.verboseprint = print if verbose else lambda *a, **k: None
		self.verboseprint("\n\n---------------------------- RUNNING AN HRL_FollowLane INSTANCE. --------------------------\n-")

		# Action space consists of choosing between the source policy and the student_policy + the number of steps to execute
		self.action_space = spaces.Box(low=np.asarray([0, 1]), high=np.asarray([1, 10]), dtype=int)
		#self.action_space = spaces.MultiDiscrete([2, 10])
		# Add one box to the observation space to store an icm value (i.e., the error of the icm)
		self.observation_space = spaces.Box(low=self.low_env.observation_space.low + [-1], high=self.low_env.observation_space.high + [1], dtype=self.low_env.observation_space.dtype)
		#self.observation_space = spaces.Box(low=np.asarray([-1]), high=np.asarray([1]), dtype=np.float32)
		
		self.obs, _ = self.low_env.reset()

		#self.verboseprint("\n\ninit_size obs: \n", self.observation_space)
		self.verboseprint("init_size act: \n", self.action_space)
		print('\n')

	def step(self, action):
		done = False
		infos = {}
		rew_eps = 0
		infos.update({"success": False})
		action = action.astype(int)
		#print("Action: ", action)
		last_step_student = 0

		model = self.teacher_policy if action[0] == 0 else self.student_policy
		obs = self.obs

		# Execute the action
		for i in range(action[1]): # 1-10 steps per operator
			low_action, _states = model.predict(obs, deterministic=True)
			next_obs, reward, termination, truncated, info = self.low_env.step(low_action)
			rew_eps += reward
			self.step_executor += 1
			rew_eps += reward
			#print(info, self.step_executor)
			done = done or termination or truncated
			obs = next_obs
			if self.step_executor > self.horizon:
				truncated = True
				done = True
			if action[0] == 1: # If the student_policy is being used
				# Add the transition to the student_policy buffer
				last_step_student = self.step_executor
				self.buffer.append([obs, next_obs, low_action, reward, done, [{"TimeLimit.truncated": truncated}]])
			if done:
				break

		info.update({"success": termination})

		if done and self.buffer != [] and self.student_policy_buffer is not None:
			print("Training student_policy")
			if termination:
				infos.update({"success": True})
				# Replace last reward with a high reward
				rew_eps = self.horizon / (1+(self.step_executor - last_step_student))
				self.buffer[-1][3] += rew_eps 
			# Add the transitions to the student_policy buffer
			for transition in self.buffer:
				#print("Adding transition to student_policy buffer: ", *transition)
				self.student_policy_buffer.add(*transition)
			# Set the student_policy buffer as the student_policy buffer
			self.student_policy.replay_buffer = self.student_policy_buffer
			# Train SAC on the data in the buffer
			if len(self.buffer) > 1000:
				self.student_policy.learn(total_timesteps=10000)
				self.buffer = []

		# Compute ICM error
		_, _, icm_reward = self.icm_network(torch.tensor(self.obs, dtype=torch.float32), torch.tensor(low_action, dtype=torch.float32), torch.tensor(next_obs, dtype=torch.float32))
		#print("ICM reward: ", icm_reward)
		observation = obs + [icm_reward.item()]
		self.obs = obs
		#observation = [icm_reward.item()]

		return observation, rew_eps, termination, truncated, info

	def reset(self, **kwargs):
		self.step_executor = 0
		obs = self.low_env.reset(**kwargs)
		#return obs
		return np.asarray([0]), {}
	

	def close(self):
		self.low_env.close()
	
	def overwrite_executor_id(self, executor_id):
		self.executor_id = executor_id
