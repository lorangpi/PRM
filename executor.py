'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This files implements the structure of the executor object used in this paper.

'''
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from rl_model import load_policy, get_model
set_random_seed(0, using_cuda=True)

class Executor():
	def __init__(self, id, mode, I=None, Beta=None, Circumstance=None, basic=False):
		super().__init__()
		self.id = id
		self.I = I
		self.Circumstance = Circumstance
		self.Beta = Beta
		self.basic = basic
		self.mode = mode
		self.policy = None

	def path_to_json(self):
		return {self.id:self.policy}

class Executor_RL(Executor):
	def __init__(self, id, alg, policy, I, Beta, Circumstance=None, basic=False):
		super().__init__(id, "RL", I, Beta, Circumstance, basic)
		self.alg = alg
		self.policy = policy
		self.model = None

	def execute(self, env, operator, render, obs):
		'''
		This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
		produced by the policy on that state. 
		'''
		print("Loading policy {}".format(self.policy))
		if self.model is None:
			self.model = load_policy(self.alg, env, self.policy, seed=0)
		step_executor = 0
		done = False
		success = False
		while not done:
			action, _states = self.model.predict(obs)
			try: 
				obs, reward, terminated, truncated, info = env.step(action)
				done = terminated or truncated
			except:
				obs, reward, done, info = env.step(action)
			step_executor += 1
			success = self.Beta(operator=operator, env=env)
			if step_executor > 500:
				done = True
			if render:
				env.render()
		return obs, success