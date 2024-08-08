'''
# This file is the pre-novelty executor learner.
'''
import sys
import gymnasium
sys.modules["gym"] = gymnasium

import os
import time
import gymnasium as gym
import argparse
from datetime import datetime
from gym_panda_novelty.operator_learners.train import train
from domain_specific.synapses import *
from stable_baselines3.sac.policies import MlpPolicy, MultiInputPolicy
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from executor import Executor

from stable_baselines3.common.utils import set_random_seed
set_random_seed(0, using_cuda=True)

# HER parameters
# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
# If True the HER transitions will get sampled online
online_sampling = True

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)


ap = argparse.ArgumentParser()
ap.add_argument("-steps", "--steps", default=500_000, help="Number of steps to train", type=int)
ap.add_argument("-eval_freq", "--eval_freq", default=10_000, help="Number of steps to train", type=int)
ap.add_argument("-s", "--seed", default=0, help="Experiment seed", type=int)
ap.add_argument("-e", "--executor", help="Executor to train: reach, pick or drop", default='reach', type=str)
args = vars(ap.parse_args())


operator = args["executor"]
env_id = training_env[operator]
print("\nCreating training and eval envs.")
eval_env = gym.make(env_id)
env = gym.make(env_id)

data_folder = './gym_panda_novelty/policies/'
name = operator
folder = data_folder + name + f"{to_datestring(time.time())}"  + '/'
policy_folder = folder + "policy/" + name
tensorboard_folder = folder+"tensorboard/"

os.makedirs(folder, exist_ok=True)
os.makedirs(policy_folder, exist_ok=True)
os.makedirs(tensorboard_folder, exist_ok=True)


policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]))
model = SAC(MultiInputPolicy, 
	env, 
	replay_buffer_class=HerReplayBuffer,
	# Parameters for HER
	replay_buffer_kwargs=dict(
		n_sampled_goal=4,
		goal_selection_strategy=goal_selection_strategy,
	), 
	verbose=1, gamma=0.95, learning_rate=0.0003, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_folder+name, device="cuda")

model = train(
	env, 
	eval_env=eval_env, 
	model=model,
	policy_kwargs=policy_kwargs, 
	save_freq=args["eval_freq"], 
	total_timesteps=args["steps"],
	best_model_save_path=policy_folder,
	eval_freq=args["eval_freq"], 
	n_eval_episodes=20,
	save_path=folder,
	run_id=name)

model.save(f"{folder}models/{name}")