from domain_specific.synapses import *
import gym
import gym_panda_novelty
from stable_baselines3 import SAC
from HDDL.generate_hddl import *
from state import *
from carla	import *
from novelties import *
from gym_panda_novelty.novelties.novelty_wrapper import NoveltyWrapper

#from stable_baselines3 import HerReplayBuffer, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback



training_steps = 300_000
eval_each_n_steps = 300
eval_episodes = 100
seed = 0

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True

folder = "TEST_RL/weights"
folder_buffer = "TEST_RL/buffer/"
folder_pretrain = "TEST_RL/weights_Pretrained/"
folder_buffer_pretrain = "TEST_RL/buffer_Pretrained/"


#env_config = {'discrete': False, 'n_distance_points_per_camera': 0}
env_config = {'discrete': False, 'stop_prob':0.0, 'max_offset':30}
# set to False, if you wish to watch the agent train in the viewport (very slow!):
rendering = {'no_rendering': False}

novelty_name = "black_ice" # is an input in the actual framework
failed_operator = "goForward" # is an input in the actual framework

env_id = training_env[failed_operator]
env = gym.make(env_id, runtime_settings=rendering, params=env_config, verbose=True)

env = NoveltyWrapper(env, loc="l2", dir="s")
novelties = [novelties_mapping[novelty_name](env, 1)]
env.set_novelties(novelties)
env = Monitor(env)

executor_queue = executors[applicator[failed_operator][0]]

name_saved = "transfer"+str(seed)
model = SAC.load(executor_queue.policy, env=env, tensorboard_log="./TEST_RL/tensorboard/transfer"+str(seed), device="cuda") 
model.set_env(env)
parameters = model.get_parameters()
model.set_parameters(parameters)
env_eval = env
# Separate evaluation env
eval_callback = EvalCallback(env_eval, best_model_save_path="./TEST_RL/logs/",
                             log_path="./TEST_RL/logs/", eval_freq=500,
                             deterministic=True, render=False)

#event_callback = EveryNTimesteps(n_steps=eval_each_n_steps, callback=eval_callback)
model.learn(total_timesteps=100000, log_interval=10, callback=eval_callback)
model.save("./follow_lane.zip")
