import argparse
import numpy as np
import os
import robosuite as suite
import gymnasium
from stable_baselines3.common.env_checker import check_env
from robosuite.wrappers.gym_wrapper import GymWrapper
from HER_wrapper import HERWrapper
from operator_wrapper import ReachWrapper
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--timesteps', type=int, default=int(1e6), help='Number of timesteps to train for')
parser.add_argument('--task', type=str, default='reach', choices=['pick', 'reach', 'place'], help='Task to learn')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
args = parser.parse_args()

# Create the environment
env = suite.make(
    #"PickPlaceCan",
    "Locked",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon=1000,
    render_camera="agentview",
)
env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
env = ReachWrapper(env, dense_reward=True)
env = HERWrapper(env, dense_reward=True)
#check_env(env)
env = Monitor(env, filename=None, allow_early_resets=True)

# Define the goal selection strategy
goal_selection_strategy = GoalSelectionStrategy.FUTURE

# Define the model
model = SAC(
    'MultiInputPolicy',
    env,
    replay_buffer_class=HerReplayBuffer,
     #Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
    #    online_sampling=True,),
    ),
    learning_rate=args.lr,
    buffer_size=int(1e6),
    learning_starts=10000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_kwargs=dict(net_arch=[512, 512, 256]),
    verbose=1,
    tensorboard_log='./logs/'
)

# Create the evaluation environment
eval_env = suite.make(
    #"PickPlaceCan",
    "Locked",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon=1000,
    render_camera="agentview",
)
eval_env = GymWrapper(eval_env, keys=['robot0_proprio-state', 'object-state'])
eval_env = ReachWrapper(eval_env, dense_reward=True)
eval_env = HERWrapper(eval_env, dense_reward=True)
eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)

# Define the evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/',
    log_path='./logs/',
    eval_freq=10000,
    n_eval_episodes=50,
    deterministic=True,
    render=False,
    callback_on_new_best=None,
    verbose=1
)

# Train the model
model.learn(
    total_timesteps=args.timesteps,
    #callback=eval_callback
)

# Save the model
model.save(os.path.join('./models/', args.task + '_sac'))