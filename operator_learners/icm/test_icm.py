import warnings
warnings.filterwarnings("ignore")

#import robosuite as suite
#from robosuite.wrappers import GymWrapper
#from robosuite.wrappers.behavior_cloning.hanoi_pick_place import PickPlaceWrapper
from argparse import ArgumentParser
from icm import ICM
from icm_wrapper import ICMWrapper
from icm_callback import ICMTrainingCallback
from stable_baselines3 import SAC
# Import evaluation callback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
# Import callback list
from stable_baselines3.common.callbacks import CallbackList
import gymnasium as gym

# # Create an argument parser that parses the seed and whether to use ICM or not
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_icm", action="store_true")
args = parser.parse_args()


# # Load the controller config
# controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

# env = suite.make(
#     "Hanoi",
#     robots="Kinova3",
#     controller_configs=controller_config,
#     has_renderer=False,
#     has_offscreen_renderer=True,
#     horizon=100000,
#     use_camera_obs=False,
#     render_camera="robot0_eye_in_hand",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
#     random_reset=True,
# )

# # Wrap the environment
# env = GymWrapper(env)
# env = PickPlaceWrapper(env)

# Create a double inverted pendulum environment
env = gym.make("InvertedDoublePendulum-v2")
eval_env = gym.make("InvertedDoublePendulum-v2")

env.reset(seed=args.seed)
eval_env.reset(seed=args.seed)
# Initialize the ICM and the environment

if args.use_icm:
    feature_dim = 4
    icm = ICM(env, feature_dim, eta=10)
    env = ICMWrapper(env, icm)


# Initialize the SAC model
sac = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./icm_sac_tensorboard/", seed=args.seed)

if args.use_icm:
    # Initialize the ICM training callback
    icm_training_callback = ICMTrainingCallback(icm)

    # Evaluate the model every 1000 steps
    eval_callback = EvalCallback(eval_env, best_model_save_path="./icm_sac_logs/",
                                 log_path="./icm_sac_logs/", eval_freq=1000, n_eval_episodes=50,
                                 deterministic=True, render=False)

    # Create a callback list
    callback = CallbackList([icm_training_callback, eval_callback])

    # Train the SAC model
    sac.learn(total_timesteps=100000, callback=callback, log_interval=30)
else:
    # Evaluate the model every 1000 steps
    eval_callback = EvalCallback(eval_env, best_model_save_path="./icm_sac_logs/",
                                log_path="./icm_sac_logs/", eval_freq=1000, n_eval_episodes=50,
                                deterministic=True, render=False)

    # Train the SAC model
    sac.learn(total_timesteps=100000, callback=eval_callback, log_interval=30)