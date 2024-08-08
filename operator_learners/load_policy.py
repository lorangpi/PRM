import argparse
import numpy as np
import os
import robosuite as suite
import gymnasium
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from robosuite.wrappers.gym_wrapper import GymWrapper
from operator_wrapper import PickWrapper, ReachWrapper
from detector import RoboSuite_Detector
from HER_wrapper import HERWrapper
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

# Create the environment
env = suite.make(
    #"PickPlaceCan",
    "Elevated",
    #"Obstacle",
    #"Door",
    #"Locked",
    #"Hole",
    #"Lightoff",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon=100000000,
    render_camera="birdview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
)
env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
env = ReachWrapper(env, dense_reward=True, render_init=True)
env = HERWrapper(env, symbolic_goal=False, augmented_obs=False)

device = Keyboard()
env.viewer.add_keypress_callback(device.on_press)

# Test the environment
obs, _ = env.reset()
detector = RoboSuite_Detector(env, single_object_mode=True, object_to_use='can')
# Initialize device control
device.start_control()

model = SAC.load("/home/lorangpi/data_visu_test/data/her_dense_True_seed_1/2023-12-12_13:21:43/Elevated/models/best_model", env=env)

for i in range(1000):
    action = model.predict(obs, deterministic=True)[0]
    #print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    #print(detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False))
    if terminated or truncated:
        obs, _ = env.reset()

