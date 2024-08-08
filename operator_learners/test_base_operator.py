import argparse
import numpy as np
import sys
import os
import robosuite as suite
import gymnasium
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, HerReplayBuffer
from robosuite.wrappers.gym_wrapper import GymWrapper
from operator_wrapper import PickWrapper, ReachWrapper
from detector import RoboSuite_PickPlace_Detector
from state import State
from HDDL.planner import *
from plan_wrapper import PlanWrapper
from HER_wrapper import HERWrapper
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action


controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

# Create the environment
env = suite.make(
    #"PickPlaceCan",
    "PickPlaceCanNovelties",
    #"Elevated",
    #"Obstacle",
    #"Door",
    #"Hole",
    #"Locked",
    #"Lightoff",
    robots="Kinova3",
    #robots="Fetch",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    horizon=100000000,
    render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
)
env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
env = ReachWrapper(env, dense_reward=True, render_init=True)
env = HERWrapper(env, symbolic_goal=False)

device = Keyboard()
env.viewer.add_keypress_callback(device.on_press)

# Define the detector
detector = RoboSuite_PickPlace_Detector(env)
env = PlanWrapper(env, sub_goal="at(can,drop)", task_goal="(at can drop) (not (picked_up can))", detector=detector)
#Initialize device control
device.start_control()

# Load the model
#model = SAC.load("/home/lorangpi/HyGOAL/operator_learners/models/sac/best_model.zip")
model = SAC.load("/home/lorangpi/HyGOAL/operator_learners/models/ltl/base/best_model.zip", env=env)

# for i in range(500):
#     #action = env.action_space.sample()
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     #print(detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False))
#     if terminated or truncated or i == 499:
#         env.close()
#         env.close_renderer()
#         #sys.exit()
#         #obs, _ = env.reset()
#         break

start = True

while True:
    # Set active robot
    active_robot = env.robots[0]

    # Get the newest action
    action, grasp = input2action(
        device=device, robot=active_robot, active_arm="right", env_configuration="single-arm-opposed"
    )

    # If action is none, then this a reset so we should break
    if action is None:
        break

    # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
    # toggle arm control and / or camera viewing angle if requested

    # Update last grasp
    last_grasp = grasp

    # Fill out the rest of the action space if necessary
    rem_action_dim = env.action_dim - action.size
    if rem_action_dim > 0:
        # Initialize remaining action space
        rem_action = np.zeros(rem_action_dim)
        # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
        action = np.concatenate([action, rem_action])

    elif rem_action_dim < 0:
        # We're in an environment with no gripper action space, so trim the action space to be the action dim
        action = action[: env.action_dim]

    # Step through the simulation and render
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except:
        obs, reward, done, info = env.step(action)
    
    if start:
        #state = detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
        state = State(detector)
        generate_pddls(state, "(at can drop)")
        plan, game_action_set = call_planner("domain", "problem") # get a plan
        print(plan)
        #detector.display_state(state)
        start = False
    
    new_state = State(detector)#detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
    #print("State: ", state)
    #print("New State: ", new_state)
    #print("")
    difference = new_state.compare(state)
    if difference != {}:
        print("Difference: ", difference)
        state = new_state

    #print("Obs: {}\n\n".format(obs))
    env.render()