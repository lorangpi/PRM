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
from prm.state import State
#from HDDL.planner import *
from prm.plan_wrapper_numerical import PlanWrapper
from HER_wrapper import HERWrapper
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action
from prm.action import replace_actions_in_domain, numerical_operator_learner, update_action_cost, merge_actions, split_action #load_action_from_file, restrict_action,

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

# Create the environment
env = suite.make(
    #"PickPlaceCan",
    #"PickPlaceCanNovelties",
    #"Elevated",
    #"Obstacle",
    #"Door",
    #"Hole",
    "Locked",
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
env = ReachWrapper(env, dense_reward=False, augmented_obs=True)
task_goal = {"at(can,drop)":1, "grasped(can)":True,}
detector = RoboSuite_PickPlace_Detector(env, grid_size=200)
env = PlanWrapper(env, task_goal=task_goal, actions=[], detector=detector, num_timesteps=10000000)
#env = HERWrapper(env, symbolic_goal=False)

device = Keyboard()
env.viewer.add_keypress_callback(device.on_press)

# Define the detector
#detector = RoboSuite_PickPlace_Detector(env)
#env = PlanWrapper(env, task_goal="(at can drop)", detector=detector)
#Initialize device control
device.start_control()

# Load the model
#model = SAC.load("/home/lorangpi/PRM/operator_learners/results/prm_icm/sac_augmented_dense_False_seed_0/prm_new/Elevated/models/best_model.zip", env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
model = SAC.load("/home/lorangpi/PRM/operator_learners/best_model.zip", env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})

print("\nResetting environment\n")
obs, _ = env.reset()
print("Environment reset\n")

for episode in range(10):
    for i in range(1000):
        #action = env.action_space.sample()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=True, as_grid=True)
        if i % 100 == 0:
            print(state, reward)
        #print(state)
        #if not(state["grasped(can)"]):
        #    print("FAILED - FAILED - FAILED - ")
        #    break
        #print(detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False))
        if terminated or truncated or i == 499:
            env.reset()
            #env.close_renderer()
            #sys.exit()
            #obs, _ = env.reset()
            break

start = True
actions = []
state = State(detector, numerical=True)
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
    print("Step")
    print("Action: ", action)
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except:
        obs, reward, done, info = env.step(action)

    new_state = State(detector, numerical=True)

    # if start:
    #     #state = detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
    #     state = State(detector)
    #     generate_pddls(state, "(at can drop)")
    #     plan, game_action_set = call_planner("domain", "problem") # get a plan
    #     print(plan)
    #     #detector.display_state(state)
    #     start = False

    """
    if env.light_on:
        learned_action = numerical_operator_learner(state.grounded_predicates, new_state.grounded_predicates, detector.obj_types, predicates_type=detector.predicate_type, object_generalization=detector.object_generalization, name="a")
        learned_action = update_action_cost(learned_action, cost=1)
        known_action = False
        for action in actions:
            if learned_action == action:
                known_action = True
        if learned_action.effects != {} or learned_action.numerical_effects != {} or len(learned_action.function_effects.keys()) > 1 and not known_action:
            print("State: ", state)
            print("Learned Action: ", learned_action._to_pddl())
            learned_actions_list = split_action(learned_action)
            for learned_action in learned_actions_list:
                # Check if the learned action is not equal to any of the actions in the set of actions
                print("Split Action: ", learned_action._to_pddl())
                exist = False
                for action in actions:
                    if learned_action == action:
                        action._cheaper_cost_(learned_action)
                        print("Action Exist: ", action._to_pddl())
                        exist = True
                        continue
                    # Test is the action already has the same effects, then keep the one with the least preconditions 
                    # effects, numerical_effects ad functions_effects are dictionnary
                    elif learned_action._same_effects_(action):
                        merged_action = merge_actions(learned_action, action)
                        if merged_action is not False:
                            merged_action._is_weaker_(action)
                            merged_action.name = action.name
                            actions.remove(action)
                            actions.append(merged_action)
                            print("Action Merged: ", merged_action._to_pddl())
                            exist = True
                        elif learned_action._is_weaker_(action):
                            actions.remove(action)
                            learned_action.name = action.name
                            actions.append(learned_action)
                            print("Action Replaced: ", learned_action._to_pddl())
                            exist = True
                        continue
                if not exist:
                    #print("NAME ", learned_action.name)
                    #print("EFFECTS: ", learned_action.effects)
                    #print("NUM EFFECTS: ", learned_action.numerical_effects)
                    #print("F EFFECTS: ", learned_action.function_effects)
                    #print("PRE: ", learned_action.preconditions)
                    #print()
                    actions.append(learned_action)
            for action in actions:
                print("ACTION LIST")
                print(action._to_pddl())
                print()
    state = State(detector, numerical=True)#detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
    #print("State: ", state)
    #print("New State: ", new_state)
    #print("")
    """
  
    print("Obs: {}\n\n".format(obs))
    env.render()