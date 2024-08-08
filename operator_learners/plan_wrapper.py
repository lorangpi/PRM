import gymnasium as gym
import numpy as np
import hashlib
import json
from state import State
from planner import *

class PlanWrapper(gym.Wrapper):
    # This class builds is a wrapper for a gym.Goal environment that utilizes a symbolic detector function, and a symbolic planner given a path the to pddl domain file
    # During training, the detector is used to determine the current state of the environment
    # Also during training, the planner is used to generate a plan to reach the goal state
    # If the plan is not empty, the current state as detected by the detector is added in the set of desired goals

    def __init__(self, env, sub_goal, task_goal, detector=None, domain="generated_domain", problem="generated_problem"):
        super().__init__(env)
        self.env = env
        if detector is None:
            self.detector = env.detector
        else:
            self.detector = detector
        self.domain = domain
        self.problem = problem
        if type(sub_goal) is dict:
            sub_goal = [sub_goal for sub_goal in sub_goal.keys()]
        elif type(sub_goal) is str:
            sub_goal = [sub_goal]
        else:
            sub_goal = sub_goal
        self.desired_goals = sub_goal
        self.task_goal = task_goal
        self.no_path_set_hashes = []
        self.goal_set_hashes = []
        self.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        state = State(self.detector)
        if state.compare(self.memory_state) != []:
            state_hash = self.hash_state(state)
            if state_hash not in self.no_path_set_hashes and state_hash not in self.goal_set_hashes:
                # The planner will return a plan to reach the task goal, if a plan it exists, else False
                generate_pddls(state, self.task_goal._to_pddl(), filename=self.problem)
                plan, _ = call_planner(self.domain, self.problem)
                # print("The plan is: ", plan)
                if plan != False:
                    self.desired_goals.append(state.grounded_predicates)
                    self.goal_set_hashes.append(state_hash)
                else:
                    self.no_path_set_hashes.append(state_hash)
            self.memory_state = state
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        # Randomly choose a desired goal from the set of desired goals
        self.desired_goal = np.random.choice(self.desired_goals)
        self.env.set_goal(self.desired_goal)
        try:
            obs, info = self.env.reset(seed=seed, **kwargs)
        except TypeError:
            obs, info = self.env.reset()
        except:
            obs = self.env.reset(seed=seed, **kwargs)
            info = {}
        self.memory_state = State(self.detector)
        return obs, info

    def hash_state(self, state):
        # This function will hash the state of the environment
        state_str = json.dumps(state.grounded_predicates, sort_keys=True)  # Convert dict to string
        state_bytes = state_str.encode()  # Convert string to bytes
        hash_object = hashlib.sha256(state_bytes)  # Hash bytes
        hex_dig = hash_object.hexdigest()  # Get hexadecimal string representation of hash
        return hex_dig


