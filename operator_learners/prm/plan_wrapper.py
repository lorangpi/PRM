import gymnasium as gym
import numpy as np
import hashlib
import json
from prm.state import State
from prm.reward_machine import RewardMachine
from prm.action import load_action_from_file, restrict_action, replace_actions_in_domain, numerical_operator_learner, update_action_cost
from prm.planner import *

class PlanWrapper(gym.Wrapper):
    # This class builds is a wrapper for a gym.Goal environment that utilizes a symbolic detector function, and a symbolic planner given a path the to pddl domain file
    # During training, the detector is used to determine the current state of the environment
    # Also during training, the planner is used to generate a plan to reach the goal state
    # If the plan is not empty, the current state as detected by the detector is added in the set of desired goals

    def __init__(self, env, task_goal, actions, detector=None, domain="generated_domain", problem="generated_problem"):
        super().__init__(env)
        self.env = env
        if detector is None:
            self.detector = env.detector
        else:
            self.detector = detector
        self.domain = domain
        self.problem = problem
        self.desired_goals = [task_goal]
        #self.desired_goal = task_goal
        self.task_goal = task_goal
        self.no_path_set_hashes = []
        self.goal_set_hashes = []
        self.state_transitions_hashes = []
        self.reward_machine = None
        self.plan_counter = 0
        self.reset_plan = 10
        state = State(self.detector)
        generate_pddls(state, goal=State(self.detector, task_goal)._to_pddl(), filename="problem_dummy")
        if actions is None:
            self.actions = load_action_from_file("./PDDL_files/domain.pddl", "./PDDL_files/problem_dummy.pddl")
        else:
            self.actions = actions
        print("Actions: ", [action.name for action in self.actions])
        self.reset()

    def update_actions_goals(self):
        self.transition_cost += 1
        state = State(self.detector)
        if state.compare(self.memory_state) != []:
            state_hash = self.hash_state(state)
            if state_hash + self.memory_state_hash not in self.state_transitions_hashes:
                self.state_transitions_hashes.append(state_hash + self.memory_state_hash)
                learned_actions = numerical_operator_learner(self.memory_state.grounded_predicates, state.grounded_predicates, self.detector.obj_types, predicates_type=self.detector.predicate_type, name="a" + str(len(self.actions)))
                learned_actions = update_action_cost(learned_actions, cost=self.transition_cost)
                self.actions.append(learned_actions)
            # if state_hash not in self.no_path_set_hashes and state_hash not in self.goal_set_hashes:
            #     # The planner will return a plan to reach the task goal, if a plan it exists, else False
            #     planner.generate_pddls(state, self.task_goal._to_pddl(), filename=self.problem)
            #     plan, _ = planner.call_planner(self.domain, self.problem)
            #     # print("The plan is: ", plan)
            #     if plan != False:
            #         self.desired_goals.append(state)
            #         self.goal_set_hashes.append(state_hash)
            #     else:
            #         self.no_path_set_hashes.append(state_hash)
            self.memory_state = state
            self.memory_state_hash = state_hash
            self.transition_cost = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.update_actions_goals()
        state = State(self.detector)
        reward = self.reward_machine.get_reward(state)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        self.transition_cost = 0
        obs, info = self.env.reset(seed=seed, **kwargs)
        if self.plan_counter == 0:
            self.desired_goal = State(self.detector, np.random.choice(self.desired_goals))
        self.reset_state = State(self.detector)
        self.memory_state = State(self.detector)
        self.memory_state_hash = self.hash_state(self.memory_state)
        replace_actions_in_domain("./PDDL_files/domain.pddl", self.actions)
        # Randomly choose a desired goal from the set of desired goals every each 5 resets
        # if self.plan_counter % self.reset_plan == 0 or self.plan_counter == 0:
        #     self.desired_goal = State(self.detector, np.random.choice(self.desired_goals))
        #     print("ARRAY DESIRED GOAL: ", self.desired_goal)
        #     self.reset_state = State(self.detector)
        #     self.memory_state = State(self.detector)
        #     self.memory_state_hash = self.hash_state(self.memory_state)
        #     replace_actions_in_domain("./PDDL_files/domain.pddl", self.actions)
        self.reward_machine = self.generate_reward_machine()
        self.plan_counter += 1
        return obs, info

    def generate_reward_machine(self, state=None, goal=None):
        #(:functions (total-cost))
        if state is None:
            state = self.reset_state
        if goal is None:
            goal = self.desired_goal
        # Generate a reward machine for the current desired goal
        #print("PDDL Goal: ", goal._to_pddl())
        generate_pddls(state, goal=goal._to_pddl(), filename="problem_dummy")
        plan, _ = call_planner("domain_dummy", "problem_dummy")
        #print("Goal: ", goal.grounded_predicates)
        print("Plan: ", plan)
        if plan == False:
                return False
        try:
            if type(float(plan[-1])) == float:
                plan.pop()
        except:
            pass
        return RewardMachine(plan, self.actions, self.reset_state)

    def hash_state(self, state):
        # This function will hash the state of the environment
        state_str = json.dumps(state.grounded_predicates, sort_keys=True)  # Convert dict to string
        state_bytes = state_str.encode()  # Convert string to bytes
        hash_object = hashlib.sha256(state_bytes)  # Hash bytes
        hex_dig = hash_object.hexdigest()  # Get hexadecimal string representation of hash
        return hex_dig





