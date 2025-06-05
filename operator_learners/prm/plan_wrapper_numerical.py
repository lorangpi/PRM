import gymnasium as gym
import numpy as np
import hashlib
import json
import sys
from prm.state import State
from prm.reward_machine_numerical import RewardMachine
from prm.action import replace_actions_in_domain, numerical_operator_learner, update_action_cost, merge_actions, split_action #load_action_from_file, restrict_action,
from prm.planner_numerical import *

class PlanWrapper(gym.Wrapper):
    # This class builds is a wrapper for a gym.Goal environment that utilizes a symbolic detector function, and a symbolic planner given a path the to pddl domain file
    # During training, the detector is used to determine the current state of the environment
    # Also during training, the planner is used to generate a plan to reach the goal state
    # If the plan is not empty, the current state as detected by the detector is added in the set of desired goals

    def __init__(self, env, task_goal, constraint, actions, num_timesteps=None, detector=None, domain="domain_numerical", problem="problem_dummy", pddl_path="./PDDL_files/"):
        super().__init__(env)
        self.env = env
        if detector is None:
            self.detector = env.detector
        else:
            self.detector = detector
        self.num_timesteps = num_timesteps
        if num_timesteps is not None:
            self.plan_counter_log = [int(2**i) for i in range(int(np.log2(num_timesteps)))]
            # Add 2 to all the elements of the plan counter log
            self.plan_counter_log = [i + 2 for i in self.plan_counter_log]
        # PDDL files paths
        self.base_domain = "./PDDL_files/" + domain + ".pddl"
        self.base_problem = "./PDDL_files/" + problem + ".pddl"
        self.new_domain = pddl_path + os.sep + domain + "_new.pddl"
        self.new_domain_name = domain + "_new"
        self.new_problem = pddl_path + os.sep + problem + ".pddl"
        self.new_problem_name = problem + "_new"
        self.pddl_path = pddl_path
        # Goal and state hashing lists initialization
        self.desired_goals = [task_goal]
        self.task_goal = task_goal
        self.constraint = constraint
        self.no_path_set_hashes = []
        self.goal_set_hashes = []
        self.state_transitions_hashes = []
        self.reward_machine = None
        self.plan_counter = 0
        self.reset_plan = 20
        state = State(self.detector, numerical=True)
        generate_pddls(state, goal=State(self.detector, init_predicates=task_goal, numerical=True)._to_pddl(), filename=self.new_problem_name, pddl_dir=pddl_path)
        #if actions is None:
        #    self.actions = load_action_from_file(self.domain, self.problem)
        #else:
        #    self.actions = actions
        self.actions = actions
        self.action_counter = len(self.actions)
        print("Actions: ", [action.name for action in self.actions])
        self.reset()

    def update_actions_goals(self):
        self.transition_cost += 1
        state = State(self.detector, numerical=True)
        
        if state.compare(self.memory_state) != {}:
            state_hash = state.__hash__()
            if state_hash + self.memory_state_hash not in self.state_transitions_hashes:
                self.state_transitions_hashes.append(state_hash + self.memory_state_hash)
                learned_action = numerical_operator_learner(self.memory_state.grounded_predicates, state.grounded_predicates, self.constraint, self.detector.obj_types, predicates_type=self.detector.predicate_type, object_generalization=self.detector.object_generalization, name="a" + str(self.action_counter))
                # add 3 cost for each effect in the action
                action_cost = self.transition_cost + 5 * len(learned_action.effects) + 5 * len(learned_action.numerical_effects) + 5 * len(learned_action.function_effects)
                learned_action = update_action_cost(learned_action, cost=action_cost)
                if learned_action.effects != {} or learned_action.numerical_effects != {} or len(learned_action.function_effects.keys()) > 1:
                    learned_actions_list = split_action(learned_action, constraint=self.constraint)
                    #learned_actions_list = [learned_action]
                    for learned_action in learned_actions_list:
                         # Check if the learned action is not equal to any of the actions in the set of actions
                        exist = False
                        for action in self.actions:
                            if learned_action == action:
                                action._cheaper_cost_(learned_action)
                                exist = True
                                break
                            # Test is the action already has the same effects, then keep the one with the least preconditions 
                            # effects, numerical_effects ad functions_effects are dictionnary
                            elif learned_action._same_effects_(action):
                                merged_action = merge_actions(learned_action, action, constraint=self.constraint)
                                if merged_action is not False:
                                    merged_action._is_weaker_(action)
                                    merged_action.name = action.name
                                    self.actions.remove(action)
                                    self.actions.append(merged_action)
                                    exist = True
                                elif learned_action._is_weaker_(action):
                                    self.actions.remove(action)
                                    learned_action.name = action.name
                                    self.actions.append(learned_action)
                                    exist = True
                                break
                        if not exist:
                            self.actions.append(learned_action)
                            self.action_counter += 1
            self.memory_state = state
            self.memory_state_hash = state_hash
            self.transition_cost = 0
        

    def filter_actions(self):
        # This function will update the entire set of actions to make sure no duplicatas are present
        # If any action is exactly the same as the action, then the action with the least cost is kept
        # If any action in the set of actions has the same effects as the action, then the action with the least preconditions is kept
        # If the action is not in the set of actions, then it is added
        new_actions = []
        for action in self.actions:
            unique = True
            for new_action in new_actions:
                if action == new_action:
                    new_action._cheaper_cost_(action)
                    unique = False
                    break
                elif action._same_effects_(new_action):
                    merged_action = merge_actions(action, new_action, constraint=self.constraint)
                    if merged_action is not False:
                        merged_action._is_weaker_(new_action)
                        merged_action.name = new_action.name
                        new_actions.remove(new_action)
                        action = merged_action
            if unique:
                new_actions.append(action)
        self.actions = new_actions

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.update_actions_goals()
        state = State(self.detector, numerical=True)
        generated_reward = self.reward_machine.get_reward(state)
        reward = max(reward, generated_reward)
        if not(state.grounded_predicates['grasped(can)']):
            reward = -2
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        self.transition_cost = 0
        obs, info = self.env.reset(seed=seed, **kwargs)
        self.reset_state = State(self.detector, numerical=True)
        self.memory_state = State(self.detector, numerical=True)
        self.memory_state_hash = self.memory_state.__hash__()
        # Generate a logarithmic frequency list of reward machines generation
        if self.num_timesteps is not None:
            if self.plan_counter == 0:
                print("Plan Counter Log: ", self.plan_counter_log)
            if self.plan_counter == 0 or self.plan_counter in self.plan_counter_log:
                print("Generating New Reward Machine. Plan Counter: ", self.plan_counter )
                self.desired_goal = State(self.detector, init_predicates=np.random.choice(self.desired_goals), numerical=True)
                self.filter_actions()
                print("Action Split and Merged.")
                replace_actions_in_domain(self.base_domain, self.new_domain, self.actions)
                # Pop the first element of the plan counter log
                self.plan_counter_log.pop(0)
                self.reward_machine = self.generate_reward_machine()
        else:
            if self.plan_counter == 0 or self.plan_counter % self.reset_plan == 0:
                self.desired_goal = State(self.detector, init_predicates=np.random.choice(self.desired_goals), numerical=True)
                self.filter_actions()
                replace_actions_in_domain(self.base_domain, self.new_domain, self.actions)
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
        print("Planning...")
        generate_pddls(state, goal=goal._to_pddl(), filename=self.new_problem_name, pddl_dir=self.pddl_path)
        plan, _ = call_planner(self.new_domain_name, self.new_problem_name, pddl_dir=self.pddl_path)
        
        #plan = []
        print("Goal: ", goal.grounded_predicates)
        print("State: ", state.grounded_predicates)
        print("Plan: ", plan)
        return RewardMachine(plan, actions=self.actions, goal=goal, initial_state=state)

    def hash_state(self, state):
        # This function will hash the state of the environment
        state_str = json.dumps(state.grounded_predicates, sort_keys=True)  # Convert dict to string
        state_bytes = state_str.encode()  # Convert string to bytes
        hash_object = hashlib.sha256(state_bytes)  # Hash bytes
        hex_dig = hash_object.hexdigest()  # Get hexadecimal string representation of hash
        return hex_dig