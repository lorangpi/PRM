from prm.state import State

class RewardMachine:
    def __init__(self, plan, actions, goal, initial_state):
        self.plan = plan
        self.current_state_index = 0
        self.visited_states = []
        self.total_reward = 800  # Define the total reward
        self.plan = plan
        self.goal = goal
        self.reward = -1
        if plan != False:
            self.state_sequence = self.convert_plan_to_states(plan, actions, initial_state)
            #self.state_sequence = self.state_sequence[1:]
            self.label_sequence = self.state_to_label(self.state_sequence)
            print("State Sequence = ", [state._to_pddl() for state in self.state_sequence], "\n")

    def state_to_label(self, state_sequence):
        # Convert the state sequence to a sequence of labels
        label_sequence = []
        for state in state_sequence:
            label = state._to_pddl()
            label_sequence.append(label)
        return label_sequence

    def reset_reward(self):
        self.current_state_index = 0
        self.reward = -1

    def get_reward(self, state):
        # If the agent is not following a plan, return the total reward if the goal is satisfied
        if self.plan == False:
            if state.satisfies(self.goal):
                return self.total_reward
            else:
                return -1
        if self.state_sequence is not None:
            for i in range(self.current_state_index, len(self.state_sequence)):
                next_state = self.state_sequence[i]
                if state.satisfies(next_state):
                    reward = -1 + 0.8 * sum(range(0 + 1, i + 2))  / sum(range(1, len(self.state_sequence) + 1))
                    return reward
        # If not all predicates are met, return -1
        return -1


    def generate_ltl_formula(self):
        # Initialize the LTL formula as an empty string
        ltl_formula = ""

        # Iterate over the labels in the sequence
        for i in range(len(self.label_sequence)):
            # For each label, add a "next" operator and the label to the LTL formula
            ltl_formula += "X(" + str(self.label_sequence[i]) + ")"

            # If this is not the last label, add an "and" operator to the LTL formula
            if i != len(self.label_sequence) - 1:
                ltl_formula += " && "

        # Return the LTL formula
        return ltl_formula
    
    def apply_action(self, state, track_predicates, action):
        # Create a copy of the state
        new_state = {}
        # Apply the action's effects to the state
        for effect, value in action.effects.items():
            new_state[effect] = value
            track_predicates[effect] = value
        for effect, value in action.numerical_effects.items():
            new_state[effect] = value
            track_predicates[effect] = value
        for effect, value in action.function_effects.items():
            if 'total-cost' in effect:
                continue
            if effect in new_state:
                new_state[effect] = state.grounded_predicates[effect] + value
            else:
                try:
                    new_state[effect] = track_predicates[effect] + value
                    track_predicates[effect] = new_state[effect]
                except KeyError:
                    pass
        # Return the resulting state
        return State(state.detector, init_predicates=new_state), track_predicates

    def convert_plan_to_states(self, plan, actions, initial_state):
        # Initialize the state sequence with the initial state
        #state_sequence = [initial_state]
        state_sequence = []

        # Initialize the current state as the initial state
        current_state = initial_state
        track_predicates = initial_state.grounded_predicates.copy()

        #"""
        # Iterate over the actions in the plan
        for action_str in plan:
            # Extract the action name from the action string and convert it to lowercase
            action_name = action_str.split()[0].lower()

            # Find the action with the extracted name
            action = next((a for a in actions if a.name == action_name), None)
            if action is None:
                raise ValueError(f"Action {action_name} not found")

            grounded_action = action.ground_action(action_str.lower())
            # Apply the action to the current state
            current_state, track_predicates = self.apply_action(current_state, track_predicates, grounded_action)

            if '(not (grasped can ))' in current_state._to_pddl():
                pass
            elif '(grasped can)' in current_state._to_pddl():
                pass
            else:
                # Add the next state to the state sequence
                state_sequence.append(current_state)
        return state_sequence