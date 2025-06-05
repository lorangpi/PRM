from prm.state import State

class RewardMachine:
    def __init__(self, plan, actions, initial_state):
        self.plan = plan
        self.current_state_index = 0
        self.total_reward = 1000  # Define the total reward
        self.state_sequence = self.convert_plan_to_states(plan, actions, initial_state)
        self.state_sequence = self.state_sequence[1:]
        self.label_sequence = self.state_to_label(self.state_sequence)
        print("State Sequence = ", [state.grounded_predicates for state in self.state_sequence], "\n")

    def state_to_label(self, state_sequence):
        # Convert the state sequence to a sequence of labels
        label_sequence = []
        for state in state_sequence:
            label = state._to_pddl()
            label_sequence.append(label)
        return label_sequence

    def get_reward(self, state):
        if not(type(state) == dict):
            state = state.grounded_predicates
        # Check if all predicates in the next state in the label sequence are met in the current state
        if self.state_sequence is not None:
            for i in range(self.current_state_index, len(self.state_sequence)):
                next_state = self.state_sequence[i]
                if all(item in state.items() for item in next_state.grounded_predicates.items()):
                    # Calculate the reward for reaching the next state in the sequence
                    # The reward is proportional to the current state index
                    # If the agent jumps a state, it gets the sum of the rewards for the skipped states
                    reward = self.total_reward * sum(range(0 + 1, i + 2))  / sum(range(1, len(self.state_sequence) + 1))
                    self.current_state_index = i + 1
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
    
    def apply_action(self, state, action):
        # Create a copy of the state
        new_state = State(state.detector, init_predicates={})
        # Apply the action's effects to the state
        for effect, value in action.effects.items():
            new_state.grounded_predicates[effect] = value

        # Return the resulting state
        return new_state

    def convert_plan_to_states(self, plan, actions, initial_state):
        # Initialize the state sequence with the initial state
        state_sequence = [initial_state]

        # Initialize the current state as the initial state
        current_state = initial_state

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
            next_state = self.apply_action(current_state, grounded_action)

            # Add the next state to the state sequence
            state_sequence.append(next_state)

            # Update the current state
            current_state = next_state

        # Return the state sequence
        return state_sequence