'''
Authors Pierrick Lorang,
Emails: pierrick.lorang@tufts.edu

# This file is the high level state representation of RAPidL.
# This file enables to store an object-oriented version of the PDDL/HDDL format knowledge. It eases the computation of on the fly inferences. 
# Eventually this file should be auto generated from a PDDL/HDDL domain file, and should be used to improve the PDDL/HDDL knowledge using continual abstraction of information.

'''
import numpy as np
from prm.generate_pddl_numerical import rewrite_predicate

class State:
    def __init__(self, detector, numerical=False, init_predicates=[], array_observation=None):
        self.detector = detector
        self.numerical = numerical
        self.step = 0
        self.failed_action = None
        self.domain = "domain"
        if array_observation is not None:
            #print("Array Observation = ", array_observation)
            self.grounded_predicates = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=numerical, as_grid=numerical)
            init_predicates = self._to_dict(array_observation)
            #print(init_predicates)
        if init_predicates == []:
            init_predicates = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=numerical, as_grid=numerical)
        elif type(init_predicates) == dict:
            init_predicates = init_predicates
        elif type(init_predicates) == str:
            init_predicates = {init_predicates: True}
        elif type(init_predicates) == list:
            init_predicates = {predicate: True for predicate in init_predicates}
        else:
            raise ValueError("init_predicates should be a dictionary, a string, or a list")
        self.init_predicates = init_predicates
        self.types = detector.types
        self.entities = [[k, v] for k, v in detector.obj_types.items()]
        #self.predicates = detector.get_ungrounded_predicates()
        self.constants = detector.get_static_predicates()
        self.grounded_predicates = init_predicates
        self.reset()

    def reset(self):
        self.grounded_predicates = self.init_predicates

    def compare(self, other_state):
        difference = {}
        for predicate, value in self.grounded_predicates.items():
            if predicate not in other_state.grounded_predicates or other_state.grounded_predicates[predicate] != value:
                difference[predicate] = value
        return difference

    def satisfies(self, goal):
        return all(item in self.grounded_predicates.items() for item in goal.grounded_predicates.items())

    def __eq__(self, other):
        if isinstance(other, State):
            return self.compare(other) == {}
        return False

    def _to_array(self):
        return np.asarray([float(v) for k, v in sorted(self.grounded_predicates.items())])

    def _to_dist_array(self):
        # Replace True values in the dict with 0.0 and False values with random values between 1 and self.detector.max_distance
        return np.asarray([0.0 if v else np.random.uniform(1, self.detector.max_distance) for k, v in sorted(self.grounded_predicates.items())])

    def _to_dict(self, array):
        return dict(zip(sorted(self.grounded_predicates.keys()), map(bool, array)))

    def _to_pddl(self):
        predicates = list()
        for key, value in self.grounded_predicates.items():
            predicates.append(rewrite_predicate((key, value), jump_line=False, write_negations=True))
        pddl_state = "".join(predicates)
        return pddl_state

    def __str__(self):
        return str(self.grounded_predicates)
    
    # Hash of the class dependind on the grounded predicates
    def __hash__(self):
        return hash(frozenset(self.grounded_predicates.items()))
    
    def __copy__(self):
        new_state = State(self.detector, self.init_predicates, self.numerical)
        return new_state