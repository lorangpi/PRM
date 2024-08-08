'''
Authors Pierrick Lorang,
Emails: pierrick.lorang@tufts.edu

# This file is the high level state representation of RAPidL.
# This file enables to store an object-oriented version of the PDDL/HDDL format knowledge. It eases the computation of on the fly inferences. 
# Eventually this file should be auto generated from a PDDL/HDDL domain file, and should be used to improve the PDDL/HDDL knowledge using continual abstraction of information.

'''
from generate_pddl import rewrite_predicate

class State:
    def __init__(self, detector, init_predicates=[]):
        self.detector = detector
        self.step = 0
        self.failed_action = None
        self.domain = "domain"
        if init_predicates == []:
            init_predicates = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.init_predicates = init_predicates
        self.types = detector.types
        self.entities = [[k, v] for k, v in detector.obj_types.items()]
        self.predicates = detector.get_ungrounded_predicates
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
        return hash(str(self.grounded_predicates))
    