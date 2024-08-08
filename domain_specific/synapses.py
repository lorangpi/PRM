'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This file are the domain synapses of RAPidL.
# This file aim is to store every domain related information that are not captured by PDDL / HDDL formats. 
# Our goal is to eventually make the structure of RAPidL completely domain agnostic, while when
# necessary, only use the synapses to generate domain dependent structures (as the sate class for instance).

'''

from executor import *
from state import State
from domain_specific.novelties import novelties_info
from domain_specific.coded_executors import *
from domain_specific.detector import *

# domain dependant ids of executors
executors_id_list = ['Reach','Pick','Drop']

# used to select source policy to transfer from whenever the agent needs to train on a novelty pattern
novelty_patterns = {'Reach':[],'Pick':[],'Drop':[],'Nop':[]}

# set up applicator mapping from operators to executors
applicator = {'MOVE':['Reach'],
		'PICK':['Pick'],
		'DROP':['Drop'],
		}

pddl_op ={'MOVE':'move','PICK':'pick','DROP':'drop'}

adaptive_op = ['MOVE','PICK','DROP']

def beta_indicator(operator, env):
	detector = RoboSuite_PickPlace_Detector(env)
	state = State(detector)
	desired_effects = effects(operator)
	return all(item in state.grounded_predicates.items() for item in desired_effects.items())

# set up executor classes
executor1 = Executor_Pick(id='Pick', Beta=beta_indicator, basic=True, mode="Coded")
executor2 = Executor_Drop(id='Drop', Beta=beta_indicator, basic=True, mode="Coded")
executor3 = Executor_RL(id='Reach', alg="her_symbolic", policy="/home/lorangpi/HyGOAL/operator_learners/models/her_symbolic/base/best_model.zip", I={}, Beta=beta_indicator, basic=True)
executors = {'Reach':executor3, 'Pick':executor1, 'Drop':executor2}

def effects(operator):
	if "MOVE" in operator: 
		location = operator.split(" ")[2].lower()
		gripper = operator.split(" ")[3].lower()
		return {f"at(can,{location})": True}
	elif "PICK" in operator:
		obj = operator.split(" ")[1].lower()
		return {f"picked_up({obj})": True}
	elif "DROP" in operator:
		obj = operator.split(" ")[1].lower()
		return {f"grasped({obj})": False}
	elif "nop" in operator:
		return {}

# utility functions
def inverse_dict(e):
	new_dic = {}
	for k,v in e.items():
		for x in v:
			new_dic.setdefault(x,[]).append(k)
	return new_dic

def select_closest_pattern(novelty_list, operator, same_operator=False, use_base_policies=True):
	if novelty_list == None:
		novelty_list = []
	similarity_indicator = 0
	source_policy = None
	associate_source_novelty_pattern = [None]

	# loop through all known executors and their associate novelty pattern (on which they were trained)
	for policy, pattern in zip(list(novelty_patterns), list(novelty_patterns.values())):
		similarity = len(set(pattern) & set(novelty_list))

		if not(use_base_policies) and executors[policy].basic:
			continue

		# if the novelty pattern is the same
		if similarity == similarity_indicator:
			# if the executor is mapped to the operator
			if policy in applicator[operator]:
				# if the previously selected executor on its side is not mapped to operator, then select the current executor (mapped to operator)
				if source_policy not in applicator[operator]:
					source_policy = policy
					associate_source_novelty_pattern = pattern
				# if both previous and current executors are mapped to operator, select the one that has as accomodated on as few novelties not in novelty_list
				else:
					if len(associate_source_novelty_pattern) > len(pattern):
						source_policy = policy
						associate_source_novelty_pattern = pattern
		# if current executor pattern is more similar to novelty_list than previously selected executor
		elif similarity > similarity_indicator:
			# if we want the source policy to be mapped to the operator anyway
			if same_operator:
				# then select current executor only if it is mapped to the operator
				if policy in applicator[operator]:
					source_policy = policy
					associate_source_novelty_pattern = pattern
					similarity_indicator = similarity
			# else select current executor even if it is not mapped to the operator
			else:
				source_policy = policy
				associate_source_novelty_pattern = pattern
				similarity_indicator = similarity
	return source_policy

def select_best_executor(novelty_list, queue, state):

	if novelty_list == None:
		novelty_list = []
	patterns = [novelties_info[item]["pattern"] for item in novelty_list]
	similarity_indicator = 0
	source_executor = queue[0]
	associate_source_novelty_pattern = novelty_patterns[source_executor]

	for executor in queue:
		if I_in_state(executors[executor].I, state):
			similarity = len(set(novelty_patterns[executor]) & set(patterns))
			if similarity == similarity_indicator:
					if len(associate_source_novelty_pattern) > len(novelty_patterns[executor]):
							source_executor = executor
							associate_source_novelty_pattern = novelty_patterns[executor]
			elif similarity > similarity_indicator:
				source_executor = executor
				associate_source_novelty_pattern = novelty_patterns[executor]
				similarity_indicator = similarity

	return source_executor

def I_in_state(I, state):
	# form of I:  I=[[["at","car","l2"],["dir","e"]],[["at","car","l3"],["dir","e"]]]
	if I == None:
		return True
	for set in I:
		I_is_valid = True
		for predicate in set:
			if predicate not in state.grounded_predicates:
				I_is_valid = False
			if not(I_is_valid):
				break
		if I_is_valid:
			return True
	return False