'''
# authors: Shivam Goel; Pierrick Lorang
# email: shivam.goel@tufts.edu; pierrick.lorang@tufts.edu

# This file is the Brain of RAPidL.
# It is the main file which talks to the planner, learner and the game instance.

Important References

'''
from __future__ import print_function
import copy, sys, json, os, gym
import domain_specific.synapses as domain_synapses
from domain_specific.create_env import create_env
from importlib import reload
from datetime import datetime
from learner import Learner
from generate_pddl import *
from planner import *
from domain_specific.novelties import *

class Brain:
	def __init__(self, rl_alg="her_symbolic", steps_num=500_000, eval_freq=20000, env_id=None, domain="domain", verbose=False, DATA_DIR="", transfer=True, seed=0, test=False, use_base_policies=True):
		'''
		This is the constructor method for the class. It initializes various attributes of the class instance.

		It takes the following parameters:
		- rl_alg: The reinforcement learning algorithm to use. Default is "her_symbolic", i.e. Hindsight Experience Replay (HER) on top of SAC with symbolic goal space.
		- steps_num: The number of steps for the experiment.
		- domain: The domain of the experiment. Default is "domain".
		- verbose: A flag indicating whether to print verbose output. Default is False.
		- DATA_DIR: The directory for storing data. Default is an empty string.
		- transfer: A flag indicating whether to transfer learning. Default is True.
		- seed: The seed for random number generation. Default is 0.
		- test: A flag indicating whether this is a test run. Default is False.
		- use_base_policies: A flag indicating whether to use base policies. Default is True.

		It initializes the following attributes:
		- domain: Set to the domain parameter.
		- learner: Initialized to None.
		- steps_num: Set to the steps_num parameter.
		- learned_policies_dict: A dictionary for storing learned policies. Initialized with keys 'goForward', 'turnLeft', 'turnRight' and empty dictionaries as values.
		- task_goal_episode: An empty dictionary.
		- verbose: Set to the verbose parameter.
		- verboseprint: A function that prints output if verbose is True, otherwise does nothing.
		- DATA_DIR: Set to the DATA_DIR parameter.
		- transfer: Set to the transfer parameter.
		- seed: Set to the seed parameter.
		- use_base_policies: Set to the use_base_policies parameter.
		- env_id: Initialized to None.
		- env: Initialized to None.
		- test: Set to the test parameter.
		- novelty_list: Initialized to None.
		'''
		self.rl_alg = rl_alg
		self.domain = domain
		self.learner = None
		self.steps_num = steps_num
		self.eval_freq = eval_freq
		self.learned_policies_dict = {key: {} for key in applicator.keys()} # store failed action:learner_instance object
		self.task_goal_episode = {}
		self.verbose = verbose
		self.verboseprint = print if verbose else lambda *a, **k: None
		self.DATA_DIR = DATA_DIR
		self.transfer = transfer
		self.seed = seed
		self.use_base_policies = use_base_policies
		self.env_id = env_id,
		self.env = None
		self.test = test
		self.novelty_list = None

	def generate_env(self, env_id, novelty_list=None, dense_reward=False, alg=None, render=False, only_train=False):
		'''
		This method is responsible for generating a new environment for the experiment. It takes an environment ID, a list of novelties, 
		a reset location and direction, a render flag, and a training flag as parameters. 

		If the environment does not exist or the novelty list has changed, it closes the existing environment (if any) and creates a new one. 
		It sets the render, reset location and direction, and environment ID attributes of the instance. 
		It then injects the novelties into the environment by creating a new instance of the novelty wrapper for each novelty and adding it to 
		the environment. 
		Finally, it seeds the environment and sets the environment and novelty list attributes of the instance.
		'''
		if alg == None:
			alg = self.rl_alg
		if novelty_list == []:
			novelty_list = env_id

		if self.env == None or novelty_list != self.novelty_list:

			if self.env != None:
				self.close_env()

			self.render = render
			self.env_id = env_id

			#if not(only_train):

			print("\nCreating execution env.")
			# get environment instances before injecting novelty
			self.verboseprint("env id is: ",env_id)
			env, detector, obs = create_env(env_id, render=render, dense_reward=dense_reward, alg=alg, seed=self.seed)
			self.env = env
			self.detector = detector
			self.obs = obs
		self.novelty_list = novelty_list

	def close_env(self):
		'''
		This method is responsible for closing the environment associated with the instance of the class. 
		If the environment exists (i.e., is not None), it calls the close method on the environment and then sets 
		the environment attribute to None. It also prints a message indicating that the environment is being closed.
		'''
		if self.env != None:
			self.env.close()
			self.env = None
			print("Closing execution env.")

	def save_infos(self):
		'''
		This method is responsible for saving data from the current experiment. It saves the applicator, novelty patterns, 
		policies paths, executors ID list, and learned policies into JSON files. It first creates a dictionary of policies paths 
		by iterating over the applicator items. Then, it serializes each piece of data into JSON format and writes it into a 
		corresponding file in the DATA_DIR directory. It repeats the same process for the main_folder directory.
		'''
		policies_paths = {}
		for operator, executors_set in applicator.items():
			for executor in executors_set:
				policies_paths.update(executors[executor].path_to_json())
		j = json.dumps(applicator, indent=4)
		with open(self.DATA_DIR+'applicator.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(novelty_patterns, indent=4)
		with open(self.DATA_DIR+'novelty_patterns.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(policies_paths, indent=4)
		with open(self.DATA_DIR+'policies_paths.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(executors_id_list, indent=4)
		with open(self.DATA_DIR+'executors_id_list.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(self.learned_policies_dict, indent=4)
		with open(self.DATA_DIR+'learned_policies.json', 'w') as f:
			print(j, file=f)

		main_folder, _ = os.path.split(self.DATA_DIR[:-1])
		if not(main_folder.endswith('/')):
			main_folder += '/'
		j = json.dumps(applicator, indent=4)
		with open(main_folder+'applicator.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(novelty_patterns, indent=4)
		with open(main_folder+'novelty_patterns.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(policies_paths, indent=4)
		with open(main_folder+'policies_paths.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(executors_id_list, indent=4)
		with open(main_folder+'executors_id_list.json', 'w') as f:
			print(j, file=f)
		j = json.dumps(self.learned_policies_dict, indent=4)
		with open(main_folder+'learned_policies.json', 'w') as f:
			print(j, file=f)

	def load_infos(self):
		'''
		This method is responsible for loading data from previous experiments. It loads the applicator, novelty patterns, 
		policies paths, executors ID list, and learned policies from JSON files. It then creates Executor objects for each 
		executor ID and adds them to the executors dictionary. After loading all the data, it calls the reload_synapses 
		method and prints out the loaded information.
		'''
		self.verboseprint('\n_____________________________________________________________________________________________________________________')
		self.verboseprint('\n\n\t\t\t\t RELOADING SYNAPSES FROM PREVIOUS EXPERIMENT :')
		main_folder, _ = os.path.split(self.DATA_DIR[:-1])
		if not(main_folder.endswith('/')):
			main_folder += '/'
		self.verboseprint('\t\t\t\t',main_folder,'\n\n')
		f = open(main_folder+'applicator.json')
		domain_synapses.applicator = json.load(f)
		f = open(main_folder+'novelty_patterns.json')
		domain_synapses.novelty_patterns = json.load(f) 
		f = open(main_folder+'policies_paths.json')
		policies_paths = json.load(f)
		f = open(main_folder+'executors_id_list.json')
		domain_synapses.executors_id_list = json.load(f)
		f = open(main_folder+'learned_policies.json')
		self.learned_policies_dict = json.load(f)

		for exec_id in domain_synapses.executors_id_list:
			print("Loading executor: ", policies_paths[exec_id])
			executor = Executor(id=exec_id, policy=policies_paths[exec_id], Beta=beta_indicator, basic=True)
			ex_dict = {exec_id:executor}
			domain_synapses.executors.update(ex_dict)
		
		self.reload_synapses()

		self.verboseprint("\n\nLoading executors_id_list information.\n", executors_id_list)
		self.verboseprint("\n\nLoading novelty_patterns information.\n", novelty_patterns)
		self.verboseprint("\n\nLoading applicator information.\n", applicator)
		self.verboseprint("\n\nLoading executors information.\n", executors)
		self.verboseprint("\n\nLoading learned_policies information.\n", self.learned_policies_dict)
		self.verboseprint('_____________________________________________________________________________________________________________________\n')

	def reload_synapses(self):
		'''
		This method is responsible for reloading the synapses (connections) in the brain. It does this by updating the global variables 
		applicator, executors, novelty_patterns, and executors_id_list with the corresponding attributes from the domain_synapses object.

		It assumes that the domain_synapses object has already been initialized and contains the updated synapses.
		'''
		global applicator 
		global executors
		global novelty_patterns
		global executors_id_list

		executors_id_list = domain_synapses.executors_id_list
		novelty_patterns = domain_synapses.novelty_patterns
		applicator = domain_synapses.applicator
		executors = domain_synapses.executors

	def run_brain(self, task_goal, trial=1, max_trials=3, only_eval=False, only_train=False, direct_train=False, failed_operator="MOVE PICK DROP GRIPPER", failure_state=None):
		'''        
		This method is responsible for running the AI agent's trials in the environment. 

		Parameters:
		- task_goal: The task_goal for the agent to perform.
		- trial: The current trial number (default is 1).
		- max_trials: The maximum number of trials (default is 3).
		- only_eval: A flag to indicate if the function should only evaluate (default is False).
		- only_train: A flag to indicate if the function should only train (default is False).
		- direct_train: A flag to indicate if the function should directly train (default is False).

		The method initializes variables and checks training conditions. If training is required, it closes the environment and trains 
		necessary operators. If not, it launches a trial, updates the task_goal count, and prepares the environment. It generates planning 
		data and executes a plan. If the plan fails, it returns the trial status. If not evaluating, it enters recovery mode. 
		If direct training is enabled, it sets failure states. If the first trial and plan fail, it learns a new action using RL and tests it. 
		If the learning fails or the agent fails to execute a trained policy, it returns the trial status. 
		If the task_goal is successful, it prints a success message, otherwise a failure message. 
		If none of these conditions are met, it returns the trial status.
		'''
		if failure_state == None:
			failure_state = {'at(can,drop)': False, 'at(can,pick)': True, 'at_grab_level(gripper,can)': False, 'at_gripper(gripper,activate)': True, 'at_gripper(gripper,drop)': False, 'at_gripper(gripper,lightswitch)': False, 'at_gripper(gripper,pick)': True, 'door_collision': False, 'dropped_off': False, 'grasped(can)': True, 'light_off': False, 'locked(door)': False, 'open(door)': True, 'open(gripper)': False, 'over(gripper,can)': True, 'picked_up(can)': True}
		direct_training = {'direct':direct_train, 'failed_operator':failed_operator, 'failure_state':failure_state}
		if direct_training["direct"]:
			print("\nWarning: direct_training is set to True, no execution in this run.\n")
		novelty_pattern_name = '' if self.novelty_list == None  else '_'.join(sorted(self.novelty_list))
		done = False
		flag = False

		if type(self.novelty_list) == str:
			self.novelty_list = [self.novelty_list]

		# GENERATE ENVIRONMENT
		if only_train:
			self.close_env()
			for operator in adaptive_op:
				if self.novelty_list != None:
					operator = adaptive_op[0]
					flag = True
				if novelty_pattern_name not in self.learned_policies_dict[operator].keys() or flag:
					direct_training["direct"] = True
					direct_training["failed_operator"] = operator
		else:
			if not(direct_training["direct"]):
				self.verboseprint("Agent launching trial {} on task_goal {}.".format(trial, task_goal))
				try:
					self.task_goal_episode[str(task_goal)] += 1
				except:
					self.task_goal_episode.update({str(task_goal): 0})
					
				if self.env == None:
					if self.env_id == None:
						print("Neither env, nor env_id specified for run_brain trial ", trial)
						sys.exit()
					else:
						self.generate_env(env_id=self.env_id, novelty_list=self.novelty_list, render=self.render)
				
				# GENERATE PDDL AND PLAN
				obs = self.env.reset(seed=self.seed)
				state = State(self.detector)
				generate_pddls(state, goal=task_goal._to_pddl())
				self.verboseprint("\033[1m" + "\n\n\tPlanning ...\n\n" + "\033[0m")
				plan, game_action_set = call_planner(self.domain, "problem") # get a plan
				if plan==False or game_action_set==False:
					self.verboseprint("\033[1m" + "\n\n\tAgent couldn't find a plan for: {}, on trial {}.\n\n".format(task_goal, trial) + "\033[0m")
					return done, trial
				# EXECUTE PLAN
				done, failed_operator, failure_state = self.execute_plan(self.env, game_action_set, plan, obs)
		# CHECK SUCCESS AND ENTER RECOVERY MODE
		if done and not(direct_training['direct']):
			self.verboseprint("\033[1m" + "\n\n\tAgent successfully achieved task_goal: {}, on trial {}.".format(task_goal, trial) + "\033[0m")
			return done, trial # agent is successful in achieving the task_goal
		elif not(only_eval): # Enters recovery mode
			# REMOVE THE FAILED OPERATOR AND RE-PLAN
			#remove_action(pddl_op[failed_operator.split(' ')[0]], self.domain, "modified_domain")
			self.verboseprint("\033[1m" + "\n\n\tRe-planning after removing the grounded failed operator: {}.\n\n".format(failed_operator) + "\033[0m")
			restrict_pddl(failed_operator.split(' ')[0].lower(), failed_operator.split(' ')[1].lower(), domain_name=self.domain, modified_domain_name="modified_domain", modified_problem_name="modified_problem")
			plan, game_action_set = call_planner("modified_domain", "modified_problem")
			# When re-planning still fails: launch learning method
			if trial == 1: # cases when the plan and re-plan failed for the first time and the agent needs to learn a new action using RL
				if self.novelty_list == None:
					novelty_pattern = []
					novelty_pattern_name = ''
				else:
					novelty_pattern = [novelties_info[n]["pattern"] for n in self.novelty_list]
					novelty_pattern_name = '_'.join(sorted(self.novelty_list))
				if novelty_pattern_name not in self.learned_policies_dict[failed_operator.split(' ')[0]].keys():
					# if no policy has yet been trained on the novelty pattern
					self.verboseprint("\033[1m" + "\n\n\tInstantiating a RL Learner to learn a new action to solve the impasse, and learn a new executor for {}.".format(failed_operator.split(" ")[0]) 
					+ "\n\tThe detected novelty patterns are {} associated with this ids {}.".format(novelty_pattern, self.novelty_list)
					+ "\n\tcurrent time: {}\n\n".format(datetime.now()) + "\033[0m")
					self.learned = self.call_learner(task_goal=task_goal, failed_operator=failed_operator, failure_state=failure_state, novelty_pattern=novelty_pattern, verbose=self.verbose)
					if self.learned: # when the agent successfully learns a new action, it should now test it to re-run the environment.
						self.verboseprint("Agent succesfully learned a new action in the form of policy. Now resetting to test.")
						while (not done) and (trial<max_trials):
							done, trial = self.run_brain(task_goal=task_goal, trial=trial+1, only_eval=True)
					else:
						self.verboseprint("Agent failed to learn a new action in the form of policy. Exit..")
						return done, trial # Agent was unable to learn an executor that overcomes the novelty
				else:
					self.verboseprint("Agent failed to execute a policy already trained on the novelty pattern.")
					return done, trial # Agent was unable to learn an executor that overcomes the novelty
			else:
				return done, trial # for trials >1, returns the result of using the newly learned operator in the global task_goal
			if done:
				self.verboseprint("Success!")
			else:
				self.verboseprint("Agent was unable to achieve the task_goal: {}, despite {} trials.".format(task_goal, trial))
			return done, trial # back to main, returns the global results of run_nrain function
		else: 
			return done, trial

	def call_learner(self, task_goal, failed_operator, failure_state, novelty_pattern=None, verbose=False):
		'''
		This method is responsible for instantiating a reinforcement learning (RL) learner to find interesting states to send to the planner. 

		The learner computes Beta from the expected effects of the operator and I from the failure state. It then learns a policy that finds a path 
		from states validated by I to states validated by Beta.

		The method takes the following parameters:
		- failed_operator: The operator that failed.
		- failure_state: The state in which the operator failed.
		- novelty_pattern: The novelty pattern to be used by the learner. Default is None.
		- verbose: A flag indicating whether to print verbose output. Default is False.

		It initializes a new Learner with the given parameters and the instance attributes steps_num, test, seed, transfer, novelty_list, DATA_DIR, and use_base_policies.

		It then updates the learned_policies_dict with the learned policy from the learner, using the failed operator and the novelty pattern as keys.

		Finally, it returns the learned policy.
		'''
		self.close_env()
		self.learner = Learner(self.env_id, task_goal, rl_alg=self.rl_alg, domain=self.domain, steps_num=self.steps_num, eval_freq=self.eval_freq, test=self.test, seed=self.seed, transfer=self.transfer, failed_operator=failed_operator, failure_state=failure_state, novelty_pattern=novelty_pattern, verbose=verbose, data_folder=self.DATA_DIR, use_basic_policies=self.use_base_policies) # learns the policiy, I, Beta and C. Stores result as an executor object and maps the executor to the operator's list
		novelty_pattern_name = '' if self.novelty_list == None  else '_'.join(sorted(self.novelty_list))
		self.learned_policies_dict[failed_operator.split(' ')[0]].update({novelty_pattern_name: self.learner.learned}) # save the learner instance object to the learned policies dict.
		return self.learner.learned

	def execute_plan(self, env, sub_plan, plan, obs):
		'''
		This method executes a given plan in a provided environment. It takes as input the environment, a sub-plan, the main plan, and the initial observation. 

		The method iterates over each operator in the plan, and for each operator, it selects the best executor based on the novelty list and the current state. 
		It then uses the executor's policy to predict the next action and applies this action to the environment. 

		The method continues to execute actions until the executor's Beta condition is met or a maximum of 1000 steps have been taken. 
		If the execution effects match the expected effects of the operator, the method moves on to the next operator. 
		If not, it tries the next executor in the queue. 

		If all executors have been tried and none of them achieved the expected effects, the method launches recovery mode and returns 
		False along with the failed operator and the old state. If all operators in the plan have been executed successfully, it returns True.
		'''
		self.verboseprint("Running plan execution.")
		self.sub_plan = sub_plan
		self.plan = plan
		i = 0

		
		self.verboseprint("The plan is: ", plan)
		self.verboseprint(sub_plan)
		while (i < len(plan)): # this is for the operators.
			success = True
			#step_count = env.step_count
			old_state = State(self.detector)
			self.current_state = old_state
			queue = copy.deepcopy(sub_plan[i])
			if not(self.use_base_policies):
				for executor in queue:
					if executors[executor].basic:
						queue.remove(executor)
			self.verboseprint("\n{}   {}\n".format(i, queue))
			while (len(queue) > 0): # this is for the executors policies queue mapped to operator i.
				executor = select_best_executor(self.novelty_list, queue, old_state)
				self.verboseprint("\nExecutor : {}, specialized on {}.".format(executor, novelty_patterns[executor]))
				queue.remove(executor)
				self.verboseprint("\nExecuting plan_step: {}, mapped to executor {}.".format(plan[i], executor))

				try:
					# EXECUTE EXECUTOR (POLICY OR CODED)
					obs, success = executors[executor].execute(env, plan[i], render=self.render, obs=obs)
					new_state = State(self.detector)
					print("New state: ", new_state)
					expected_effects = effects(plan[i])
					execution_effects = new_state.compare(old_state)
					self.verboseprint("The operator expected effects are: {}, the execution effects are: {}.".format(expected_effects, execution_effects))
					success = all(item in execution_effects.items() for item in expected_effects.items())
					if success:
						break
					else:
						self.verboseprint("\n{} failed. Moving to next executor in {} queue.\n".format(executors[executor].id, plan[i]))
				except FileNotFoundError: 
					self.verboseprint("FileNotFound: Tried to execute {} '{}', but it failed. Trying to continue...".format(executors[executor].id, executor))
				except RuntimeError:
					self.verboseprint("\nRuntime Error while executing {}. Moving to next executor in {} queue.\n".format(executors[executor].id, plan[i]))
					success = False

			if not success:
				self.verboseprint("\nThe execution effects don't match the operator {}. Launching recovery mode.\n".format(plan[i]))
				return False, plan[i], old_state
			i+=1
		return True, None, None