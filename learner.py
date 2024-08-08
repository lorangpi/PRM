'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This file is the learner of HyGOAL.
# This file launches a learning instance, which builds an MDP on the fly and learns to accomodate a novelty, i.e. a performant enough RL policy, with a termination condition
# beta generated from higher level information and transfer knowledge based on some pattern information (here only the label of the novelty, the detection and charaterization of
# the pattern is assumed and is not studied in the paper.)

'''
import domain_specific.synapses as domain_synapses
from domain_specific.create_env import create_env
from rl_model import load_policy, get_model
from executor import Executor_RL
from generate_pddl import *
from state import *
from planner import *
from domain_specific.novelties import *
from domain_specific.synapses import *
from operator_learners.train import train

class Learner:
	def __init__(self, env_id, task_goal, rl_alg, domain, steps_num, eval_freq, test, seed, transfer, failed_operator, failure_state, novelty_pattern, verbose, data_folder="", use_basic_policies=True) -> None:

		self.reload_synapses()
		self.verboseprint = print if verbose else lambda *a, **k: None
		
		self.rl_alg = rl_alg
		self.test = test
		self.steps_num = steps_num
		self.eval_freq = eval_freq
		self.seed = seed

		self.failed_operator = failed_operator.split(" ")[0]
		self.queue_number = '_' + str(len(applicator[self.failed_operator]))
		self.failure_state = failure_state
		
		# GENERATE TRAINING PDDL DOMAIN AND PROBLEM
		restrict_pddl(failed_operator.split(' ')[0].lower(), failed_operator.split(' ')[1].lower(), domain_name=domain, modified_domain_name="training_domain", modified_problem_name="training_problem")
		self.verboseprint("PDDL domain generated.")

		# CREATE ENVIRONMENTS
		self.verboseprint("Creating training env.")
		env, _, _ = create_env(env_id, render=False, dense_reward=False, alg=rl_alg, seed=seed, mode="training", sub_goal= effects(failed_operator), task_goal=task_goal, pddl_domain="training_domain", pddl_problem="training_problem")
		self.verboseprint("\nCreating eval env.")
		eval_env, _, _ = create_env(env_id, render=False, dense_reward=False, alg=rl_alg, seed=seed, mode="training", sub_goal= effects(failed_operator), task_goal=task_goal, pddl_domain="training_domain", pddl_problem="training_problem")
		self.env = env
		self.eval_env = eval_env

		self.name = failed_operator.split(' ')[0] + self.queue_number
		self.folder = data_folder + self.name + '/'
		self.policy_folder = self.folder + "policy/" + self.name
		self.tensorboard_folder = self.folder+"tensorboard/"

		# Source policy transfer strategy
		self.verboseprint("faced novelty pattern = ", novelty_pattern)
		source_policy = self.select_source_policy(novelty_pattern, transfer, use_basic_policies)
		self.learned = self.learn_policy(source_policy)
		if self.learned:
			self.abstract_to_executor()	
			novelty_patterns.update({self.name:novelty_pattern})
		self.env.close()
		self.verboseprint("Closing training env.")
		try:
			self.eval_env.close()
			self.verboseprint("Closing eval env.")
		except:
			pass

	def reload_synapses(self):
		global applicator 
		global executors
		global novelty_patterns
		global executors_id_list

		executors_id_list = domain_synapses.executors_id_list
		novelty_patterns = domain_synapses.novelty_patterns
		applicator = domain_synapses.applicator
		executors = domain_synapses.executors

	def DesiredGoal_generator(self, env, operator, state=None):
		# Use planner to map state to desired effects knowing the target operator to train on
		if state == None:
			state = State(env.detector)
		generate_pddls(state, task=operator, filename="beta_problem")
		plan, game_action_set = call_planner("beta_domain", "beta_problem") # get a plan
		if plan==False or game_action_set==False:
			return False
		desired_goal = effects(plan[0])
		return desired_goal

	def select_source_policy(self, novelty_pattern, transfer, use_basic_policies):
		# Handling policy source RL transfer - Source policy transfer strategy
		if transfer and novelty_pattern is not None: # if the agent charaterizes some information about the novelty
			source = select_closest_pattern(novelty_pattern, self.failed_operator.split(' ')[0], same_operator=True, use_base_policies=use_basic_policies)
			if source != None:
				self.verboseprint("Transferring from source policy: {}, trained on {}.".format(source, novelty_patterns[source]))
				self.verboseprint("Path to source model: {}".format(executors[source].policy))
				return executors[source].policy 
		return None

	def learn_policy(self, source_policy):
		if source_policy == None:
			model = get_model(alg=self.rl_alg, env=self.env, path=source_policy, log_dir=self.tensorboard_folder, seed=self.seed)
		else:
			model = load_policy(alg=self.rl_alg, env=self.env, path=source_policy, log_dir=self.tensorboard_folder, seed=self.seed)

		policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], qf=[512, 512, 512]))
		if self.test:
			model = train(
				self.env, 
				eval_env=self.eval_env, 
				model=model, 
				policy_kwargs=policy_kwargs, 
				reward_threshold=810,
				save_freq=200, 
				total_timesteps=300,
				best_model_save_path=self.policy_folder,
				eval_freq=200, 
				n_eval_episodes=1,
				save_path=self.folder,
				run_id=self.name)
		else:
			model = train(
				self.env, 
				eval_env=self.eval_env, 
				model=model, 
				policy_kwargs=policy_kwargs, 
				reward_threshold=810,
				save_freq=self.eval_freq, 
				total_timesteps=self.steps_num,
				best_model_save_path=self.policy_folder,
				eval_freq=self.eval_freq, 
				n_eval_episodes=20,
				save_path=self.folder,
				run_id=self.name)

		self.model = model
		#self.model.save(f"{self.folder}models/{self.name}")
		return True

	def abstract_to_executor(self):
		applicator[self.failed_operator].append(self.name)
		executor = Executor_RL(id=self.name, alg=self.rl_alg, policy=f"{self.policy_folder}/best_model", I=self.failure_state, Beta=beta_indicator)
		ex_dict = {self.name:executor}
		executors.update(ex_dict)
		executors_id_list.append(self.name)
		self.verboseprint("Executor {} added to the list.".format(self.name))
