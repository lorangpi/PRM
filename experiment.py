'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu
'''
import os, csv, argparse, uuid, random, json, time
from argparse import ArgumentParser
from datetime import datetime
from brain import *
from operator_learners.train import train
from operator_learners.evaluate_baseline import eval_success

from domain_specific.synapses import *

class Experiment:
	def __init__(self, args, novelty_list, extra_run_ids=''):
		self.DATA_DIR = args['log_storage_folder']
		self.hashid = uuid.uuid4().hex
		self.experiment_id = extra_run_ids
		self.results_dir = self._get_results_dir()
		os.makedirs(self.results_dir, exist_ok=True)
		self.recover = args['recover']
		expe_path = os.path.split(self.results_dir[:-1])[0]
		if not(expe_path.endswith('/')):
			expe_path += '/'
		with open(expe_path+'args.txt', 'w') as f:
			json.dump(args, f, indent=2)
		self.novelty_list = novelty_list
		self.test = args["test"]
		self.fixed_locations = args["fixed_locations"]
		if self.fixed_locations:
			self.evaluations_num = 20
		else:
			self.evaluations_num = args['trials_eval']
		if self.test:
			self.trials_eval_pre = 2
			self.trials_eval_post = 2
			self.trials_training = 2
		else:
			self.trials_eval_pre = self.evaluations_num
			self.trials_eval_post = self.evaluations_num
			self.trials_training = args['trials_training']
		self.render = args['render']
		self.env_id="PickPlaceCanNovelties"
		self.verbose = args['verbose']
		self.transfer = args['transfer']
		self.seed = args['seed']
		self.steps_num = args['steps']
		self.direct_training = args["direct_training"]
		if self.direct_training:
			self.trials_eval_pre = 0
			self.trials_eval_post = 0

	def _get_results_dir(self):
		if self.experiment_id == '':
			return self.DATA_DIR + os.sep
		return self.DATA_DIR + os.sep + self.experiment_id + os.sep

	def write_row_to_results(self, data, tag):
		db_file_name = self.results_dir + os.sep + str(tag) + "results.csv"
		with open(db_file_name, 'a') as f:  # append to the file created
			writer = csv.writer(f)
			writer.writerow(data)


class HyGoalExperiment(Experiment):
	HEADER_TRAIN = ['Episode', 'Done']
	HEADER_TEST = ['Novel','success_rate'] # Novel: 0=pre-novelty_domain, 1=post-novelty_domain

	def __init__(self, args, novelty_list=None, experiment_id='no_id', brain=None, eval_only=False, eval_only_pre_post=False):
		if novelty_list == None:
			novelty_list = args['novelty_list']
		if experiment_id == None:
			experiment_id = '_'.join(novelty_list)
		super(HyGoalExperiment, self).__init__(args, novelty_list, experiment_id)

		self.write_row_to_results(self.HEADER_TRAIN, "train")
		self.write_row_to_results(self.HEADER_TEST, "test")

		if eval_only_pre_post:
			self.trials_training = 0
			self.trials_eval_post = 0
		elif eval_only:
			self.trials_training = 0
			self.trials_eval_pre = 0

		self.novelty = novelty_list
		if type(self.novelty_list) == str:
			self.novelty_list = [self.novelty_list]
		self.brain = brain
		self.rl_alg = args["rl_alg"]
		self.num_steps = args["steps"]
		self.eval_freq = args["eval_freq"]

		print('General experiment directory is:', os.path.split(self.results_dir[:-1])[0])
		print('Novelty result directory is:', self.results_dir)

	def run(self):
		print("\033[1m" + "\n\n\t\t\t\t\t===> HyGOAL EXPERIMENT ON: {} <===\n\n".format(self.novelty) + "\033[0m")
		if self.brain == None:
			brain = Brain(rl_alg=self.rl_alg, steps_num=self.num_steps, eval_freq=self.eval_freq, verbose=self.verbose, DATA_DIR=self.results_dir, transfer=self.transfer, seed=self.seed, test=self.test)
			if self.recover:
				brain.load_infos()
		else:
			brain = self.brain
			brain.DATA_DIR = self.results_dir
			self.env_id = self.novelty_list[0]
		# run the pre novelty evaluation on self.trials_eval_pre episodes
		if self.trials_eval_pre > 0:
			print("\n\n\nEVALUATION. PRE-NOVELTY")
			brain.generate_env(env_id=self.env_id, alg=self.rl_alg, render=self.render)
			succ = self.eval(brain, "Pre-novelty", eval_num=self.trials_eval_pre)
			self.write_row_to_results([0, succ], "test")

		# inject novelty and run again evaluation on self.trials_eval_pre episodes
		if self.trials_eval_pre > 0:
			print("\n\n\nEVALUATION. NOVELTY INJECTION")
			brain.generate_env(env_id=self.env_id, alg=self.rl_alg, novelty_list=self.novelty_list, render=self.render)
			succ = self.eval(brain, "Pre-training", eval_num=self.trials_eval_pre)
			self.write_row_to_results([1, succ], "test")	

		# train on novelty on self.trials_training episodes 
		if self.trials_training > 0:
			print("\n\n\nTRAINING.")
			brain.generate_env(env_id=self.env_id, alg=self.rl_alg, novelty_list=self.novelty_list, render=self.render, only_train=True)
			for episode in range(self.trials_training):
				task_goal = self.generate_random_task(brain)
				max_trials = 1
				if self.direct_training:
					max_trials = 1
				done, trial = brain.run_brain(task_goal=task_goal, trial=1, max_trials=max_trials, only_train=True, direct_train=self.direct_training)
				print("\tPost-novelty domain  > Train Success on episode {}: {}\n\n".format(episode, done))
				self.write_row_to_results([episode, done], "train")

		# run the post novelty evaluation on self.trials_eval_post episodes
		if self.trials_eval_post > 0:
			print("\n\n\nEVALUATION. TRAINED ON NOVELTY")
			brain.generate_env(env_id=self.env_id, alg=self.rl_alg, novelty_list=self.novelty_list, render=self.render)
			succ = self.eval(brain, "Post-training", eval_num=self.trials_eval_post)
			self.write_row_to_results([2, succ], "test")

		brain.close_env()
		#if self.trials_eval_pre > 0:
		brain.save_infos()
		print("\n\n\n\n\n\n")
		return brain

	def train_init_policies(self):
		print("\033[1m" + "\n\n\t\t\t\t\t===> HyGOAL EXPERIMENT ON: {} <===\n\n".format(self.novelty) + "\033[0m")
		brain = Brain(verbose=self.verbose, DATA_DIR=self.results_dir, transfer=self.transfer, seed=self.seed, test=self.test, base_port=self.base_port, eval_port=self.eval_port, steps_num=self.steps_num, use_base_policies=False)

		print("\n\n\nTRAINING")
		brain.generate_env(env_id=self.env_id, alg=self.rl_alg, novelty_list=self.novelty_list, render=self.render, only_train=True)
		for episode in range(3):
			task_goal = self.generate_random_task(brain)
			brain.run_brain(task_goal=task_goal, trial=1, max_trials=1, only_train=True, direct_train=True)

		brain.close_env()
		brain.save_infos()
		print("\n\n\n\n\n\n")
		return brain

	def generate_random_task(self, brain):
		task_goal = {"at(can,drop)":True, "picked_up(can)":False}
		task_goal = State(brain.detector, init_predicates=task_goal)
		return task_goal

	def eval(self, brain, title, eval_num=20):
		succ = 0
		for episode in range(eval_num):
			try:
				task_goal = self.generate_random_task(brain)
				done, trial = brain.run_brain(task_goal=task_goal, only_eval=True, direct_train=self.direct_training)
				print("\t"+title+" domain  > Test Success on episode {}: {}\n\n".format(episode, done))
				if done:
					succ +=1
			except Exception as e:
				print("Exception occured with message {}".format(e)) # error independent of the method performances
				episode -= 1
		return succ/eval_num

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--experiment", default="experiment", help="Type of experiment")
	ap.add_argument("-alg", "--rl_alg", default="her_symbolic", help="RL algorithm to use: can be her_symbolic, her, sac")
	ap.add_argument("-N", "--novelty_list", nargs='+', default=None, help="List of novelty (-N n1 n2...) to inject including: #Elevated #rain #mist #Hole #obstacle #traffic", type=str)
	ap.add_argument("-log", "--log_storage_folder", default='data/', help="Path to log storage folder", type=str)
	ap.add_argument("-te", "--trials_eval", default=100, help="Number of episode to evaluate the agent pre and post novelty performances", type=int)
	ap.add_argument("-tt", "--trials_training", default=3, help="Number of episodes of novelty accomodation", type=int)
	ap.add_argument("-steps", "--steps", default=500_000, help="Number of steps to train on each novelty", type=int)
	ap.add_argument("-s", "--seed", default=0, help="Experiment seed", type=int)
	ap.add_argument("-m", "--model", help="Path to file model to load for SAC RL baseline and folder containing basic operators for hygoal.", default='models/navigate', type=str)
	ap.add_argument("-V", "--verbose", action="store_true", default=False, help="Boolean verbose")
	ap.add_argument("-T", "--transfer", action="store_true", default=False)
	ap.add_argument("-R", "--render", action="store_true", default=False)
	ap.add_argument("-F", "--fixed_locations", action="store_true", default=False)
	ap.add_argument("-quick_training", "--test", action="store_true", default=False, help="Boolean quick test, sets low number of training episodes")
	ap.add_argument("-direct_training", "--direct_training", action="store_true", default=False, help="Boolean direct_training, sets operator failure to True with pre-defined failed operator and failure states: cf brain file direct_training dict.")
	ap.add_argument("-eval", "--eval", action="store_true", default=False, help="Boolean eval, sets the experiment for eval only after novelty injection.")
	ap.add_argument("-eval_pre_post", "--eval_pre_post", action="store_true", default=False, help="Boolean eval_pre_post, sets the experiment for eval only before and after novelty injection.")
	ap.add_argument("-pre_training", "--pre_training", action="store_true", default=False, help="Boolean pre_training, regenerates initial policies for goForward, turnLeft and turnRight.")
	ap.add_argument("-eval_freq", "--eval_freq", default=100, help="Frequency of evaluation during training", type=int)
	ap.add_argument("-recover", "--recover", action="store_true", default=False, help="Boolean recover brain from -log <experiment_path>.")
	ap.add_argument("-load", "--load", action="store_true", default=False, help="Boolean load brain from -log <experiment_path>.")
	args = vars(ap.parse_args())

	set_random_seed(args['seed'], using_cuda=True)
	random.seed(args['seed'])

	if args['novelty_list'] == None:
		args['novelty_list'] = []

	checkpoint = args['novelty_list']
	if checkpoint == []:
		reached_checkpoint = True  
	else: 
		reached_checkpoint = False
		model = args['model']
	
	ex_id = args['experiment'] + f"{to_datestring(time.time())}"#self.hashid
	if args['load']:
		ex_id = ''
	if args['recover']:
		ex_id = ''
		save_log = args['log_storage_folder']
		save_model = args['model']
		with open(args['log_storage_folder']+'/args.txt', 'r') as f:
			args = json.load(f)
		args['recover'] = True
		args['log_storage_folder'] = save_log
		args['model'] = save_model
		print('Loagind arguments from checkpoint. Args are: \n', args)
		ex_id = ''
		if checkpoint == []:
			checkpoint = args['checkpoint']
			reached_checkpoint = False
			print('\n==> RECOVERING FROM {} <=='.format(checkpoint))
			if args['experiment'] == 'baseline':
				if save_model == 'models/navigate' and args['previous'] != []: # If it consists of the default value, overwrites model using the previous checkpoint model
					name_novelty = ("_").join([novelty.split("_")[-1] for novelty in args['previous']])
					policy_file = '_'.join(sorted(args['previous']))
					model = ("/").join([args['log_storage_folder'], name_novelty, policy_file])
				print('\n==> WITH INITIAL POLICY {} <=='.format(model))

	if  args['experiment'] == 'hygoal':

		brain = None
		if args['pre_training']:
			init = HyGoalExperiment(args, brain=brain, experiment_id=ex_id+"/init")
			brain = init.train_init_policies()

		# Novelty 'Hole'
		if reached_checkpoint or checkpoint == ['Hole']:
			args['checkpoint'], args['previous'] = ['Hole'], []
			experiment1 = HyGoalExperiment(args, novelty_list=['Hole'], brain=brain, experiment_id=ex_id+"/tire")
			brain = experiment1.run()
			reached_checkpoint = True
			args['previous'] = []

		# Novelty 'Elevated'
		if reached_checkpoint or checkpoint == ['Elevated']:
			args['checkpoint'], args['previous'] = ['Elevated'], ['Hole']
			experiment2 = HyGoalExperiment(args, novelty_list=['Elevated'], brain=brain, experiment_id=ex_id+"/ice")
			brain = experiment2.run()
			eval = HyGoalExperiment(args, novelty_list=['Hole'], brain=brain, experiment_id=ex_id+"/ice"+"/tire_eval", eval_only=True)
			eval.run()
			reached_checkpoint = True

		# Novelty 'Obstacle'
		if reached_checkpoint or checkpoint == ['Obstacle']:
			args['checkpoint'], args['previous'] = ['Obstacle'], ['Elevated']
			experiment3 = HyGoalExperiment(args, novelty_list=['Obstacle'], brain=brain, experiment_id=ex_id+"/tire_rain")
			brain = experiment3.run()
			eval = HyGoalExperiment(args, novelty_list=['Hole'], brain=brain, experiment_id=ex_id+"/tire_rain"+"/tire_eval", eval_only=True)
			eval.run()
			eval = HyGoalExperiment(args, novelty_list=['Elevated'], brain=brain, experiment_id=ex_id+"/tire_rain"+"/ice_eval", eval_only=True)
			eval.run()
			reached_checkpoint = True
		
		# Novelty 'Locked'
		if reached_checkpoint or checkpoint == ['Locked']:
			args['checkpoint'], args['previous'] = ['Locked'], ['Obstacle']
			experiment4 = HyGoalExperiment(args, novelty_list=['Locked'], brain=brain, experiment_id=ex_id+"/tire_rain_mist")
			brain = experiment4.run()
			eval = HyGoalExperiment(args, novelty_list=['Hole'], brain=brain, experiment_id=ex_id+"/tire_rain_mist"+"/tire_eval", eval_only=True)
			eval.run()
			eval = HyGoalExperiment(args, novelty_list=['Elevated'], brain=brain, experiment_id=ex_id+"/tire_rain_mist"+"/ice_eval", eval_only=True)
			eval.run()
			eval = HyGoalExperiment(args, novelty_list=['Obstacle'], brain=brain, experiment_id=ex_id+"/tire_rain_mist"+"/tire_rain_eval", eval_only=True)
			eval.run()
			reached_checkpoint = True
		
		# Novelty 'Lightoff'
		if reached_checkpoint or checkpoint == ['Lightoff']:
			args['checkpoint'], args['previous'] = ['Lightoff'], ['Locked']
			experiment5 = HyGoalExperiment(args, novelty_list=['Lightoff'], brain=brain, experiment_id=ex_id+"/tire_rain_mist_hole")
			brain = experiment5.run()
			eval = HyGoalExperiment(args, novelty_list=['Hole'], brain=brain, experiment_id=ex_id+"/tire_rain_mist_hole"+"/tire_eval", eval_only=True)
			eval.run()
			eval = HyGoalExperiment(args, novelty_list=['Elevated'], brain=brain, experiment_id=ex_id+"/tire_rain_mist_hole"+"/ice_eval", eval_only=True)
			eval.run()
			eval = HyGoalExperiment(args, novelty_list=['Obstacle'], brain=brain, experiment_id=ex_id+"/tire_rain_mist_hole"+"/tire_rain_eval", eval_only=True)
			eval.run()
			eval = HyGoalExperiment(args, novelty_list=['Locked'], brain=brain, experiment_id=ex_id+"/tire_rain_mist_hole"+"/tire_rain_mist_eval", eval_only=True)
			eval.run()
			reached_checkpoint = True


	else:
		experiment = HyGoalExperiment(args, experiment_id=ex_id, eval_only=args['eval'], eval_only_pre_post=args['eval_pre_post'])
		brain = experiment.run()
