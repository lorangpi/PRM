import argparse
import numpy as np
import os
import robosuite as suite
import time
import warnings
from datetime import datetime
from stable_baselines3.common.env_checker import check_env
from robosuite.wrappers.gym_wrapper import GymWrapper
from HER_wrapper import HERWrapper
from operator_wrapper import ReachWrapper
from icm.icm_big import ICM
from icm.icm_wrapper import ICMWrapper
from icm.icm_callback import ICMTrainingCallback
from prm.plan_wrapper_numerical import PlanWrapper
from prm.plan_wrapper_numerical_her import HERPlanWrapper
#from prm.action import load_action_from_file, restrict_action, replace_actions_in_domain, update_action_cost
from detector import RoboSuite_PickPlace_Detector
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy

warnings.filterwarnings("ignore")
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')


# Custom Evaluation Callback to save the results in a csv file
class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, icm=None, eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False, verbose=1):
        super(CustomEvalCallback, self).__init__(eval_env=eval_env, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render, verbose=verbose)
        self.icm = icm

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.icm != None:
                    self.icm.save_model(os.path.join(self.best_model_save_path, "icm"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()
            
            # Save the results in a csv file located in the second to last directory of log_path
            # Split the log_path to get the second to last directory
            csv_path = os.path.split(self.log_path)[0]
            with open(os.path.join(csv_path, 'results_eval.csv'), 'a') as f:
                f.write("{},{},{},{}\n".format(self.num_timesteps, success_rate, mean_reward, mean_ep_length))
                f.close()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

def learn_policy(args, env, eval_env, name, policy=None, icm=None):
    # Define the goal selection strategy
    goal_selection_strategy = GoalSelectionStrategy.FUTURE
    # Define the model
    if policy == None:
        # Verifies if "her" is in args.experiment
        if "her" in args.experiment or args.prm_her:
            model = SAC(
                'MultiInputPolicy',
                env,
                replay_buffer_class=HerReplayBuffer,
                #Parameters for HER
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy=goal_selection_strategy,
                #    online_sampling=True,),
                ),
                learning_rate=args.lr,
                buffer_size=int(1e6),
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                policy_kwargs=dict(net_arch=[512, 512, 256]),
                verbose=1,
                tensorboard_log=args.logdir,
                seed=args.seed
            )
        else:
            model = SAC(
                'MlpPolicy',
                env,
                learning_rate=args.lr,
                buffer_size=int(1e6),
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                policy_kwargs=dict(net_arch=[512, 512, 256]),
                verbose=1,
                tensorboard_log=args.logdir,
                seed=args.seed
            )
    else:
        model = policy

    print("Saving the model in: {}, as best_model.zip and final model {}".format(args.modeldir, os.path.join(args.bufferdir, args.task + '_sac')))
    # Define all callbacks
    callbacks = []

    # Create the ICM callback
    if args.icm:
        # Initialize the ICM training callback
        icm_training_callback = ICMTrainingCallback(icm)

        # Add the ICM callback to the list of callbacks
        callbacks.append(icm_training_callback)

    # Define the evaluation callback
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=args.modeldir,
        log_path=args.logdir,
        icm=icm,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,

    )
    # Add a stop callback on success rate of 100%
    #callbacks.append(StopTrainingOnRewardThreshold(reward_threshold=0.99, verbose=1))
    # Add a stop callback on success rate of 100%
    #callbacks.append(StopTrainingOnNoModelImprovement(check_freq=1000, max_no_improvement=10000, verbose=1))

    callbacks.append(eval_callback)

    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
    )

    # Save the model
    model.save(os.path.join(args.modeldir, name + '_sac'))

    # Save the replay buffer
    #model.save_replay_buffer(os.path.join(args.bufferdir, args.task + '_sac'))

    return model

def load_policy(args, env, path):
    # Load the model
    model = SAC.load(path, env=env, learning_rate=args.lr, tensorboard_log=args.logdir, seed=args.seed)
    # Load the replay buffer
    #model.load_replay_buffer(os.path.join(args.bufferdir, args.task + '_sac'))
    return model

if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='her_symbolic_augmented', choices=['her', 'sac', 'her_augmented', 'sac_augmented', 'her_symbolic', 'her_symbolic_augmented'], 
                        help='Name of the experiment. Used to name the log and model directories. Augmented means that the observations are augmented with the detector observation.')
    parser.add_argument('--data_folder', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--timesteps', type=int, default=int(5e5), help='Number of timesteps to train for')
    parser.add_argument('--eval_freq', type=int, default=20000, help='Evaluation frequency')
    parser.add_argument('--n_eval_episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--task', type=str, default='reach', choices=['pick', 'reach', 'place'], help='Task to learn')
    parser.add_argument('--no_transfer', action='store_true', help='No transfer learning')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate') # 0.00005 0.00001
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dense', action='store_true', help='Use dense reward')
    parser.add_argument('--icm', action='store_true', help='Use ICM')
    parser.add_argument('--prm', action='store_true', help='Use PRM')
    parser.add_argument('--prm_her', action='store_true', help='Use PRM with HER')
    parser.add_argument('--init_policy', type=str, default=None, help='Path to initial policy')
    parser.add_argument('--novelty', type=str, default=None, help='Novelty to learn')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')

    args = parser.parse_args()
    # Set the random seed
    np.random.seed(args.seed)

    # Define the evaluation frequency
    args.eval_freq = min(args.eval_freq, args.timesteps)
    n_eval_between_novelty = 20 if args.n_eval_episodes > 1 else 1

    # Define the directories
    data_folder = args.data_folder
    experiment_name = args.experiment + '_dense_' + str(args.dense) + '_seed_' + str(args.seed)
    experiment_id = f"{to_datestring(time.time())}"#self.hashid
    if args.name is not None:
        experiment_id = args.name
    args.experiment_dir = os.path.join(data_folder, experiment_name, experiment_id)

    print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))

    # Create the directories
    os.makedirs(args.experiment_dir, exist_ok=True)

    # Save args in a txt file
    with open(os.path.join(args.experiment_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
        f.close()

    # Define the list of novelties
    list_of_novelties = ["PickPlaceCanNovelties", "Hole", "Elevated", "Obstacle", "Locked", "Lightoff"]
    
    # Jump to args.novelty if it is not None
    if args.novelty != None:
        index = list_of_novelties.index(args.novelty)
        list_of_novelties = list_of_novelties[index:]

    for i in range(len(list_of_novelties)):
        # Create the directories
        args.logdir = os.path.join(args.experiment_dir, list_of_novelties[i], 'logs')
        args.modeldir = os.path.join(args.experiment_dir, list_of_novelties[i], 'models')
        args.bufferdir = os.path.join(args.experiment_dir, list_of_novelties[i], 'buffers')
        args.pddldir = os.path.join(args.experiment_dir, list_of_novelties[i], 'pddl')
        os.makedirs(args.logdir, exist_ok=True)
        os.makedirs(args.modeldir, exist_ok=True)
        os.makedirs(args.bufferdir, exist_ok=True)
        os.makedirs(args.pddldir, exist_ok=True)

        # Test if the file 'best_model.zip' already exists in the folder './models/'+ args.experiment + '/'
        #if list_of_novelties[i] == 'PickPlaceCan' and (os.path.isfile('./models/'+ args.experiment + '/best_model.zip') or (args.no_transfer)):
        #    continue
        print("\nNovelty: {}".format(list_of_novelties[i]))
        # Create the environment
        env = suite.make(
            list_of_novelties[i],
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            horizon=1000,
            render_camera="agentview",
        )
        eval_env = suite.make(
            list_of_novelties[i],
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=1000,
            render_camera="agentview",
        )

        dense_reward = args.dense #(i==0) or args.dense 
        print("Using dense reward: {}".format(dense_reward))

        env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
        eval_env = GymWrapper(eval_env, keys=['robot0_proprio-state', 'object-state'])
        if args.experiment in ['her','her_symbolic','her_augmented','her_symbolic_augmented']:
            print("Wrapping the environment in a HER_wrapper.")
        if args.experiment == 'her':
            env = ReachWrapper(env, dense_reward=dense_reward, augmented_obs=False)
            # ICM
            if args.icm:
                feature_dim = 32
                icm = ICM(env, feature_dim, eta=10)
            env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=False, symbolic_goal=False)
            eval_env = ReachWrapper(eval_env, dense_reward=dense_reward, augmented_obs=False)
            eval_env = HERWrapper(eval_env, dense_reward=dense_reward, augmented_obs=False, symbolic_goal=False)
        elif args.experiment == 'her_symbolic':
            env = ReachWrapper(env, dense_reward=dense_reward, augmented_obs=False)
            # ICM
            if args.icm:
                feature_dim = 32
                icm = ICM(env, feature_dim, eta=10)
            env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=False, symbolic_goal=True)
            eval_env = ReachWrapper(eval_env, dense_reward=dense_reward, augmented_obs=False)
            eval_env = HERWrapper(eval_env, dense_reward=dense_reward, augmented_obs=False, symbolic_goal=True)
        elif args.experiment == 'her_augmented':
            env = ReachWrapper(env, dense_reward=dense_reward, augmented_obs=False)
            env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=True, symbolic_goal=False)
            eval_env = ReachWrapper(eval_env, dense_reward=dense_reward, augmented_obs=False)
            eval_env = HERWrapper(eval_env, dense_reward=dense_reward, augmented_obs=True, symbolic_goal=False)
        elif args.experiment == 'her_symbolic_augmented':
            env = ReachWrapper(env, dense_reward=dense_reward, augmented_obs=False)
            # ICM
            if args.icm:
                feature_dim = 32
                icm = ICM(env, feature_dim, eta=10)
            env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=True, symbolic_goal=True)
            eval_env = ReachWrapper(eval_env, dense_reward=dense_reward, augmented_obs=False)
            eval_env = HERWrapper(eval_env, dense_reward=dense_reward, augmented_obs=True, symbolic_goal=True)
        elif args.experiment == 'sac':
            env = ReachWrapper(env, dense_reward=dense_reward, augmented_obs=False)
            eval_env = ReachWrapper(eval_env, dense_reward=dense_reward, augmented_obs=False)
            # ICM
            if args.icm:
                feature_dim = 32
                icm = ICM(env, feature_dim, eta=10)
        elif args.experiment == 'sac_augmented':
            env = ReachWrapper(env, dense_reward=dense_reward, augmented_obs=True)
            eval_env = ReachWrapper(eval_env, dense_reward=dense_reward, augmented_obs=True)
            # ICM
            if args.icm:
                feature_dim = 32
                icm = ICM(env, feature_dim, eta=10)
        else:
            print("Experiment not recognized. Please verify the name.")
            exit()
        #check_env(env)

        # Check env visually for 100 steps
        #for _ in range(100):
        #    env.render()
        #    env.step(env.action_space.sample())

        # Transfer learning
        source_model = None
        source_path = None
        previous_model_dir = os.path.join(args.experiment_dir, list_of_novelties[i-1], 'models')

        if not(args.no_transfer):
            if i == 0: 
                if args.init_policy != None:
                    source_path = os.path.join(args.init_policy, 'best_model')
                    source_model = load_policy(args, env, source_path)
                else:
                    pass
            elif i == 1:
                if args.init_policy == None and os.path.isfile('./models/'+ args.experiment + '/best_model.zip'):
                    source_path = './models/'+ args.experiment + '/best_model'
                else:
                    #source_path = os.path.join(args.modeldir, list_of_novelties[i-1] + '_sac')
                    source_path = os.path.join(previous_model_dir, 'best_model')
                source_model = load_policy(args, env, source_path)
            else:
                #source_path = os.path.join(args.modeldir, list_of_novelties[i-1] + '_sac')
                source_path = os.path.join(previous_model_dir, 'best_model')
                source_model = load_policy(args, env, source_path)
            
            if args.icm and i > 0:
                icm.load_model(os.path.join(previous_model_dir, 'icm'))
                print("Loading ICM model from: ", os.path.join(previous_model_dir, 'icm'))
            elif args.icm and args.init_policy != None:
                icm.load_model(os.path.join(args.init_policy, 'icm'))
                print("Loading ICM model from: ", os.path.join(args.init_policy, 'icm'))

        if args.no_transfer:
            print("\nNo Transfer Learning\n")
            source_model = None 
        else:
            print("\nTransfer learning")
            print("Transfering from source policy: ", source_path)
            print("")

        # Evaluates the source_model on the eval_env for n_eval_between_novelty if it is not None, and saves the results as a csv file in logdir
        if source_model != None:
            print("Evaluating the source policy on the eval_env for {} episodes.".format(n_eval_between_novelty))
            source_model.set_env(eval_env)
            mean_reward = 0
            mean_episode_length = 0
            success_rate = 0
            for episode in range(n_eval_between_novelty):
                episode_rew = 0
                success = False
                episode_length = 0
                done = False
                obs, info = eval_env.reset()
                while not done:
                    action, _ = source_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    if info.get('is_success') is not None:
                        success = info.get('is_success')
                    episode_rew += reward
                    episode_length += 1
                    done = terminated or truncated
                success_rate += success
                mean_reward += episode_rew
                mean_episode_length += episode_length
            success_rate = success_rate / n_eval_between_novelty
            mean_reward = mean_reward / n_eval_between_novelty
            mean_episode_length = mean_episode_length / n_eval_between_novelty
            # Save the success rate, the reward and the episode length in a csv file
            with open(os.path.join(args.logdir, 'results_eval.csv'), 'a') as f:
                f.write("{},{},{},{}\n".format('pre_training', success_rate, mean_reward, mean_episode_length))
                f.close()
            print("Pre-training Success rate: {}, Reward: {}, Episode length: {}".format(success_rate, mean_reward, mean_episode_length))

            source_model.set_env(env)

        # Trains the policy
        # PRM
        if args.prm:
            # Update the Pddl domain and problem files
            #actions = load_action_from_file("./PDDL_files/domain_numerical.pddl", "./PDDL_files/problem.pddl")
            actions = []
            print("Actions: ", [action._to_pddl() for action in actions])
            #actions = restrict_action(actions, "move(pick, drop, gripper)")
            #actions = restrict_action(actions, "move_obj(pick, drop, gripper, can)")
            #actions = update_action_cost(actions, cost=1000)
            #print("Restricted actions: ", [action._to_pddl() for action in actions])
            #replace_actions_in_domain("./PDDL_files/domain_numerical.pddl", actions)
            #task_goal = State(RoboSuite_PickPlace_Detector(env), {"at(can,drop)":True, "picked_up(can)":False,})
            #task_goal = {"at(can,drop)":0, "grasped(can)":False,}
            task_goal = {"at(can,drop)":1, "grasped(can)":True,}
            detector = RoboSuite_PickPlace_Detector(env, grid_size=200)
            env = PlanWrapper(env, task_goal=task_goal, actions=actions, detector=detector, num_timesteps=args.timesteps, pddl_path=args.pddldir)
        elif args.prm_her:
            actions = []
            print("Actions: ", [action._to_pddl() for action in actions])
            task_goal = {"at(can,drop)":1,}
            detector = RoboSuite_PickPlace_Detector(env, grid_size=200)
            env = HERPlanWrapper(env, task_goal=task_goal, actions=actions, detector=detector, num_timesteps=args.timesteps, pddl_path=args.pddldir)
            eval_env = HERPlanWrapper(eval_env, task_goal=task_goal, actions=actions, detector=detector, num_timesteps=args.timesteps, pddl_path=args.pddldir)
        if args.icm:
            print("Using ICM")
            env = ICMWrapper(env, icm)
            env = Monitor(env, filename=None, allow_early_resets=True)
            if source_model != None:
                source_model.set_env(env)
            eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)
            env.reset(seed=args.seed)
            eval_env.reset(seed=args.seed)
            model = learn_policy(args, env, eval_env, list_of_novelties[i], source_model, icm)
        else:
            env = Monitor(env, filename=None, allow_early_resets=True)
            eval_env = Monitor(eval_env, filename=None, allow_early_resets=True)
            env.reset(seed=args.seed)
            eval_env.reset(seed=args.seed)
            model = learn_policy(args, env, eval_env, list_of_novelties[i], source_model)

        # Evaluates the model on the eval_env for n_eval_between_novelty, and adds the results to the csv file in logdir
        print("Evaluating the best model policy on the eval_env for {} episodes.".format(n_eval_between_novelty))
        # Loads best_model from args.modeldir
        model = load_policy(args, env, os.path.join(args.modeldir, 'best_model'))
        success_rate = 0
        mean_reward = 0
        mean_episode_length = 0
        for episode in range(n_eval_between_novelty):
            episode_rew = 0
            success = False
            episode_length = 0
            done = False
            obs, info = eval_env.reset()
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                if info.get('is_success') is not None:
                    success = info.get('is_success')
                episode_rew += reward
                episode_length += 1
                done = terminated or truncated
            success_rate += success
            mean_reward += episode_rew
            mean_episode_length += episode_length
        success_rate = success_rate / n_eval_between_novelty
        mean_reward = mean_reward / n_eval_between_novelty
        mean_episode_length = mean_episode_length / n_eval_between_novelty
        # Save the success rate, the reward and the episode length in the same csv file
        with open(os.path.join(args.logdir, 'results_eval.csv'), 'a') as f:
            f.write("{},{},{},{}\n".format('post_training', success_rate, mean_reward, mean_episode_length))
            f.close()
        print("Post-training Success rate: {}, Reward: {}, Episode length: {}".format(success_rate, mean_reward, mean_episode_length))
        # Deletes the models
        del model
        del source_model