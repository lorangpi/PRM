import os
import sys
#import gym_panda_novelty
import numpy as np

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy

save_path = "./"
save_freq = 100000
total_timesteps = 1000000
policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]))

# Custom Evaluation Callback to save the results in a csv file
class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq=10000, n_eval_episodes=5, deterministic=True, render=False, verbose=1):
        super(CustomEvalCallback, self).__init__(eval_env=eval_env, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render, verbose=verbose)

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

def train(env, eval_env=None, model=None, reward_threshold=800,
          save_path=save_path, run_id="", save_freq=save_freq, best_model_save_path=None,
          total_timesteps=total_timesteps, policy_kwargs=policy_kwargs,
          eval_freq=50_000, n_eval_episodes=10, eval_seed=100):
 
    # setting up logging
    if run_id == "":
        if len(sys.argv) > 1:
            run_id = f"-{sys.argv[1]}"
        log_dir = f"{save_path}logs/{env.spec.id}"
        model_dir = f"{save_path}models/{env.spec.id}"

    log_dir = f"{save_path}logs/"
    model_dir = f"{save_path}models/"
    if best_model_save_path is None:
        best_model_save_path = model_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # evaluation
    callbacks = []
    if eval_env is not None:
        os.makedirs(log_dir+"-eval", exist_ok=True)
        # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
        # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=2, verbose=1)
        # eval_callback = FixedSeedEvalCallback(eval_env,
        #                                       callback_on_new_best=callback_on_best,
        #                                       callback_after_eval=stop_train_callback,
        #                                       n_eval_episodes=n_eval_episodes,
        #                                       eval_freq=eval_freq,
        #                                       deterministic=True,
        #                                       render=False,
        #                                       best_model_save_path=best_model_save_path,
        #                                       seed=eval_seed)
        eval_callback = CustomEvalCallback(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_dir+"-eval",
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)

    # start learning
    if model == None:
        model = SAC(MlpPolicy, env, verbose=1, gamma=0.95, learning_rate=0.0003, policy_kwargs=policy_kwargs, tensorboard_log=log_dir+"-tensorboard")
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=model_dir, name_prefix=run_id)
    callbacks.append(checkpoint_callback)
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), log_interval=10)

    return model
