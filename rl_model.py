'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu
'''
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.utils import set_random_seed

def load_policy(alg, env, path, lr=0.0003, log_dir=None, seed=0):
    # Load the model
    set_random_seed(seed, using_cuda=True)
    model = SAC.load(path, env=env, learning_rate=lr, tensorboard_log=log_dir, seed=seed)
    return model

def get_model(alg, env, path=None, log_dir=None, lr=0.0003, seed=0):
    set_random_seed(seed, using_cuda=True)
    # Define the goal selection strategy
    goal_selection_strategy = GoalSelectionStrategy.FUTURE
    # Define the model
    if path == None:
        # Verifies if "her" is in experiment
        if "her" in alg:
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
                learning_rate=lr,
                buffer_size=int(1e6),
                learning_starts=10000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                policy_kwargs=dict(net_arch=[512, 512, 256]),
                verbose=1,
                tensorboard_log=log_dir,
                seed=seed
            )
        else:
            model = SAC(
                'MlpPolicy',
                env,
                learning_rate=lr,
                buffer_size=int(1e6),
                learning_starts=10000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                policy_kwargs=dict(net_arch=[512, 512, 256]),
                verbose=1,
                tensorboard_log=log_dir,
                seed=seed
            )
    else:
        model = load_policy
    return model