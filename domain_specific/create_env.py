import warnings
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from operator_learners.HER_wrapper import HERWrapper, No_end
from operator_learners.operator_wrapper import TaskWrapper
from operator_learners.plan_wrapper import PlanWrapper
from stable_baselines3.common.monitor import Monitor
from domain_specific.detector import RoboSuite_PickPlace_Detector

warnings.filterwarnings("ignore")
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

def create_env(env_id, render=False, dense_reward=False, alg="her_symbolic", seed=None, mode="execution", sub_goal=None, task_goal=None, pddl_domain="training_domain", pddl_problem="training_problem"):
    '''
    This method is responsible for generating a new environment for the experiment. It takes an environment ID, a list of novelties, 
    a reset location and direction, a render flag, and a training flag as parameters. 

    If the environment does not exist or the novelty list has changed, it closes the existing environment (if any) and creates a new one. 
    It sets the render, reset location and direction, and environment ID attributes of the instance. 
    It determines the locality of the environment based on the novelties. If there are multiple novelties or a global novelty, the locality 
    is set to "global". If there is a single local novelty, the locality is set to "local".
    It then injects the novelties into the environment by creating a new instance of the novelty wrapper for each novelty and adding it to 
    the environment. 
    Finally, it seeds the environment and sets the environment and novelty list attributes of the instance.
    '''
    print("\nCreating env: {}".format(env_id))
    # Create the environment
    env = suite.make(
        env_id,
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=render,
        has_offscreen_renderer=render,
        use_camera_obs=False,
        horizon=1000,
        render_camera="agentview",
    )

    print("Using dense reward: {}".format(dense_reward))

    env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
    if alg in ['her','her_symbolic','her_augmented','her_symbolic_augmented']:
        env = TaskWrapper(env, dense_reward=dense_reward, augmented_obs=False, render_init=render)
        print("Wrapping the environment in a HER_wrapper.")
        if alg == 'her':
            env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=False, symbolic_goal=False)
        elif alg == 'her_symbolic':
            if mode == "training":
                env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=False, symbolic_goal=True, factor=100, return_int=True)
            else:
                env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=False, symbolic_goal=True)
        elif alg == 'her_augmented':
            env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=True, symbolic_goal=False)
        elif alg == 'her_symbolic_augmented':
            env = HERWrapper(env, dense_reward=dense_reward, augmented_obs=True, symbolic_goal=True)
    elif alg == 'sac':
        env = TaskWrapper(env, dense_reward=dense_reward, augmented_obs=False, render_init=render)
    elif alg == 'sac_augmented':
        env = TaskWrapper(env, dense_reward=dense_reward, augmented_obs=True, render_init=render)
    else:
        print("Experiment not recognized. Please verify the name.")
        exit()
    #check_env(env)
    #env = Monitor(env, filename=None, allow_early_resets=True)
    #if mode == "execution":
        #env = No_end(env)
    if mode == "training":
        env = PlanWrapper(env, sub_goal, task_goal, domain=pddl_domain, problem=pddl_problem)
        env = Monitor(env, filename=None, allow_early_resets=True)
    if seed == None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    if render:
        env.render()
    return env, RoboSuite_PickPlace_Detector(env), obs
