from brain import *
from state import *
from stable_baselines3 import SAC, HerReplayBuffer



brain_test = Brain(verbose=True)
brain_test.generate_env("PickPlaceCanNovelties", render=True)
detector = brain_test.detector
task_goal = {"at(can,drop)":True, "picked_up(can)":False}
task_goal = State(detector, init_predicates=task_goal)

env = brain_test.env
obs = brain_test.obs
model = SAC.load("/home/lorangpi/HyGOAL/operator_learners/models/her_symbolic/base/best_model.zip", env=env)

# executors["Pick"].execute(env, "PICK can", render=True)

# for i in range(500):
#     #action = env.action_space.sample()
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     #print(detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False))
#     if terminated or truncated or i == 499:
#         env.close()
#         env.close_renderer()
#         #sys.exit()
#         #obs, _ = env.reset()
#         break

done, trial = brain_test.run_brain(task_goal)