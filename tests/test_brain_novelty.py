from brain import *

brain_test = Brain(verbose=True)
brain_test.generate_env("Hole", novelty_list=["Hole"], render=False)
detector = brain_test.detector
task_goal = {"at(can,drop)":True, "picked_up(can)":False}
task_goal = State(detector, init_predicates=task_goal)
done, trial = brain_test.run_brain(task_goal, trial=1, max_trials=5)