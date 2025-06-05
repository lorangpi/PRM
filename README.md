# Code related to the paper "Curiosity-Driven Imagination: Discovering Plan Operators and Learning Associated Policies for Open-World Adaptation" ICRA 2025

Link to the paper: https://arxiv.org/pdf/2503.04931

## Setup

First make sure you have installed the modified `Robosuite` git repository (https://github.com/lorangpi/robosuite/tree/goal_env) which is a sub-module of this git repository.
To install, clone this repo and run the following:

From the root of the repo just run:
```
conda create -n prm python=3.8.10
conda activate prm
pip install -r requirements.txt
(If the code is given as zipfile, you do not need the next step)
git submodule update --init --recursive
cd robosuite
pip install -e .
```
Other libraries might be needed.


## Usage

To run a test of the environment go to operator_learners and run:

```
cd /operator_learners
python test_base_operator.py
```

To run a training of a base policy run
```
python train_base_operator.py --task <> --lr <> --timesteps <>
```
The training is currently running with HER Wrapper and replay buffer. Feel free to run it with and without it. There is a dense_reward flag, initially set to True in the base wrappers and False in the HER Wrapper. This can also be modified as wished.

### The Operators Environments

You can find what is mostly relevant to the splitting of the pick and place task into the 3 pick, reach and drop operators in the file operator_learners/operator_wrapper.py. This files sets up the important base methods for each operator by providing the user with operator classes that iniherit from gymnasium.Wrapper. Mostly, these wrappers build the reset, step, compute_reward and sample_goal methods that are required for each sub-task.

An Hindsight Experience Replay Wrapper is also provided at operator_learners/HER_wrapper.py, which builds on top of any env to make it suitable for HER algorithmic. Currently the desired_goal computation is only set for the reach task.

### The Detector Function

The detector function is provided as a class at operator_learners/detector.py (from detector import RoboSuite_Detector). It ouptuts the high level state description from an analysis of the environment. You can define it simply as detector = RoboSuite_Detector(env, single_object_mode=True, object_to_use='can') or simply detector = RoboSuite_Detector(env) if you add all 4 objects of the usual RoboSuite PickPlace env (i.e., you make an instance of "PickPlaceCan" or "PickPlace" respectively, cf test_base_operator.py).

To output the grounded symbolic representation of the state, simply run:

```
detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
```
Note that it can output this state as a dict (including the semantics as keys) or an ordered array of values. These values, either as in the dict or the array, can be expressed as float, i.e., binary_to_float=True (for instance at(can,pick)=True will become at(can,pick)=1.0) and you can also choose to output the distance metric to the predicate reference if available for the predicate (for instance at(can,pick)=False will become at(can,pick)=6, the value is multiplied by some factor as the number are too low initially, and returned as int).

### New Items in PickPlace Environment

I added a door and a visual ball (representing the activation area to unlock the door) in the pick and place environment (robosuite/robosuite/environments/manipulation/pick_place.py). Notable changes can be seen around lines: 236, 245, 457, 500, 742 (the _reset_internalts method) these parts are necessary to add an object in the environment, by playing with the "placement_initializer". Some methods are private, and I have not added or modified yet objects that need to occur out of the PickPlace env instanciation. You will need to do that for instance to change the door configuration to the configuration commented at lines 464/467 for it to block the reach operator and become a novelty in a wrapper, out of the environment instanciation (you might need to make some private functions public to do so).

### Actuating the End Effector using the OSC_POSE / OSC_POSITION controllers. 

The controller is set when making the environment (cf test_base_operators file) controller_config = suite.load_controller_config(default_controller='OSC_POSITION'). If you want to have an idea on how to then move the robot gripper toward a target, please refer to the method pick_reset(self) line 141 in perator_learners/operator_wrapper.py ReachWrapper class.

### Run brain

To run experiments with the brain just run:
python experiment.py --experiment hygoal -V -F -R

cf flags
