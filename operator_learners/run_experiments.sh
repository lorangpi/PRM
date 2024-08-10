#!/bin/bash
# Export PYTHONPATH with multiple paths
export PYTHONPATH=~/hybridPlanningLearning/PRM:~/hybridPlanningLearning/PRM/robosuite:~/hybridPlanningLearning/PRM/robosuite/src:~/hybridPlanningLearning/PRM/robosuite/src/robosuite:$PYTHONPATH

for seed in {0..9}
do
    python experiment.py --experiment sac_augmented --icm --prm --no_transfer --seed $seed
done