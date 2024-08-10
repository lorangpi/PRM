#!/bin/bash
# Path to the virtual environment
VENV_PATH=~/hybridPlanningLearning/PRM/.venv
PYTHONPATH=~/hybridPlanningLearning/PRM:~/hybridPlanningLearning/PRM/robosuite:~/hybridPlanningLearning/PRM/robosuite/src:~/hybridPlanningLearning/PRM/robosuite/src/robosuite:\$PYTHONPATH && \
                        
# Start a new tmux session
tmux new-session -d -s experiment_session

for seed in {0..9}
do
    if [ $seed -eq 0 ]; then
        # Use the first pane for the first seed
        tmux send-keys "source $VENV_PATH/bin/activate && \
                        export PYTHONPATH=$PYTHONPATH && \
                        python experiment.py --experiment sac_augmented --icm --prm --no_transfer --seed $seed; \
                        if [ \$? -eq 0 ]; then echo 'Seed $seed: Experiment completed successfully'; \
                        else echo 'Seed $seed: Experiment failed'; fi" C-m
    else
        # Split the window and run the next command in the new pane
        tmux split-window -v
        tmux select-layout tiled
        tmux send-keys "source $VENV_PATH/bin/activate && \
                        export PYTHONPATH=$PYTHONPATH && \
                        python experiment.py --experiment sac_augmented --icm --prm --no_transfer --seed $seed; \
                        if [ \$? -eq 0 ]; then echo 'Seed $seed: Experiment completed successfully'; \
                        else echo 'Seed $seed: Experiment failed'; fi" C-m
    fi
done

# Attach to the tmux session to view the experiments
tmux attach-session -t experiment_session