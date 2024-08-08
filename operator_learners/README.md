# Operator Learners

Collect learners for operators here.

## Instructions to learn the follow-lane operator

0. Make sure you have installed `gym_panda_novelty` as outlined in the README of the repo as well as the Carla simulator including additional maps as well as `stable_baselines3`.
1. Start the Carla server via `CarlaUE4.sh`.
2. Learn the operator: `python lane_follow_sac.py`, plot training curve via `python plot_results.py`.
3. Generate a video of the learned model via `python render_video.py`.
