#!/bin/bash


PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v40 --env_cfg_updates rewards.scales.zmp_avgfoot_dist=-0.5
PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v41 --env_cfg_updates rewards.scales.zmp_avgfoot_dist=-2.0
PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v42 --env_cfg_updates rewards.scales.torques=-0.001



