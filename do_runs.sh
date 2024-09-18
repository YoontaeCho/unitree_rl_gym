#!/bin/bash


# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v43 --env_cfg_updates rewards.scales.action_double_rate=-0.0005
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v48 --env_cfg_updates rewards.scales.action_rate=-0.001 rewards.scales.action_double_rate=-0.0005
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v49 --env_cfg_updates rewards.scales.action_double_rate=-0.0002
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v52 
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v53 --env_cfg_updates rewards.scales.single_foot_contact=-1.0
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v54_2 --env_cfg_updates rewards.scales.single_foot_contact=0 rewards.scales.foot_height=-1.0
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v55_2 --env_cfg_updates rewards.scales.single_foot_contact=0 rewards.scales.foot_height=-2.0
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v56 --env_cfg_updates rewards.scales.single_foot_contact=0 rewards.scales.foot_height=-0.5
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v57 --env_cfg_updates rewards.scales.single_foot_contact=0 rewards.scales.foot_height=-1.0
PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v58 --env_cfg_updates rewards.scales.single_foot_contact=0 rewards.scales.foot_height=-1.0 rewards.scales.zmp_avgfoot_dist=0 rewards.scales.com_avgfoot_dist=-1.0
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v57 --env_cfg_updates rewards.scales.single_foot_contact=0 rewards.scales.foot_height=-1.0 rewards.scales.foot_vel=-0.1
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v55 --env_cfg_updates rewards.scales.single_foot_contact=-5.0
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v55 --env_cfg_updates rewards.scales.pelvis_height=
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v44 
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v48 --env_cfg_updates rewards.scales.foot_height=-0.5
# PYTORCH_JIT=0 python3 legged_gym/scripts/train.py --task=g1_grasp --num_envs=2048 --headless --run_name=test_v45 --env_cfg_updates rewards.scales.foot_vel_v2=-0.02 rewards.scales.foot_vel=0


