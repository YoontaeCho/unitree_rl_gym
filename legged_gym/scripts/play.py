import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import get_args, export_policy_as_jit
from icecream import ic

import numpy as np
import torch
import time
import pickle


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_init_orn = True
    observations = []
    logs = []

    # env_cfg.asset.disable_gravity = True

    # env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    try:
        for i in range(10*int(env.max_episode_length)):
            start_time = time.time()

            actions = policy(obs.detach())
            # Temporarily due to debugging
            # actions *= 0
            time.sleep(0.01)
            obs, _, rews, dones, infos = env.step(actions.detach())

            # Save the observations
            # current_obs = obs.cpu().numpy() if obs.is_cuda else obs.numpy()
            # observations.append(current_obs)
            # com = infos['log_com'].cpu().numpy()
            # zmp = infos['log_zmp'].cpu().numpy()
            # ic(com.shape)
            # ic(np.stack((com, zmp), axis=-1).shape)
            # logs.append(np.stack((com, zmp), axis=-1))

            # observations.append(current_obs)
            stop_time = time.time()

            duration = stop_time - start_time
            time.sleep(max(0.02 - duration, 0))
    finally:
        # observations = np.concatenate(observations, axis=0)
        # # Save the concatenated observations to a pickle file
        # ic(observations.shape)
        # with open('/tmp/training_dist.pkl', 'wb') as f:
        # # with open('/tmp/testing_dist.pkl', 'wb') as f:
        #     pickle.dump(observations, f)
        # logs = np.concatenate(logs, axis=0)
        # # Save the concatenated observations to a pickle file
        # ic(logs.shape)
        # with open('/tmp/training_dist.pkl', 'wb') as f:
        # # with open('/tmp/testing_dist.pkl', 'wb') as f:
        #     pickle.dump(logs, f)
        pass

if __name__ == '__main__':
    # EXPORT_POLICY = True
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args(test=True)
    play(args)
