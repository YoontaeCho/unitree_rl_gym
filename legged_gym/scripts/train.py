import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
import torch
from icecream import ic
import wandb

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ic(dict(class_to_dict(env_cfg), **class_to_dict(train_cfg)))
    if not args.no_wandb:
        wandb.init(project='humanoid_grasp', 
                name=args.run_name, 
                config=dict(class_to_dict(env_cfg), **class_to_dict(train_cfg)))
    # ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)

if __name__ == '__main__':
    args = get_args()
    # args.headless = False
    train(args)
