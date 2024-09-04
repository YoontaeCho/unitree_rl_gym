import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

import os, shutil

def log_files(log_dir, curr_task_path):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for task_path in [curr_task_path, 'base']:
        task_full_path = os.path.join(LEGGED_GYM_ENVS_DIR, task_path)
        for f in os.listdir(task_full_path):
            file_path = os.path.join(task_full_path, f)
            if os.path.isfile(file_path):
                shutil.copy2(file_path, os.path.join(log_dir, f))

    
    rsl_rl_src_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'rsl_rl', 'rsl_rl')
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'algorithms', 'ppo.py'), os.path.join(log_dir, 'ppo.py'))
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'modules', 'actor_critic.py'), os.path.join(log_dir, 'actor_critic.py'))
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'runners', 'on_policy_runner.py'), os.path.join(log_dir, 'on_policy_runner.py'))
    shutil.copy2(os.path.join(rsl_rl_src_dir, 'storage', 'rollout_storage.py'), os.path.join(log_dir, 'rollout_storage.py'))


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()   

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()