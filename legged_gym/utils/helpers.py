import os
import copy
import torch
import numpy as np
import random
import itertools
import sys
from isaacgym import gymapi
from isaacgym import gymutil
from typing import Optional
from functools import partial
from legged_gym.utils.math import quat_rotate

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        if args.no_wandb is not None:
            cfg_train.runner.no_wandb = args.no_wandb

    return env_cfg, cfg_train

def get_args(test=False):
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--no_wandb", "action": "store_true", "default": False,  "help": "Don't use wandb for logging"},

    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    args.test = test


    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

def TTT(*x, device: Optional[torch.device] = None):
    return torch.as_tensor(x,
                           device=device,
                           dtype=torch.float32)

def draw_axis(gym, viewer, env, txn, rxn, device: Optional[torch.device] = None):
    _TTT = partial(TTT, device=device)
    points = torch.stack([
        # X-axis
        txn,
        txn + quat_rotate(rxn, _TTT(0.2, 0, 0)[None, ...]),
        # Y-axis
        txn,
        txn + quat_rotate(rxn, _TTT(0, 0.2, 0)[None, ...]),
        # Z-axis
        txn,
        txn + quat_rotate(rxn, _TTT(0, 0, 0.2)[None, ...]),
    ])
    return gym.add_lines(viewer, env,
                         3, points.detach().cpu().numpy(),
                         torch.eye(3, device=device,
                                   dtype=torch.float32).detach().cpu().numpy()
                         )

def draw_bbox(gym, viewer, env_handle,
              pose: torch.Tensor,
              bbox: torch.Tensor,
              color=None):

    # indices = th.tensor([0,7],device=bbox.device)

    bbox_geom = gymutil.TrimeshBBoxGeometry(bbox, color=color)
    xfm = dcn(pose[..., :7])
    txn = xfm[..., :3]
    rxn = xfm[..., 3:7]
    # print(txn, th.mean(bbox,-2), bbox, bbox_geom.vertices())
    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3(*txn)
    obj_pose.r = gymapi.Quat(*rxn)
    colors = list(itertools.product([0, 1], repeat=3))
    for i, point in enumerate(bbox):
        point_pose = gymapi.Transform()
        point_pose.p = gymapi.Vec3(*point)
        ball_geom = gymutil.WireframeSphereGeometry(
            radius=0.01, pose=point_pose, color=colors[i])
        gymutil.draw_lines(
            ball_geom, gym, viewer, env_handle, None)
    return (gymutil.draw_lines(
        bbox_geom,
        gym,
        viewer,
        env_handle,
        None
    ))

def get_object_size(obj, seen=None):
    """
    Returns the total size of an object in bytes.
    If the object is a Tensor, calculate its memory size based on the number of elements.
    If the object is a dictionary, recursively calculate the size of all its elements.
    """
    if seen is None:
        seen = set()
    
    size = 0

    # Avoid double counting of already processed objects
    if id(obj) in seen:
        return 0
    seen.add(id(obj))

    if torch.is_tensor(obj):
        # If the object is a tensor, calculate its size
        size += obj.element_size() * obj.nelement()
    elif isinstance(obj, dict):
        # If the object is a dictionary, recursively calculate the size of its contents
        size += sys.getsizeof(obj)  # Account for dictionary overhead
        for key, value in obj.items():
            size += get_object_size(key, seen)   # Size of the key
            size += get_object_size(value, seen) # Size of the value
    else:
        # For other objects, use sys.getsizeof
        size += sys.getsizeof(obj)

    return size
def dcn(x: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor into numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
