import time
from pathlib import Path
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import random as rd
import math_utils as math_th
import math_utils_np as math_np
from xml_helper import extract_link_data
from typing import List, Union
import pinocchio as pin
from ikctrl import IKCtrl, xyzw2wxyz
from config import Config
from eetrack import EETrack
from act_to_dof import ActToDof, index_map
DEBUG = True
from math_utils import (
    as_np,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate_inverse,
    axis_angle_from_quat,
    wrap_to_pi,
    subtract_frame_transforms,
    quat_rotate,
    quat_apply_yaw,
    quat_from_euler_xyz,
    compute_pose_error
)
quat_from_angle_axis = as_np(quat_from_angle_axis)
quat_mul = as_np(quat_mul)
quat_rotate_inverse = as_np(quat_rotate_inverse)
axis_angle_from_quat = as_np(axis_angle_from_quat)
wrap_to_pi = as_np(wrap_to_pi)
subtract_frame_transforms = as_np(subtract_frame_transforms)
quat_rotate = as_np(quat_rotate)
quat_apply_yaw = as_np(quat_apply_yaw)
quat_from_euler_xyz = as_np(quat_from_euler_xyz)
compute_pose_error = as_np(compute_pose_error)



def print_obs(obs):
    # print("base_ang_vel", obs[0:3])
    # print("projected_gravity", obs[3:6])
    # print("foot_pose", obs[6:18])
    print("hand_pose", obs[18:30])
    # print("projected_com", obs[30:32])
    # print("joint_pos", obs[32:61])
    # print("joint_vel", obs[61:90])
    # print("actions", obs[90:119])
    print("hands_command", obs[119:125])
    # print("right_arm_com", obs[125:128])
    # print("left_arm_com", obs[128:131])
    # print("pelvis_height", obs[131:132])

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def get_link_pose_quat_world_frame(model, data, link_name):
    link_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    if (link_idx == -1):
        print(f"Link {link_name} not found in model")
        exit()
    return (data.xpos[link_idx],data.xquat[link_idx])

def get_link_pose_quat_root_frame(model, data, link_name):
    root_pos, root_quat = get_link_pose_quat_world_frame(model, data, "pelvis")
    link_pos, link_quat = get_link_pose_quat_world_frame(model, data, link_name)
    return subtract_frame_transforms(root_pos, root_quat, link_pos, link_quat)

def get_foot_pos(model, data):
    fp_l_p, fp_l_q = get_link_pose_quat_root_frame(model, data, "left_ankle_roll_link")
    fp_r_p, fp_r_q = get_link_pose_quat_root_frame(model, data, "right_ankle_roll_link")
    fp_l_a = wrap_to_pi(axis_angle_from_quat(fp_l_q))
    fp_r_a = wrap_to_pi(axis_angle_from_quat(fp_r_q))
    return np.concatenate([fp_l_p, fp_r_p, fp_l_a, fp_r_a])

def get_hand_pos(model, data):
    hp_l_p, hp_l_q = get_link_pose_quat_root_frame(model, data, "left_wrist_yaw_link")
    hp_r_p, hp_r_q = get_link_pose_quat_root_frame(model, data, "right_wrist_yaw_link")
    hp_l_a = wrap_to_pi(axis_angle_from_quat(hp_l_q))
    hp_r_a = wrap_to_pi(axis_angle_from_quat(hp_r_q))
    return np.concatenate([hp_l_p, hp_r_p, hp_l_a, hp_r_a])

def compute_com(model, data, body_frames: Union[List, None] = None):
    com_data = extract_link_data(config.xml_path)
    mass_list = []
    com_list = []
    if body_frames is None:
        body_frames = com_data.keys()
    for frame in body_frames:
        try:
            frame_data = com_data[frame]
        except KeyError:
            continue
        try:
            link_pos, link_wxyz = get_link_pose_quat_root_frame(model, data, frame)
        except:
            continue

        com_pos_b, com_wxyz = frame_data['pos'], frame_data['quat']

        # compute com from world coordinates
        # NOTE 'math_utils' package will be brought from isaaclab
        com_pos = link_pos + quat_rotate(link_wxyz, com_pos_b)
        com_list.append(com_pos)

        # get math
        mass = frame_data['mass']
        mass_list.append(mass)

    com = sum([m * pos for m, pos in zip(mass_list, com_list)]) / sum(mass_list)
    return com

def get_left_arm_com(model, data):
    return compute_com(model, data,[
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_pitch_link"
        "left_wrist_roll_link",
        "left_wrist_yaw_link"
    ])

def get_right_arm_com(model, data):
    return compute_com(model, data,[
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_pitch_link"
        "right_wrist_roll_link",
        "right_wrist_yaw_link"
    ])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd
    # return (target_q - q) * kp * 0.1 + (target_dq - dq) * kd * 0.1


if __name__ == "__main__":
    # get config file name from command line
    isaaclab_joint_order = [
        'left_hip_pitch_joint',
        'right_hip_pitch_joint',
        'waist_yaw_joint',
        'left_hip_roll_joint',
        'right_hip_roll_joint',
        'waist_roll_joint',
        'left_hip_yaw_joint',
        'right_hip_yaw_joint',
        'waist_pitch_joint',
        'left_knee_joint',
        'right_knee_joint',
        'left_shoulder_pitch_joint',
        'right_shoulder_pitch_joint',
        'left_ankle_pitch_joint',
        'right_ankle_pitch_joint',
        'left_shoulder_roll_joint',
        'right_shoulder_roll_joint',
        'left_ankle_roll_joint',
        'right_ankle_roll_joint',
        'left_shoulder_yaw_joint',
        'right_shoulder_yaw_joint',
        'left_elbow_joint',
        'right_elbow_joint',
        'left_wrist_roll_joint',
        'right_wrist_roll_joint',
        'left_wrist_pitch_joint',
        'right_wrist_pitch_joint',
        'left_wrist_yaw_joint',
        'right_wrist_yaw_joint'
    ]

    raw_joint_order = [
        'left_hip_pitch_joint',
        'left_hip_roll_joint',
        'left_hip_yaw_joint',
        'left_knee_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
        'right_hip_pitch_joint',
        'right_hip_roll_joint',
        'right_hip_yaw_joint',
        'right_knee_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint',
        'waist_yaw_joint',
        'waist_roll_joint',
        'waist_pitch_joint',
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint',
        'left_elbow_joint',
        'left_wrist_roll_joint',
        'left_wrist_pitch_joint',
        'left_wrist_yaw_joint',
        'right_shoulder_pitch_joint',
        'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint',
        'right_elbow_joint',
        'right_wrist_roll_joint',
        'right_wrist_pitch_joint',
        'right_wrist_yaw_joint'
    ]

    # Create a mapping tensor
    # mapping_tensor = torch.zeros((len(sim_b_joints), len(sim_a_joints)), device=env.device)
    mapping_tensor = torch.zeros((len(raw_joint_order), len(isaaclab_joint_order)))

    # Fill the mapping tensor
    for b_idx, b_joint in enumerate(raw_joint_order):
        if b_joint in isaaclab_joint_order:
            a_idx = isaaclab_joint_order.index(b_joint)
            # mapping_tensor[b_idx, a_idx] = 1.0
            mapping_tensor[a_idx, b_idx] = 1.0

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("use_log", type=int, help="use log or not")
    args = parser.parse_args()
    USE_LOG = args.use_log
    config_file = args.config_file
    config = Config(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}")
    # define context variables
    action = np.zeros(config.num_actions, dtype=np.float32)
    target_dof_pos = config.default_angles.copy()
    target_dof_eff = 0
    obs = np.zeros(config.num_obs, dtype=np.float32)
    ikctrl = IKCtrl(
        # "../../resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf",
        "../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf",
        config.arm_joint
    )
    actmap = ActToDof(config, ikctrl)  
    counter = 0

    # Load robot model

    m = mujoco.MjModel.from_xml_path(config.xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config.simulation_dt
    policy = torch.jit.load(config.policy_path)
    logpath = Path("deploy_log/eet9")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        input("Press Enter to start simulation.")
        start = time.time()
        eetrack = EETrack(d.qpos[:7])
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        while viewer.is_running() and time.time() - start < config.simulation_duration:
            step_start = time.time()
            tau = pd_control(
                target_dof_pos,
                d.qpos[7:],
                np.multiply(1.0, config.kps),
                np.zeros_like(config.kds),
                d.qvel[6:],
                np.multiply(1.0, config.kds)
            )
            d.ctrl[:] = tau #+ target_dof_eff
            mujoco.mj_step(m, d)
            counter += 1
            if counter % config.control_decimation == 0:
                # input()
                # eetrack visualization
                eetrack.vis(viewer)
                print("=================== STEP ===================")
                # create observation
                """
                base_ang_vel 0:3
                projected_gravity 3:6
                foot_pose 6:18
                hand_pose 18:30
                projected_com 30:32
                joint_pos 32:61
                joint_vel 61:90
                actions 90:119
                hands_command 119:125
                right_arm_com 125:128
                left_arm_com 128:131
                pelvis_height 131:132
                """
                # base_ang_vel 0:3
                base_ang_vel = quat_rotate_inverse(d.qpos[3:7], d.qvel[3:6])
                # projected_gravity 3:6
                projected_gravity = get_gravity_orientation(d.qpos[3:7])
                # foot_pose 6:18
                foot_pose = get_foot_pos(m,d)
                # hand_pose 18:30
                hand_pose = get_hand_pos(m,d)
                # projected_com 30:32
                projected_com = compute_com(m,d)[:2]
                projected_com += np.random.normal(size=projected_com.shape) * 0.01

                # joint_pos 32:61

                joint_pos = d.qpos[7:] # raw joint order
                # joint_vel 61:90
                joint_vel = d.qvel[6:] # raw joint order
                # actions 90:119
                actions = action # lab joint order
                # hands_command 119:125
                root_state_w = np.concatenate(get_link_pose_quat_world_frame(m, d, "pelvis"))
                hand_state_w = np.concatenate(get_link_pose_quat_world_frame(m, d, "left_wrist_yaw_link"))
                
                print("hand_root  : ", np.concatenate(get_link_pose_quat_root_frame(m, d, "left_wrist_yaw_link")))
                # print("hand_world : ", np.concatenate(get_link_pose_quat_world_frame(m, d, "left_hand_palm_link")))
                hands_command = eetrack.get_command(root_state_w, hand_state_w)
                # hands_command = np.zeros(6)
                # right_arm_com 125:128
                right_arm_com = get_right_arm_com(m,d) 
                right_arm_com += np.random.normal(size=right_arm_com.shape) * 0.01
                # left_arm_com 128:131
                left_arm_com = get_left_arm_com(m,d)
                left_arm_com += np.random.normal(size=left_arm_com.shape) * 0.01
                # pelvis_height 131:132
                # pelvis_height = np.asarray([get_link_pose_quat_world_frame(m, d, "pelvis")[0][2]])
                pelvis_height = np.asarray([d.qpos[2]])

                obs =[
                    base_ang_vel,
                    projected_gravity,
                    foot_pose,
                    hand_pose,
                    projected_com,
                    joint_pos,
                    joint_vel,
                    actions,
                    hands_command,
                    right_arm_com,
                    left_arm_com,
                    pelvis_height
                ]

                obs = np.concatenate(obs, axis=-1)
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                # raw -> lab for joint pos and vel
                obs_tensor[..., 32:61] = obs_tensor[..., 32:61] @ mapping_tensor.transpose(0, 1)
                obs_tensor[..., 61:90] = obs_tensor[..., 61:90] @ mapping_tensor.transpose(0, 1)
                # subtract joint offset
                obs_tensor[..., 32:61] -= torch.Tensor(config.lab_joint_offsets)


                print("muj obs")
                print_obs(obs)
                if USE_LOG :
                    obs = np.load(F"{logpath}/obs{counter:03d}.npy")
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    obs_tensor = obs_tensor.detach().clone()
                    print("log obs")
                    print_obs(obs)

                obs_tensor.add_(1e-3 * torch.randn_like(obs_tensor))
                action = policy(obs_tensor).detach().numpy().squeeze()

                if USE_LOG :
                    action = np.load(F"{logpath}/act{counter:03d}.npy")

                # act_to_dof need lab joint order
                obs = obs_tensor.numpy().squeeze()

                # solve IK
                target_dof_pos, target_dof_eff = actmap(obs, action) # raw joint order
                # if USE_LOG :
                #     print("muj dof", target_dof_pos)
                #     target_dof_pos = np.load(F"{logpath}/dof{counter:03d}.npy")
                #     print("log dof", target_dof_pos)
                # smoothing
                scale = 0.3
                target_dof_pos = (scale * d.qpos[7:] + (1-scale) * target_dof_pos)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
