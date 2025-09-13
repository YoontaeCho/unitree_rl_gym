import numpy as np
import matplotlib.pyplot as plt
import math_utils

euler_xyz_from_quat = math_utils.as_np(math_utils.euler_xyz_from_quat)

from pathlib import Path

lab_joint = [
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

def plot_log(file_path):
    data = np.load(file_path, allow_pickle=True).item()

    target_poses_w = data["trajectories"]["target_poses_w"]
    target_poses_b = data["trajectories"]["target_poses_b"]

    target_pos_b, target_quat_w = target_poses_b[:,:3], target_poses_b[:,3:]
    target_rpy_b = euler_xyz_from_quat(target_quat_w)

    changed = np.diff(target_pos_b[:,:3], axis=0).any(axis=1).nonzero()[0]
    # plt.subplot(2,3,1)
    plt.plot(target_pos_b[:,0], label='x')
    # plt.subplot(2,3,2)
    plt.plot(target_pos_b[:,1], label='y')
    # plt.subplot(2,3,3)
    plt.plot(target_pos_b[:,2], label='z')
    plt.plot(target_rpy_b[1], label='pitch')
    plt.legend()
    # for x in changed:
    #     plt.axvline(x, c='r')

            
    plt.tight_layout()
    plt.show()
    
    # # Plot
    # num_joints = q_traj.shape[1]    
    # fig, axs = plt.subplots(5, 6, figsize=(20, 15))
    # axs = axs.flatten()
    
    # for idx in range(len(axs)):
    #     if idx >= num_joints:
    #         axs[idx].axis('off')
    #         continue

    #     axs[idx].plot(timestamp_high_freq, q_traj[:, idx], color='r', label='q_traj')
    #     axs[idx].plot(timestamp_low_freq, raw_joint_pos_targets[:, idx], color='g', label='raw_target')
    #     axs[idx].plot(timestamp_low_freq, joint_pos_targets[:, idx], color='b', label='filtered_target')
    #     axs[idx].set_title(f"{lab_joint[idx]}")
        
    #     # axs[idx].plot(timestamp_high_freq, dq_traj[:, idx], color='r', label='dq_traj')
        
    #     # axs[idx].legend()
            
    # plt.tight_layout()   
    # plt.show()

def main():
    file_path = "log_sit_ver4_ikctrl_1757658661.npy"
    plot_log(file_path)

if __name__ == "__main__":
    main()