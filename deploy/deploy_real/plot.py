import numpy as np
import matplotlib.pyplot as plt

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

def plot_log(file_path, start_time, end_time):
    data = np.load(file_path, allow_pickle=True).item()
    
    timestamp_high_freq = data['trajectories']['timestamp_high_freq']
    q_traj = data['trajectories']['q_traj']
    dq_traj = data['trajectories']['dq_traj']
    tau_traj = data['trajectories']['tau_traj']
    
    timestamp_low_freq = data['trajectories']['timestamp_low_freq']
    observations = data['trajectories']['observations']
    actions = data['trajectories']['actions']
    raw_joint_pos_targets = data['trajectories']['raw_joint_pos_targets']
    joint_pos_targets = data['trajectories']['joint_pos_targets']

    # Calculate dof pos jitter
    # |q_t - 2 * q_t-1 + q_t-2|
    # plot jitter by timestamp_high_freq vs jitter
    jitter = q_traj[:-2] - 2 * q_traj[1:-1] + q_traj[2:]
    timestamp = timestamp_high_freq[1:-1]

    # Plot
    num_joints = q_traj.shape[1]    
    fig, axs = plt.subplots(5, 6, figsize=(20, 15))
    axs = axs.flatten()

    start_idx = -1000
    
    for idx in range(len(axs)):
        if idx >= num_joints:
            axs[idx].axis('off')
            continue

        axs[idx].plot(timestamp[start_idx:], jitter[start_idx:, idx], color='r', label='jitter')
        axs[idx].set_title(f"{lab_joint[idx]}")
        axs[idx].set_ylim(-0.02, 0.02)
        
        # axs[idx].plot(timestamp_high_freq, dq_traj[:, idx], color='r', label='dq_traj')
        
        # axs[idx].legend()
            
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
    # NOTE: change the file_path
    file_path = "/tmp/eetrack_stand/log_daop_sit_exported_init_0_8_later_1_0_kpkd_1_0_height_0_3_1743163059.npy"
    
    # NOTE: change the start_time and end_time
    start_time = -5
    end_time = 0
    plot_log(file_path, start_time, end_time)

if __name__ == "__main__":
    main()