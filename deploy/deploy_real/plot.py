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

def plot_log(file_name, start_time, end_time):
    dir_path = Path('/tmp/eetrack_stand/')
    file_path = dir_path / file_name
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
    
    # Filter the data
    t_start = timestamp_high_freq[-1] + start_time
    t_end = timestamp_high_freq[-1] + end_time
    high_freq_filter = np.logical_and(timestamp_high_freq >= t_start, timestamp_high_freq <= t_end)
    low_freq_filter = np.logical_and(timestamp_low_freq >= t_start, timestamp_low_freq <= t_end)
    
    timestamp_high_freq = timestamp_high_freq[high_freq_filter]
    q_traj = q_traj[high_freq_filter]
    dq_traj = dq_traj[high_freq_filter]
    tau_traj = tau_traj[high_freq_filter]
    
    timestamp_low_freq = timestamp_low_freq[low_freq_filter]
    observations = observations[low_freq_filter]
    actions = actions[low_freq_filter]
    raw_joint_pos_targets = raw_joint_pos_targets[low_freq_filter]
    joint_pos_targets = joint_pos_targets[low_freq_filter]
    
    # Plot
    num_joints = q_traj.shape[1]    
    fig, axs = plt.subplots(5, 6, figsize=(20, 15))
    axs = axs.flatten()
    
    for idx in range(len(axs)):
        if idx >= num_joints:
            axs[idx].axis('off')
            continue

        axs[idx].plot(timestamp_high_freq, q_traj[:, idx], color='r', label='q_traj')
        axs[idx].plot(timestamp_low_freq, raw_joint_pos_targets[:, idx], color='g', label='raw_target')
        axs[idx].plot(timestamp_low_freq, joint_pos_targets[:, idx], color='b', label='filtered_target')
        axs[idx].set_title(f"{lab_joint[idx]}")
        
        # axs[idx].plot(timestamp_high_freq, dq_traj[:, idx], color='r', label='dq_traj')
        
        axs[idx].legend()
            
    plt.tight_layout()   
    plt.show()

def main():
    # NOTE: change the file_name
    file_name = "log"
    
    # NOTE: change the start_time and end_time
    start_time = -5
    end_time = 0
    plot_log(file_name, start_time, end_time)

if __name__ == "__main__":
    main()