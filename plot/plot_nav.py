import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import yourdfpy
import trimesh


def index_map(k_to, k_from):
    """
    Returns an index mapping from k_from to k_to.

    Given k_to=a, k_from=b,
    returns an index map "a_from_b" such that
    array_a[a_from_b] = array_b

    Missing values are set to -1.
    """
    index_dict = {k: i for i, k in enumerate(k_to)}  # O(len(k_from))
    return [index_dict.get(k, -1) for k in k_from]  # O(len(k_to))


def unique_consecutive(arr, return_index=False, return_counts=False):
    arr = np.asarray(arr)
    if arr.size == 0:
        result = (arr,)
        if return_index: result += (np.array([], dtype=int),)
        if return_counts: result += (np.array([], dtype=int),)
        return result if len(result) > 1 else result[0]

    # Mask of places where value changes
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    uniq = arr[change]

    result = (uniq,)

    if return_index:
        idx = np.nonzero(change)[0]
        result += (idx,)

    if return_counts:
        counts = np.diff(np.append(np.nonzero(change)[0], arr.size))
        result += (counts,)

    return result if len(result) > 1 else result[0]


def calculate_time_series_error(times1, values1, times2, values2, method="rmse"):
    """
    Calculate error between two time series with timestamps using NumPy only.
    
    Parameters:
        times1 (array-like): timestamps for series1 (in seconds or float)
        values1 (array-like): values for series1
        times2 (array-like): timestamps for series2
        values2 (array-like): values for series2
        method (str): 'mae', 'mse', or 'rmse'
        
    Returns:
        float: error value
    """
    times1 = np.asarray(times1, dtype=float)
    values1 = np.asarray(values1, dtype=float)
    times2 = np.asarray(times2, dtype=float)
    values2 = np.asarray(values2, dtype=float)

    # Interpolate series2 onto times1
    interp_values2 = np.zeros_like(values1)
    for d in range(values1.shape[1]):
        interp_values2[:, d] = np.interp(times1, times2, values2[:, d])

    # Compute differences
    diff = values1 - interp_values2

    if method == "mae":
        return np.mean(np.abs(diff))
    elif method == "mse":
        return np.mean(diff**2)
    elif method == "rmse":
        return np.sqrt(np.mean(diff**2))
    else:
        raise ValueError("method must be 'mae', 'mse', or 'rmse'")


motor_joint = [
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
    'right_wrist_yaw_joint',
]

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
    'left_shoulder_pitch_joint', #11
    'right_shoulder_pitch_joint', #12
    'left_ankle_pitch_joint', #13
    'right_ankle_pitch_joint', #14
    'left_shoulder_roll_joint', #15
    'right_shoulder_roll_joint', #16
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
    'left_shoulder_yaw_joint', #19
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


mot_from_lab = index_map(motor_joint, lab_joint)


# log_path = "/tmp/e2e/log_0911_l1_20k_1758022368.npy"
# log_path = "/tmp/e2e/log_0911_l1_20k_1758022916.npy"
# log_path = "/tmp/e2e/log_0911_l1_20k_1758023367.npy" # Old
# log_path = "/tmp/e2e/log_0911_l1_20k_1758027726.npy" # New
# log_path = "/tmp/e2e/log_0911_l1_20k_1758030285.npy"

log_path = "/tmp/e2e/log_0911_l1_20k_1758087392.npy" # 0917 test 1
log_path = "/tmp/e2e/log_0911_l1_20k_1758091899.npy" # 0917 test 2
log_path = "/tmp/e2e/log_0911_l1_20k_1758092474.npy" # 0917 test 3
log_path = "/tmp/e2e/log_0911_l1_20k_1758092928.npy" # 0917 test 4


log_path = "/tmp/e2e/log_0911_l1_20k_1758169212.npy"# nav test
log_path = "/tmp/e2e/log_0911_l1_20k_1758170517.npy"
log_path = "/tmp/e2e/log_0911_l1_20k_1758181377.npy"
log_path = "/tmp/e2e/log_0911_l1_20k_1758181829.npy"
log_path = "/tmp/e2e/log_0911_l1_20k_1758182460.npy" # Very close to the target, but not stopped and keep walking
log_path = "/tmp/e2e/log_0911_l1_20k_1758183447.npy" # directly stopped && using 10 sized window

log_path = "/tmp/e2e/log_0911_l1_20k_1758184529.npy"
log_path = "/tmp/e2e/log_0911_l1_20k_1758185459.npy" # far stopped
log_path = "/tmp/e2e/log_0911_l1_20k_1758189788.npy"

log_path = "/tmp/e2e/log_0911_l1_20k_1758268180.npy" # waist overheat
log_path = "/tmp/e2e/log_0911_l1_20k_1758269744.npy"
# log_path = "/tmp/e2e/log_0911_l1_20k_1758269945.npy"

log_path = "/tmp/e2e/log_0911_l1_20k_1758270599.npy"

# log_path = "/tmp/e2e/log_0911_l1_20k_1758271226.npy"

log_path = "/tmp/e2e/log_0911_l1_20k_1758368110.npy"

log_path = "/tmp/e2e/log_0921_l3_1758516026.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759058070.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759058513.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759061296.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759062198.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759062437.npy"

# log_path = "/tmp/e2e/log_0926_l3_30k_1759064520.npy"

# 50Hz
log_path = "/tmp/e2e/log_0926_l3_30k_1759110016.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759110404.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759111914.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759112769.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759113011.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759113517.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759116383.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759116901.npy"

# GOOOD
log_path = "/tmp/e2e/log_0926_l3_30k_1759117282.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759117516.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759117854.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759118104.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759118562.npy"

log_path = "/tmp/e2e/log_0926_l3_30k_1759118861.npy"


# USUAL
log_path = "/tmp/e2e/log_0926_l3_30k_1759119373.npy"
log_path = "/tmp/e2e/log_0926_l3_30k_1759119885.npy"

# OUTLIER NOISE
log_path = "/tmp/e2e/log_0926_l3_30k_1759119557.npy"


data = np.load(log_path, allow_pickle=True).item()
traj_data = data["trajectories"]
# print(traj_data.keys())

urdf = yourdfpy.URDF.load("../resources/robots/g1_description/g1_29dof_rev_1_0_zed2i_with_welder_v3.urdf")

t_h = traj_data["timestamp_high_freq"]
t_l = traj_data["timestamp_low_freq"]
tasks = traj_data["tasks"]
# task_unique, task_counts = unique_consecutive(tasks.flatten(), return_counts=True)
# task_changed_ids = task_counts.cumsum()[:-1]
# trajopt_ids = (tasks == "trajopt").nonzero()[0]
# q_traj = np.zeros_like(traj_data["q_traj"])
# q_traj[:, mot_from_lab] = traj_data["q_traj"]
# tau_traj = np.zeros_like(traj_data["tau_traj"])
# tau_traj[:, mot_from_lab] = traj_data["tau_traj"]
# target_poses_w = traj_data["target_poses_w"]
# target_poses_b = traj_data["target_poses_b"]
# root_states_w = traj_data["root_states_w"]
# ee_poses_w = traj_data["ee_poses_w"]
# ee_poses_b = traj_data["ee_poses_b"]
vel_cmd_b = traj_data["locomotion_vel_cmd"]
pos_error_b = traj_data["pos_command_bs"]

window_size_10 = traj_data["errors_avg_10"]
window_size_20 = traj_data["errors_avg_20"]
window_size_30 = traj_data["errors_avg_30"]
window_size_40 = traj_data["errors_avg_40"]


trans_thresh = 0.04
rot_thresh = 0.1

time_to_start = 10

plt.subplot(4,1,1)
plt.plot(window_size_10[time_to_start:,0], label="window: 10, trans")
plt.plot(window_size_10[time_to_start:,1], label="window: 10, rot")
plt.axhline(y=trans_thresh, linestyle=":", c="r", label="trans threshold")
plt.axhline(y=rot_thresh, linestyle=":", c="g", label="rot threshold")
plt.legend()
plt.subplot(4,1,2)
plt.plot(window_size_20[time_to_start:,0], label="window: 20, trans")
plt.plot(window_size_20[time_to_start:,1], label="window: 20, rot")
plt.axhline(y=trans_thresh, linestyle=":", c="r", label="trans threshold")
plt.axhline(y=rot_thresh, linestyle=":", c="g", label="rot threshold")
plt.legend()
plt.subplot(4,1,3)
plt.plot(window_size_30[time_to_start:,0], label="window: 30, trans")
plt.plot(window_size_30[time_to_start:,1], label="window: 30, rot")
plt.axhline(y=trans_thresh, linestyle=":", c="r", label="trans threshold")
plt.axhline(y=rot_thresh, linestyle=":", c="g", label="rot threshold")
plt.legend()
plt.subplot(4,1,4)
plt.plot(window_size_40[time_to_start:,0], label="window: 40, trans")
plt.plot(window_size_40[time_to_start:,1], label="window: 40, rot")
plt.axhline(y=trans_thresh, linestyle=":", c="r", label="trans threshold")
plt.axhline(y=rot_thresh, linestyle=":", c="g", label="rot threshold")
plt.legend()
plt.show()



# navigation_ids = (tasks=="navigation").nonzero()[0].flatten()

print("translational error : ", np.linalg.norm(pos_error_b[-1]))
print("x : ", abs(pos_error_b[-1, 0]))
print("y : ", abs(pos_error_b[-1, 1]))

labels = ["x", "y", "yaw"]

navigation_ids = (tasks=="navigation").nonzero()[0].flatten()
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(vel_cmd_b[navigation_ids, i][time_to_start:], label=f"{labels[i]} vel")
    plt.plot(pos_error_b[:, i][time_to_start:], label=f"{labels[i]} error")
    plt.axhline(y=0, c="r")

    plt.legend()
    # plt.ylim((-0.1, 0.1))
plt.show()

plt.plot(np.linalg.norm(vel_cmd_b[navigation_ids, :], axis=1)[time_to_start:], label="velocity norm")
plt.legend()

plt.show()

# last_n = 8000
# ee_T_bs = []
# for q in q_traj[-last_n:]:
#     urdf.update_cfg(q)
#     ee_T_b = urdf.get_transform("end_effector")
#     ee_T_bs.append(ee_T_b)
# ee_T_bs = np.stack(ee_T_b)

# eetrack_ids = (tasks=="eetrack").nonzero()[0].flatten()
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.plot(t_l[eetrack_ids], ee_poses_w[:,i], label="cur")
#     plt.plot(t_l[eetrack_ids], target_poses_w[:,i], label="target")
# plt.legend()
# plt.show()

# trimesh.Scene([
#     trimesh.creation.axis(),
#     trimesh.PointCloud(target_poses_w[:,:3], [0,255,0]),
#     trimesh.PointCloud(ee_poses_w[:,:3], [255,0,0]),
# ]).show()

# np.set_printoptions(suppress=True)
# print(q_traj[-1])

# plt.figure(figsize=(25,15))
# for i in range(29):
#     plt.subplot(6,5,i+1)
#     plt.plot(t_h, q_traj[:,i], label="q")
#     plt.title(motor_joint[i])
# plt.legend()
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(25,15))
for i in range(29):
    plt.subplot(6,5,i+1)
    plt.plot(t_h[1:], tau_traj[:,i], label="tau")
    plt.title(motor_joint[i])
    for tc_id in task_changed_ids:
        plt.axvline(t_l[tc_id], c='r', ls='--')
plt.legend()
plt.tight_layout()
plt.show()

# target_trajopt_joint_pos = traj_data["target_trajopt_joint_pos"]

# valid_trajopt_ids = trajopt_ids[(target_trajopt_joint_pos).any(axis=1)]
# valid_target_trajopt_joint_pos = target_trajopt_joint_pos[(target_trajopt_joint_pos).any(axis=1)]

# trajopt_t_l = t_l[valid_trajopt_ids]
# trajopt_start_t = trajopt_t_l[0]
# trajopt_end_t = trajopt_t_l[-1]

# trajopt_high_freq_mask = (trajopt_start_t < t_h) & (t_h < trajopt_end_t)
# trajopt_t_h = t_h[trajopt_high_freq_mask]
# trajopt_q_traj = q_traj[trajopt_high_freq_mask]

# trajopt_err = calculate_time_series_error(trajopt_t_h, trajopt_q_traj[:,-7:], trajopt_t_l, valid_target_trajopt_joint_pos[:,-7:], method="rmse")
# # trajopt_err = calculate_time_series_error(trajopt_t_l, valid_target_trajopt_joint_pos[:,-7:], trajopt_t_h, trajopt_q_traj[:,-7:], method="rmse")
# print("TrajOpt right arm joint pos following rmse error: ", trajopt_err)

# plt.figure(figsize=(25,15))
# for i in range(29):
#     plt.subplot(6,5,i+1)
#     plt.plot(t_h, q_traj[:,i], label="q")
#     plt.plot(trajopt_t_l, valid_target_trajopt_joint_pos[:,i], label="q_trg")
#     plt.title(motor_joint[i])
#     for tc_id in task_changed_ids:
#         plt.axvline(t_l[tc_id], c='r', ls='--')
# plt.legend()
# plt.tight_layout()
# plt.show()

vision_trajopt_ids = ((tasks == "vision") | (tasks == "trajopt")).nonzero()[0]
# vision_trajopt_target_dof_pos = traj_data["target_dof_pos"][vision_trajopt_ids]
# other_target_dof_pos_changed = len(np.unique(vision_trajopt_target_dof_pos[:,:-7], axis=0)) > 1
# print("Target joints other than right arm changed:", other_target_dof_pos_changed)

vision_trajopt_root_states_w = root_states_w[vision_trajopt_ids]
vision_trajopt_root_rpy = R.from_quat(np.roll(vision_trajopt_root_states_w[:,3:],-1)).as_euler("xyz", degrees=True)
print("Maximum pelvis x changed during vision and trajopt:", 
      vision_trajopt_root_states_w[:,0].max() - vision_trajopt_root_states_w[:,0].min())
print("Maximum pelvis y changed during vision and trajopt:", 
      vision_trajopt_root_states_w[:,1].max() - vision_trajopt_root_states_w[:,1].min())
print("Maximum pelvis z changed during vision and trajopt:", 
      vision_trajopt_root_states_w[:,2].max() - vision_trajopt_root_states_w[:,2].min())
print("Maximum pelvis roll changed during vision and trajopt (deg):", 
      vision_trajopt_root_rpy[:,0].max() - vision_trajopt_root_rpy[:,0].min())
print("Maximum pelvis pitch changed during vision and trajopt (deg):", 
      vision_trajopt_root_rpy[:,1].max() - vision_trajopt_root_rpy[:,1].min())
print("Maximum pelvis yaw changed during vision and trajopt (deg):", 
      vision_trajopt_root_rpy[:,2].max() - vision_trajopt_root_rpy[:,2].min())
