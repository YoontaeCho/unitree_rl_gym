import numpy as np
from scipy.spatial.transform import Rotation as R

def world_to_local(delta_world, R_local_in_world):
    """
    Convert displacement from world frame to local frame.
    
    Parameters:
    -----------
    delta_world : np.ndarray, shape (3,)
        Displacement vector in world frame (dx, dy, dz).
    R_local_in_world : np.ndarray, shape (3,3)
        Rotation matrix of the local frame expressed in world frame.
        (i.e., transforms local -> world)

    Returns:
    --------
    delta_local : np.ndarray, shape (3,)
        Displacement vector expressed in local frame.
    """
    # Transpose (inverse) to convert world -> local
    R_world_to_local = R_local_in_world.T
    delta_local = R_world_to_local @ delta_world
    return delta_local

data_path_list = [
    # "/tmp/e2e/log_0901_f1_20k_1757743326.npy",
    # "/tmp/e2e/log_0901_f1_20k_1757745554.npy",
    # "/tmp/e2e/log_0901_f1_20k_1757746762.npy",
    # TODO: add
    # /tmp/e2e/log_0901_f1_20k_1757754202.npy
]
delat_w_list = np.array([
    # [-0.01, 0.002, -0.026],
    # [-0.008, -0.002, -0.02],
    # [-0.012, 0.001, -0.018],
    # TODO: add
    # dx: 0.02100000000000001, dy: -0.03400000000000002, dz: 0.06500000000000004

])
data_list = [np.load(data_path, allow_pickle=True).item() for data_path in data_path_list]
delta_cam_list = []
delta_cam_parent_list = []
for i, (data, delta_w) in enumerate(zip(data_list, delat_w_list)):
    cam_pose_w = data["trajectories"]["zed_poses_w"][-1]
    cam_R_w = R.from_quat(np.roll(cam_pose_w[3:], 1)).as_matrix()
    delta_cam = world_to_local(delta_w, cam_R_w)
    delta_cam_parent = np.array([delta_cam[2], -delta_cam[0], -delta_cam[1]])
    print(f"delta_cam_{i}", delta_cam)
    print(f"delta_cam_{i} in parents", delta_cam_parent)
    # print(R.from_quat(np.roll(cam_pose_w[3:], 1)).inv().apply(delta_w))
    delta_cam_list.append(delta_cam)
    delta_cam_parent_list.append(delta_cam_parent)

print("Mean delta cam", np.stack(delta_cam_list).mean(axis=0))
print("Mean delta cam in parent", np.stack(delta_cam_parent_list).mean(axis=0))
