import numpy as np
from scipy.spatial.transform import Rotation as R

def multiply_tf(pos1, quat1, pos2, quat2):
    # Convert quaternions to Rotation objects
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    # Chained rotation
    r_new = r1 * r2
    # Rotate pos2 by r1, then add pos1
    pos_new = pos1 + r1.apply(pos2)
    quat_new = r_new.as_quat()
    return pos_new, quat_new


def quat_from_yaw(yaw):
    # Returns quaternion [x, y, z, w] from yaw angle (rad)
    return R.from_euler('z', yaw).as_quat()

print(quat_from_yaw(0.7085))

# midsole_to_cam = [-0.50835566, 0.17516229, -0.4933455, 0.68374185]
# # Apply z: -90deg, x: +90deg rotation to midsole_to_cam sequentially
# rot_z = R.from_euler('z', -90, degrees=True)
# rot_x = R.from_euler('x', 90, degrees=True)
# rot_total = rot_x * rot_z
# midsole_to_cam_rot = rot_total.apply(midsole_to_cam[:3])
# midsole_to_cam_quat = (rot_total * R.from_quat(midsole_to_cam)).as_quat()
# print("midsole_to_cam_rot:", midsole_to_cam_rot)
# print("midsole_to_cam_quat:", midsole_to_cam_quat)

# Example usage:
pos1, quat1 = np.array([-0.49950311, 0.16916784, -0.3]), np.array([0. ,        0.   ,      0.34688703, 0.93790692])
pos2, quat2 = np.array([0.24771635, 0.04083803, 0.71193068]), np.array([ -0.49328178,  0.68369785, -0.50844262, 0.17526106])
pos_new, quat_new = multiply_tf(pos1, quat1, pos2, quat2)


print(pos_new, quat_new)

# # Convert quat_new to roll, pitch, yaw (degrees)
rpy = R.from_quat(quat_new).as_euler('xyz', degrees=True)
print("RPY (deg):", rpy)

import numpy as np
from scipy.spatial.transform import Rotation as R

# original RPY in degrees
rpy = [-115.89941792, -15.18637807, -77.32771825]

# build rotation matrix
R_old = R.from_euler('xyz', rpy, degrees=True).as_matrix()

# old axes
x_old, y_old, z_old = R_old[:,0], R_old[:,1], R_old[:,2]

# new axes (replace x with old z)
x_new = z_old
z_new = x_old
y_new = np.cross(z_new, x_new)   # ensure right-handed frame

# build new rotation matrix
R_new = np.column_stack([x_new, y_new, z_new])

# convert back to RPY (deg)
rpy_new = R.from_matrix(R_new).as_euler('xyz', degrees=True)
quat = R.from_matrix(R_new).as_quat()
print(rpy_new)
print(quat)