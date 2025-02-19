import numpy as np
from typing import Optional, Tuple, Literal

def normalize(quat: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion along the last axis.
    """
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return quat / norm

def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """
    Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = np.copy(quat).reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    
    quat_yaw[:] = 0.0
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw[:, 3] = np.sin(yaw / 2)
    
    quat_yaw = normalize(quat_yaw)
    return quat_yaw.reshape(shape)

def quat_apply(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    
    xyz = quat[:, 1:]
    t = 2 * np.cross(xyz, vec)
    rotated = vec + quat[:, 0:1] * t + np.cross(xyz, t)
    
    return rotated.reshape(shape)

def quat_apply_yaw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Rotate a vector only around the yaw-direction.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (N, 4).
        vec: The vector in (x, y, z). Shape is (N, 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (N, 3).
    """
    quat_yaw = yaw_quat(quat)
    return quat_apply(quat_yaw, vec)

def quat_from_euler_xyz(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to Quaternions.

    Note:
        The Euler angles are assumed in XYZ convention.

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qw, qx, qy, qz], axis=-1)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: If the input shapes of `q1` and `q2` are not matching.
    """
    if q1.shape != q2.shape:
        raise ValueError(f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}.")

    shape = q1.shape
    # reshape to (N, 4) for multiplication
    q1_flat = q1.reshape(-1, 4)
    q2_flat = q2.reshape(-1, 4)
    
    # extract components from quaternions
    w1, x1, y1, z1 = q1_flat[:, 0], q1_flat[:, 1], q1_flat[:, 2], q1_flat[:, 3]
    w2, x2, y2, z2 = q2_flat[:, 0], q2_flat[:, 1], q2_flat[:, 2], q2_flat[:, 3]
    
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    
    result = np.stack([w, x, y, z], axis=-1)
    return result.reshape(shape)

def normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Normalizes the input tensor to unit length along the last axis.
    
    Args:
        x: Input array of shape (..., dims).
        eps: 작은 값으로, 0으로 나누는 것을 방지합니다.
        
    Returns:
        정규화된 배열, shape는 (..., dims).
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Computes the conjugate of a quaternion.
    
    Args:
        q: Quaternion in (w, x, y, z) form. Shape is (..., 4).
    
    Returns:
        Conjugated quaternion with the same shape.
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    q_conj = np.concatenate([q[:, 0:1], -q[:, 1:]], axis=-1)
    return q_conj.reshape(shape)

def quat_inv(q: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a quaternion.
    
    Args:
        q: Quaternion in (w, x, y, z) form. Shape is (N, 4).
    
    Returns:
        The inverse quaternion in (w, x, y, z) with shape (N, 4).
    """
    return normalize(quat_conjugate(q))

def subtract_frame_transforms(
    t01: np.ndarray,
    q01: np.ndarray,
    t02: Optional[np.ndarray] = None,
    q02: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Subtract transformations between two reference frames into a stationary frame.
    
    It performs the following transformation operation:
        T12 = T01^{-1} x T02,
    where T_AB is the homogeneous transformation matrix from frame A to B.
    
    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
             Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
             Defaults to None, in which case the orientation is assumed to be identity.
    
    Returns:
        A tuple (t12, q12), where:
            t12: Position of frame 2 w.r.t. frame 1. Shape is (N, 3).
            q12: Orientation of frame 2 w.r.t. frame 1 in quaternion (w, x, y, z). Shape is (N, 4).
    """
    q10 = quat_inv(q01)
    
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    
    return t12, q12

def axis_angle_from_quat(quat: np.array, eps: float = 1.0e-6) -> np.array:
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[0:1] < 0.0))
    mag = np.linalg.norm(quat[1:], axis=-1)
    half_angle = np.arctan2(mag, quat[0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = np.where(
        np.abs(angle) > eps, np.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[1:4] / sin_half_angles_over_angles

def compute_pose_error(
    t01: np.ndarray,
    q01: np.ndarray,
    t02: np.ndarray,
    q02: np.ndarray,
    rot_error_type: Literal["quat", "axis_angle"] = "axis_angle",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the position and orientation error between source and target frames.

    Args:
        t01: Position of source frame. Shape is (N, 3).
        q01: Quaternion orientation of source frame in (w, x, y, z). Shape is (N, 4).
        t02: Position of target frame. Shape is (N, 3).
        q02: Quaternion orientation of target frame in (w, x, y, z). Shape is (N, 4).
        rot_error_type: The rotation error type to return: "quat", "axis_angle".
            Defaults to "axis_angle".

    Returns:
        A tuple containing position and orientation error.
        - Position error: (N, 3).
        - Orientation error:
            - If rot_error_type is "quat": quaternion error (N, 4).
            - If rot_error_type is "axis_angle": axis-angle error (N, 3).

    Raises:
        ValueError: If an unsupported rotation error type is provided.
    """
    # Compute quaternion error (i.e., difference quaternion)
    # Calculate the norm of q01: q01 * q01_conjugate
    source_quat_norm = quat_mul(q01, quat_conjugate(q01))[0]
    # Inverse of q01: conjugate(q01) normalized by its norm.
    source_quat_inv = quat_conjugate(q01) / source_quat_norm[..., None]
    # Difference quaternion: q_error = q02 * q01_inv
    quat_error = quat_mul(q02, source_quat_inv)
    
    # Compute position error.
    pos_error = t02 - t01
    
    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        axis_angle_error = axis_angle_from_quat(quat_error)
        return pos_error, axis_angle_error
    else:
        raise ValueError(
            f"Unsupported orientation error type: {rot_error_type}. Valid: 'quat', 'axis_angle'."
        )
    
def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """
    Wraps input angles (in radians) to the range [-π, π].

    Args:
        angles: Input angles of any shape.

    Returns:
        Angles wrapped to the range [-π, π].
    """
    wrapped_angle = (angles + np.pi) % (2 * np.pi)
    return np.where((wrapped_angle == 0) & (angles > 0), np.pi, wrapped_angle - np.pi)

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    # a = v * (2*q_w^2 - 1)
    factor_a = 2.0 * (q_w ** 2) - 1.0
    a = v * np.expand_dims(factor_a, axis=-1)
    
    # b = 2 * q_w * (q_vec x v)
    b = 2.0 * np.expand_dims(q_w, axis=-1) * np.cross(q_vec, v, axis=-1)
    
    # c = 2 * q_vec * dot(q_vec, v)
    if q_vec.ndim == 2:
        dot_result = np.matmul(q_vec[:, None, :], v[:, :, None]).squeeze(-1)  # shape: (N, 1)
        c = 2.0 * q_vec * dot_result  # (N, 3) * (N, 1)
    else:
        dot_result = np.einsum("...i,...i->...", q_vec, v)
        c = 2.0 * q_vec * np.expand_dims(dot_result, axis=-1)
    
    return a + b + c


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    
    See https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-6:  # If vectors are nearly parallel, return identity matrix
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def quat_from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    angle-axis 표현을 quaternion (w, x, y, z)로 변환합니다.
    
    Args:
        angle: 회전 각도 (라디안). Shape은 (N, ) 이어야 합니다.
        axis: 회전축. Shape은 (N, 3) 이어야 합니다.
        
    Returns:
        단위 quaternion (w, x, y, z). Shape은 (N, 4) 입니다.
    """
    # 회전각의 절반 값을 (N, 1) 형태로 만듭니다.
    theta = (angle / 2)[..., None]
    
    # 회전축을 정규화한 후, sin(theta)를 곱합니다.
    xyz = normalize(axis) * np.sin(theta)
    
    # w 성분은 cos(theta) 입니다.
    w = np.cos(theta)
    
    # w와 xyz를 이어붙여 quaternion (w, x, y, z)를 만듭니다.
    quat = np.concatenate([w, xyz], axis=-1)
    
    # 안전하게 unit quaternion이 되도록 한 번 더 정규화합니다.
    return normalize(quat)