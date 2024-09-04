import torch
from torch import Tensor
from typing import Optional
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def quat_rotate(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    q_ax = q[..., 0:3]
    t = 2.0 * torch.cross(q_ax, x, dim=-1)
    return x + q[..., 3:4] * t + torch.cross(q_ax, t, dim=-1)

def quat_multiply(q1: torch.Tensor, q2: torch.Tensor,
                  out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
    x2, y2, z2, w2 = torch.unbind(q2, dim=-1)
    if out is None:
        out = torch.empty_like(q1)
    out[...] = torch.stack([
        x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
        -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
        x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2,
        -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2], dim=-1)
    return out

def quat_inverse(q: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = q.clone()
    out.copy_(q)
    out[..., 3] = -out[..., 3]
    return out

def quat_from_euler(euler):
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

def axis_angle_from_quat(quat, eps=1.0e-6):
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference:
    # https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    if True:
        axis = torch.nn.functional.normalize(quat[..., 0:3])

        half_angle = torch.acos(quat[..., 3:].clamp(-1.0, +1.0))
        angle = (2.0 * half_angle + torch.pi) % (2 * torch.pi) - torch.pi
        return axis * angle
    
def wrap_to_pi_minuspi(angles):
    angles = angles % (2 * np.pi)
    angles -= 2 * np.pi * (angles > np.pi)
    return angles