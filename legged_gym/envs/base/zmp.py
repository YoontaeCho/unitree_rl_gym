'''
ZMP-related functions
'''
import torch
from legged_gym.envs.base.zmp import *
from legged_gym.utils.math import *
from typing import Tuple

def compute_com(
        rb_com: torch.Tensor,
        rb_mass: torch.Tensor,
        total_mass: torch.Tensor,
        rigid_body_state: torch.Tensor,
    ) -> torch.Tensor:

    rb_com_global = rigid_body_state[:,:,:3] + \
        quat_apply(rigid_body_state[:,:,3:7], rb_com)
    return (rb_com_global * rb_mass.unsqueeze(-1)).sum(dim=1) / total_mass

def compute_lin_ang_momentum(
        rb_com: torch.Tensor,
        rb_inertia: torch.Tensor,
        rb_mass: torch.Tensor,
        rigid_body_state: torch.Tensor,
        root_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the linear and angular momentum of the articulated system.
    
    Uses the self.rb_com, self.rb_inertia, and self.rb_mass tensors that have been
    previously set up for each environment and rigid body.

    Args:
    - self.rb_com: A tensor of shape [num_envs, num_bodies, 3], representing the center of mass of each body.
    - self.rb_inertia: A tensor of shape [num_envs, num_bodies, 3, 3], representing the inertia tensor for each body.
    - self.rb_mass: A tensor of shape [num_envs, num_bodies], representing the masses of the bodies.
    - self.rigid_body_state: A tensor of shape [num_envs, num_bodies, 13], where:
        - [:, :, 0:3] represents the position of the body.
        - [:, :, 3:7] represents the quaternion of the body (qx, qy, qz, qw).
        - [:, :, 7:10] represents the linear velocity of the body.
        - [:, :, 10:13] represents the angular velocity of the body.

    Returns:
    - linear_momentum: A tensor of shape [num_envs, 3], representing the total linear momentum for each environment.
    - angular_momentum: A tensor of shape [num_envs, 3], representing the total angular momentum for each environment.
    """

    # Extract positions, linear velocities, angular velocities, and quaternions from self.rigid_body_state
    positions = rigid_body_state[:, :, 0:3]    # Shape: [num_envs, num_bodies, 3]
    quaternions = rigid_body_state[:, :, 3:7]  # Shape: [num_envs, num_bodies, 4]
    linear_velocities = rigid_body_state[:, :, 7:10]  # Shape: [num_envs, num_bodies, 3]
    angular_velocities = rigid_body_state[:, :, 10:13]  # Shape: [num_envs, num_bodies, 3]

    # Convert local com positions (self.rb_com) to global frame using quaternion rotation
    rb_com_global = positions + quat_apply(quaternions,
                                            rb_com)
    rb_com_delta = quat_apply(quaternions, rb_com)

    

    # Compute linear momentum for each body
    # Linear momentum P = m * v
    v_com = linear_velocities + torch.cross(angular_velocities, rb_com_delta, dim=-1)  # Shape: [num_envs, num_bodies, 3]

    linear_momentum_per_body = rb_mass.unsqueeze(-1) * v_com  # Shape: [num_envs, num_bodies, 3]
    
    # Sum the linear momenta of all bodies in each environment
    lin_momentum = torch.sum(linear_momentum_per_body, dim=1)  # Shape: [num_envs, 3]

    # Compute angular momentum for each body
    # Orbital angular momentum L_orb = (position - global com) x (m * v)
    rb_com_local = rb_com_global 
    rb_com_local[..., :2] -= root_states[..., :2].unsqueeze(-2)
    r_cross_m_v = torch.cross(rb_com_local, linear_momentum_per_body, dim=-1)  # Shape: [num_envs, num_bodies, 3]



    rot = matrix_from_quaternion(quaternions)
    inertia_global = torch.einsum('...ik,...kl,...jl->...ij', rot, rb_inertia, rot)  # Shape: [num_envs, num_bodies, 3, 3]

    # Rotational angular momentum L_rot = I * omega
    I_times_omega = torch.einsum('...ij,...j->...i', inertia_global, angular_velocities)  # Shape: [num_envs, num_bodies, 3]

    # Total angular momentum is the sum of orbital and rotational angular momentum
    angular_momentum_per_body = r_cross_m_v + I_times_omega  # Shape: [num_envs, num_bodies, 3]

    # Sum the angular momenta of all bodies in each environment
    ang_momentum = torch.sum(angular_momentum_per_body, dim=1)  # Shape: [num_envs, 3]

    # Compute objects related stuff
    # obj_lin_momentum = self.obj_mass.unsqueeze(-1) * self.obj_root_states[..., 7:10]
    # obj_rot = matrix_from_quaternion(self.obj_root_states[..., 3:7])
    # obj_inertia_global = torch.einsum('...ik,...kl,...jl->...ij', obj_rot, self.obj_inertia, obj_rot)
    # obj_ang_momentum = \
    #     torch.cross(self.obj_root_states[..., 0:3], obj_lin_momentum, dim=-1) \
    #     + torch.einsum('...ij,...j->...i', obj_inertia_global, self.obj_root_states[..., 10:13])
    # self.lin_momentum += obj_lin_momentum
    # self.ang_momentum += obj_ang_momentum

    return lin_momentum, ang_momentum



def compute_zmp(
        total_mass, gravity,
        dt: torch.Tensor,
        lin_momentum: torch.Tensor,
        prev_lin_momentum: torch.Tensor,
        ang_momentum: torch.Tensor,
        prev_ang_momentum: torch.Tensor,
        foot_contacts: torch.Tensor,
        com: torch.Tensor,
        root_states: torch.Tensor
    ) -> torch.Tensor:
    '''
    Computes the Zero-moment-point of the humanoid
    NOTE(ytcho): Before calling this function, the (lin, ang) momentum,
    and the COM must be refreshed.
    '''
    # Gv = 9.80665
    # Gv = -self.cfg.sim.gravity[2]
    # # Mg = self._mass * Gv
    # Mg = self.total_mass* Gv
    Mg = -total_mass * gravity[2]


    # d_lin_mom = (self.lin_momentum - self.prev_lin_momentum)/self.dt
    d_lin_mom = (lin_momentum - prev_lin_momentum)/(0.25*dt)
    # ic(self.lin_momentum)
    # ic(d_lin_mom)
    d_ang_mom = (ang_momentum - prev_ang_momentum)/(0.25*dt)

    Fgz = d_lin_mom[..., 2] + Mg

    # check contact with floor
    # contacts = [self._sim.data.contact[i] for i in range(self._sim.data.ncon)]
    # contact_flag = [(c.geom1==0 or c.geom2==0) for c in contacts]
    contact_flag = torch.any(foot_contacts, dim=-1)

    zmp = torch.zeros_like(com)
    zmp[..., 0] = torch.where(
        torch.logical_and(contact_flag, Fgz > 10),
        (Mg*(com[..., 0]-root_states[..., 0]) - d_ang_mom[..., 1])/Fgz,
        com[..., 0]-root_states[..., 0]
    )
    zmp[..., 1]= torch.where(
        torch.logical_and(contact_flag, Fgz > 10),
        ((Mg*(com[..., 1]-root_states[..., 1]) + d_ang_mom[..., 0])/Fgz),
        com[..., 1]-root_states[..., 1]
    )
    zmp[..., 0] += root_states[..., 0]
    zmp[..., 1] += root_states[..., 1]
    
    # avg_foot_pos = 0.5 * (self.left_feet_pos + self.right_feet_pos)
    # zmp_avgfoot_dist = torch.norm((self.zmp[..., :2]-avg_foot_pos[..., :2]), dim=-1)
    # For logging
    # self.extras['log_zmp'] = zmp_avgfoot_dist
    # com_avgfoot_dist = torch.norm((self.com[..., :2]-avg_foot_pos[..., :2]), dim=-1)
    # self.extras['log_com'] = com_avgfoot_dist


    # self.avg_foot_pos_base = quat_rotate_inverse(
    #     self.base_quat,
    #     avg_foot_pos - self.root_states[..., :3]
    # ) 
    # self.ZMP_base = quat_rotate_inverse(
    #     self.base_quat,
    #     self.ZMP - self.root_states[..., :3]
    # ).clip(min=-3, max=3)
    # ic(self.avg_foot_pos_base, self.ZMP_base)

    # ic(self.ZMP - self.com[..., :2])
    return zmp
