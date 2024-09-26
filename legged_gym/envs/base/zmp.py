'''
ZMP-related functions
'''
import isaacgym
import torch
from legged_gym.utils.math import quat_apply, matrix_from_quaternion
from typing import Tuple
from icecream import ic

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



def compute_min_distance(
        footpoints: torch.Tensor, 
        footpoints_mask: torch.Tensor, 
        zmp: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum distance between the anchor points and the convex hull of the polygons.

    Args:
    polygon_points: torch.Tensor [N, M, 2] -> Polygon points for each sample (N samples, each with M vertices).
    point_mask: torch.Tensor [N, M] -> Binary mask indicating which points are valid for each sample.
    anchor_points: torch.Tensor [N, 2] -> Anchor points for each sample.

    Returns:
    torch.Tensor [N] -> Minimum distance between each anchor point and the corresponding convex hull.
    """
    
    # Step 1: Compute vectors from anchor points to polygon vertices, considering only valid points
    vectors_to_vertices = footpoints - zmp[:, None, :]  # [N, M, 2]

    # Compute the angles of the vectors with respect to the positive x-axis using atan2
    angles = torch.atan2(vectors_to_vertices[..., 1], 
                         vectors_to_vertices[..., 0])  # [N, M]

    # Step 2: Only include valid points using the point mask
    valid_angles = torch.where(
        footpoints_mask, 
        angles, 
        torch.tensor(float('nan'), device=angles.device))  # Mask invalid points as NaN

    # Sort valid angles (NaNs will be pushed to the end)
    valid_angles, _ = torch.sort(valid_angles, dim=-1)  # [N, M]

    # ic(valid_angles)
    # Replace NaNs (invalid points) with the smallest valid angle of the row
    first_valid_angle = valid_angles[:, 0:1]  # [N, 1] (smallest valid angle per row)
    sorted_valid_angles = torch.where(
        torch.isnan(valid_angles), first_valid_angle, valid_angles)  # Replace NaNs
    
    # Compute the differences between consecutive sorted valid angles
    delta_angles = torch.diff(sorted_valid_angles, dim=-1)  # [N, M-1]
    wraparound_delta = \
        sorted_valid_angles[:, 0] - sorted_valid_angles[:, -1]  # [N]

    # Combine delta angles and wrap-around delta
    delta_angles = torch.cat([delta_angles, wraparound_delta.unsqueeze(-1)], dim=-1)  # [N, M]

    # Map from 0 to 2*pi
    delta_angles = torch.where(
        delta_angles >= 0,
        delta_angles,
        delta_angles + 2 * torch.pi)

    # Step 3: Check if any delta_angle exceeds pi, meaning the point is outside the polygon
    is_outside = torch.any(delta_angles > torch.pi, dim=-1) | torch.isnan(valid_angles[..., 1]) # [N]

    # Compute the distances to all edges (not just consecutive edges)
    N, _, _ = footpoints.shape
    
    # Pair up all points to form all possible edges (including non-consecutive)
    point_A = footpoints[:, :, None, :]  # [N, M, 1, 2]
    point_B = footpoints[:, None, :, :]  # [N, 1, M, 2]

    # Create a mask to handle valid pairs (ensure both points in the pair are valid)
    pair_mask = footpoints_mask[:, :, None] & footpoints_mask[:, None, :]  # [N, M, M]

    # Compute edge vectors (B - A) for all edges formed by pairs
    edge_vectors = point_B - point_A  # [N, M, M, 2]

    # Compute vectors from anchor points to the polygon points A
    vectors_to_zmp = zmp[:, None, None, :] - point_A  # [N, M, M, 2]

    # Project the vectors to anchor points onto the edge vectors
    edge_lengths_sq = torch.sum(edge_vectors ** 2, dim=-1, keepdim=True)  # [N, M, M, 1]
    t = torch.sum(vectors_to_zmp * edge_vectors, dim=-1, keepdim=True) \
        / (edge_lengths_sq + 1e-8)  # [N, M, M, 1]
    t_clipped = torch.clamp(t, 0, 1)  # Restrict to [0, 1] for line segment

    # Compute the projection points on the edges
    projection_points = point_A + t_clipped * edge_vectors  # Projected points on edges [N, M, M, 2]

    # Compute the distance from the anchor points to the projection points
    distances_to_edges = torch.norm(zmp[:, None, None, :] - projection_points, dim=-1)  # [N, M, M]

    # Apply the mask to ignore invalid edges (non-valid point pairs)
    distances_to_edges_masked = torch.where(
        pair_mask, 
        distances_to_edges, 
        torch.full_like(distances_to_edges, float('inf')))  # [N, M, M]

    # Step 6: Compute the final minimum distance
    min_distances_to_polygon = torch.min(distances_to_edges_masked.view(N, -1), dim=-1).values  # [N]

    # ic(distances_to_edges_masked)

    # If the point is inside the polygon, set the distance to 0
    min_distances = torch.where(
        is_outside, 
        min_distances_to_polygon, 
        torch.zeros_like(min_distances_to_polygon))  # [N]

    # Finally, filter out inf(no valid point case)
    min_distances = torch.where(
        torch.isinf(min_distances), 
        torch.zeros_like(min_distances), 
        min_distances)

    return min_distances

def main():
    # Define a larger set of polygon points
    polygon_points = torch.tensor([
        # First polygon: A triangle with two extra points
        [[0, 0], [2, 0], [1, 1], [0.5, 0.5], [1.5, 0.5]],
        # Second polygon: A square
        [[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]],
        # Third polygon: A pentagon (with all valid points)
        [[1, 0], [2, 1], [1.5, 2], [0.5, 2], [0, 1]],
        # Fourth polygon: A line (degenerate polygon)
        [[1, 1], [3, 1], [0, 0], [0, 0], [0, 0]],
        # Fifth polygon: A complex shape with some internal points (for anchor point 5)
        [[0, 0], [3, 0], [3, 3], [0, 3], [1.5, 1.5]],
        # Fifth polygon duplicate: A complex shape with some internal points (for anchor point 6)
        [[0, 0], [3, 0], [3, 3], [0, 3], [1.5, 1.5]]
    ], dtype=torch.float32)

    # Define the point mask, indicating the number of valid points for each polygon
    point_mask = torch.tensor([
        [1, 1, 1, 0, 0],  # Only first three points are valid (triangle)
        [1, 1, 1, 1, 0],  # First four points form a square
        [1, 1, 1, 1, 1],  # All points are valid (pentagon)
        [0, 0, 0, 0, 0],  # Only two points are valid (line)
        [1, 1, 1, 1, 1],  # All points valid (complex shape for anchor point 5)
        [1, 1, 1, 1, 1]   # All points valid (complex shape for anchor point 6)
    ], dtype=torch.bool)

    # Define anchor points for testing
    anchor_points = torch.tensor([
        [1, 0.5],  # Inside the first triangle
        [3, 1],    # Outside the second square
        [1.5, 1.5],# Inside the pentagon
        [2, 1],    # On the line (fourth polygon)
        [4, 4],    # Far outside the complex shape (fifth polygon for anchor point 5)
        [1.9, 1.9] # Inside the complex shape (fifth polygon for anchor point 6)
    ], dtype=torch.float32)

    # Run the test
    min_distances = compute_min_distance(polygon_points, point_mask, anchor_points)
    print("Min Distances:", min_distances)


if __name__ == '__main__':
    main()