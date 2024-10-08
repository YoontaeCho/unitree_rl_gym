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
        measured_frame: torch.Tensor
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
    # rb_com_local = rb_com_global 
    # rb_com_local[..., :2] -= measured_frame[..., :2].unsqueeze(-2)
    rb_com_local = rb_com_global - measured_frame.unsqueeze(-2)
    r_cross_m_v = torch.cross(rb_com_local, linear_momentum_per_body, dim=-1)  # Shape: [num_envs, num_bodies, 3]



    rot = matrix_from_quaternion(quaternions)
    inertia_global = torch.einsum('...ik,...kl,...jl->...ij', rot, rb_inertia, rot)  # Shape: [num_envs, num_bodies, 3, 3]

    # Rotational angular momentum L_rot = I * omega
    I_times_omega = torch.einsum('...ij,...j->...i', inertia_global, angular_velocities)  # Shape: [num_envs, num_bodies, 3]

    # Total angular momentum is the sum of orbital and rotational angular momentum
    angular_momentum_per_body = r_cross_m_v + I_times_omega  # Shape: [num_envs, num_bodies, 3]
    # ic(r_cross_m_v[..., :10, :])

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

def compute_zmp_v2(contact_force, contact_points, com):
    # Extract components of contact forces and positions
    fx = contact_force[:, :, 0]
    fy = contact_force[:, :, 1]
    fz = contact_force[:, :, 2]

    px = contact_points[:, :, 0]
    py = contact_points[:, :, 1]
    pz = contact_points[:, :, 2]

    # Avoid division by zero by adding a small epsilon to the sum of vertical forces
    epsilon = 1e-8
    fz_sum = torch.sum(fz, dim=1, keepdim=True) + epsilon

    # Check if all contact forces are zero for each environment
    all_zero_forces = torch.all(fz_sum < 1, dim=1)

    # Compute ZMP coordinates considering horizontal forces and vertical positions
    alpha = fz / fz_sum
    zmp_x = torch.sum(px * alpha, dim=1)
    zmp_y = torch.sum(py * alpha, dim=1)

    # Stack zmp_x and zmp_y to form the final ZMP tensor
    zmp = torch.zeros_like(com)
    zmp[..., :2] = torch.stack((zmp_x, zmp_y), dim=1)

    # Replace ZMP with center of mass if all contact forces are zero
    zmp[all_zero_forces, :2] = com[all_zero_forces, :2]

    return zmp

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
    # d_lin_mom = (lin_momentum - prev_lin_momentum)/(0.25*dt)
    d_lin_mom = (lin_momentum - prev_lin_momentum)/(dt)
    # ic(lin_momentum, prev_lin_momentum)
    # ic(ang_momentum, prev_ang_momentum)
    # ic(self.lin_momentum)
    # ic(d_lin_mom)
    # d_ang_mom = (ang_momentum - prev_ang_momentum)/(0.25*dt)
    d_ang_mom = (ang_momentum - prev_ang_momentum)/(dt)

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



def compute_convex_hull(polygon_points, point_mask):
    N, M, _ = polygon_points.shape

    # Initialize convex hull indices with -1
    convex_hull_indices = torch.full((N, M), -1, dtype=torch.long, device=polygon_points.device)

    # Mask invalid points by setting their x-coordinate to infinity
    x_coords = polygon_points[..., 0].clone()  # Shape: [N, M]
    x_coords[~point_mask] = float('inf')

    # Find the leftmost point for each sample
    _, leftmost_indices = x_coords.min(dim=1)  # Shape: [N]
    current_point_indices = leftmost_indices.clone()  # Shape: [N]

    # Initialize previous edge vector as [0, 1] for all samples
    prev_edge_vectors = torch.zeros(N, 2, device=polygon_points.device)
    prev_edge_vectors[:, 1] = -1.0  # Shape: [N, 2]

    # Compute number of valid points per sample
    valid_point_counts = point_mask.sum(dim=1)  # Shape: [N]

    # Initialize completed mask
    completed = torch.zeros(N, dtype=torch.bool, device=polygon_points.device)

    # Handle samples with 0 valid points
    no_point_mask = valid_point_counts == 0
    if no_point_mask.any():
        completed[no_point_mask] = True

    # Handle samples with 1 valid point
    one_point_mask = valid_point_counts == 1
    if one_point_mask.any():
        sample_indices = torch.where(one_point_mask)[0]
        valid_points = point_mask[one_point_mask]  # Shape: [N_one_point, M]
        valid_point_indices = valid_points.nonzero(as_tuple=False)  # Shape: [N_one_point, 2]
        sample_indices_expanded = sample_indices[valid_point_indices[:, 0]]  # Map back to original sample indices
        point_indices = valid_point_indices[:, 1]
        convex_hull_indices[sample_indices_expanded, 0] = point_indices
        completed[sample_indices_expanded] = True

    # Keep track of starting point to detect when we complete the convex hull
    starting_point_indices = leftmost_indices.clone()

    # For up to M steps
    for i in range(M):
        # For samples that are not yet completed
        active = ~completed  # Shape: [N]
        if not active.any():
            break  # All samples completed

        # Get indices of active samples
        active_indices = torch.where(active)[0]  # Shape: [N_active]

        curr_point_indices_active = current_point_indices[active_indices]  # Shape: [N_active]
        prev_edge_active = prev_edge_vectors[active_indices]  # Shape: [N_active, 2]

        # Current points for active samples
        curr_points = polygon_points[active_indices, curr_point_indices_active]  # Shape: [N_active, 2]

        # Vectors from current point to all candidates
        vectors_to_candidates = polygon_points[active_indices] - curr_points.unsqueeze(1)  # Shape: [N_active, M, 2]

        # Mask invalid candidates and exclude current point
        valid_mask = point_mask[active_indices].clone()  # Shape: [N_active, M]
        valid_mask[torch.arange(valid_mask.size(0), device=polygon_points.device), curr_point_indices_active] = False

        vectors_norm = vectors_to_candidates.norm(dim=2, keepdim=True)
        vectors_norm[vectors_norm == 0] = 1e-6  # Avoid division by zero
        vectors_normalized = vectors_to_candidates / vectors_norm

        # Compute cross products between prev_edge and vectors_to_candidates
        def dot_product_2d(a, b):
            return a[..., 0]*b[..., 0] + a[..., 1]*b[..., 1]

        # ic(curr_points, prev_edge_active, vectors_normalized)
        dot_product = dot_product_2d(prev_edge_active.unsqueeze(1), vectors_normalized)  # Shape: [N_active, M]
        # ic(dot_product)
        # dot_product = torch.where(dot_product > 0, torch.ones_like(dot_product), dot_product)
        # ic(dot_product)

        # Set cross_products to -inf for invalid candidates
        dot_product[~valid_mask] = float('inf')
        # ic(dot_product)

        # Find next point indices
        max_cross_products, next_point_indices = dot_product.min(dim=1)  # Shape: [N_active]
        # ic(next_point_indices)

        # Detect samples with no valid candidates
        no_valid_candidates = max_cross_products == float('inf')  # Shape: [N_active]

        # Update completed mask for samples with no valid candidates
        completed[active_indices[no_valid_candidates]] = True

        # Samples with valid next points
        valid_next = ~no_valid_candidates  # Shape: [N_active]
        if valid_next.any():
            valid_active_indices = active_indices[valid_next]  # Shape: [N_valid_next]

            # Update convex hull indices for samples with valid next points
            convex_hull_indices[valid_active_indices, i] = current_point_indices[valid_active_indices]

            # Update current_point_indices and prev_edge_vectors for samples with valid next points
            new_curr_point_indices = next_point_indices[valid_next]  # Shape: [N_valid_next]
            new_curr_points = curr_points[valid_next]  # Shape: [N_valid_next, 2]
            next_points = polygon_points[valid_active_indices, new_curr_point_indices]  # Shape: [N_valid_next, 2]

            current_point_indices[valid_active_indices] = new_curr_point_indices
            # prev_edge_vectors[valid_active_indices] = next_points - new_curr_points
            prev_edge_vectors[valid_active_indices] = new_curr_points - next_points

            # Check if we have returned to the starting point
            has_returned_to_start = current_point_indices[valid_active_indices] == starting_point_indices[valid_active_indices]
            completed[valid_active_indices[has_returned_to_start]] = True

        else:
            # If no samples have valid next points, we can exit early
            if not (~completed).any():
                break

        # For samples that are completed, set current_point_indices to -1 to prevent further updates
        current_point_indices[completed] = -1
        prev_edge_vectors[completed] = 0

    # Create a mask for convex hull points
    is_hull = convex_hull_indices != -1  # Shape: [N, M]
    convex_hull_lengths = is_hull.sum(dim=1)  # Shape: [N]

    # Prepare positions_in_hull for sorting
    positions_in_hull = torch.full((N, M), fill_value=M + M, device=polygon_points.device)

    # Assign positions based on the order in which points were added to the hull
    indices_i = torch.arange(M, device=polygon_points.device).unsqueeze(0).expand(N, M)  # Shape: [N, M]
    valid_hull_mask = convex_hull_indices != -1  # Shape: [N, M]

    convex_hull_indices_valid = convex_hull_indices[valid_hull_mask]  # Shape: [K_total]
    positions_i_valid = indices_i[valid_hull_mask]  # Shape: [K_total]
    batch_indices = torch.arange(N, device=polygon_points.device).unsqueeze(1).expand(N, M)[valid_hull_mask]  # Shape: [K_total]

    positions_in_hull[batch_indices, convex_hull_indices_valid] = positions_i_valid

    # Generate permutation_indices by sorting positions_in_hull
    permutation_indices = positions_in_hull.argsort(dim=1)

    # Rearrange polygon_points
    polygon_points_reordered = polygon_points[torch.arange(N)[:, None], permutation_indices]

    # Create polygon_idx with indices starting from 0 for convex hull points
    positions = torch.arange(M, device=polygon_points.device).unsqueeze(0).expand(N, M)  # Shape: [N, M]
    mask = positions < convex_hull_lengths.unsqueeze(1)
    polygon_idx = torch.full((N, M), -1, dtype=torch.long, device=polygon_points.device)
    polygon_idx[mask] = positions[mask]

    return polygon_points_reordered, polygon_idx


def compute_signed_distance(polygon_points_reordered, polygon_idx, points):
    """
    Args:
        polygon_points_reordered: torch.Tensor [N, M, 2] - Reordered polygon points with convex hull points first.
        polygon_idx: torch.Tensor [N, M] - Indices of convex hull points, -1 for invalid points.
        points: torch.Tensor [N, 2] - Query points.
    Returns:
        signed_distance: torch.Tensor [N] - Signed distance from each point to its convex hull.
    """
    N, M, _ = polygon_points_reordered.shape
    device = polygon_points_reordered.device

    # First, get the number of convex hull points per sample
    is_hull = polygon_idx != -1  # Shape: [N, M]
    convex_hull_lengths = is_hull.sum(dim=1)  # Shape: [N], number of convex hull points per sample
    max_K = max(1, convex_hull_lengths.max().item())  # Maximum number of convex hull points

    # Prepare hull points padded to [N, max_K, 2]
    hull_points = polygon_points_reordered[:, :max_K, :]  # [N, max_K, 2]
    K = convex_hull_lengths  # [N]

    # Create indices for the polygon vertices
    indices = torch.arange(max_K, device=device).unsqueeze(0).expand(N, max_K)  # [N, max_K]

    # For each sample n, mask valid indices
    valid = indices < K.unsqueeze(1)  # [N, max_K], boolean mask

    # For end indices, wrap around using modulo K[n]
    K_expanded = K.unsqueeze(1).expand(N, max_K)  # [N, max_K]
    end_indices = (indices + 1) % torch.clamp(K_expanded, min=1)  # Avoid modulo zero
    # For indices where indices >= K[n], set end_indices to indices (to avoid invalid indices)
    end_indices[~valid] = indices[~valid]

    # Get start_points and end_points
    batch_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, max_K)  # [N, max_K]

    start_points = hull_points[batch_indices, indices]  # [N, max_K, 2]
    end_points = hull_points[batch_indices, end_indices]  # [N, max_K, 2]

    # For invalid indices, set start_points and end_points to zero
    start_points[~valid] = 0
    end_points[~valid] = 0

    # Compute edge_vectors
    edge_vectors = end_points - start_points  # [N, max_K, 2]

    # Compute vectors from start_points to query points
    point_vectors = points.unsqueeze(1) - start_points  # [N, max_K, 2]

    # Compute projection lengths along the edges
    edge_lengths_sq = (edge_vectors ** 2).sum(dim=2, keepdim=True)  # [N, max_K, 1]
    edge_lengths_sq[edge_lengths_sq == 0] = 1e-8  # Avoid division by zero

    # Compute the projection scalar 't' for each point onto each edge
    t = (point_vectors * edge_vectors).sum(dim=2, keepdim=True) / edge_lengths_sq  # [N, max_K, 1]

    # Clamp 't' to the range [0, 1] to stay within the segment
    t_clamped = t.clamp(0, 1)

    # Compute the closest points on the segments
    closest_points = start_points + t_clamped * edge_vectors  # [N, max_K, 2]

    # Compute distances to the closest points
    distances = (points.unsqueeze(1) - closest_points).norm(dim=2)  # [N, max_K]

    # For invalid edges, set distances to a large value
    distances[~valid] = float('inf')

    # Minimum distance to the polygon edges
    min_distances, _ = distances.min(dim=1)  # [N]

    # Initialize inside_mask
    inside_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Handle samples with K >= 3
    mask_K_ge_3 = K >= 3
    if mask_K_ge_3.any():
        indices_K_ge_3 = torch.where(mask_K_ge_3)[0]
        points_K_ge_3 = points[indices_K_ge_3]  # [N1, 2]
        hull_points_K_ge_3 = hull_points[indices_K_ge_3]  # [N1, max_K, 2]
        lengths_K_ge_3 = K[indices_K_ge_3]  # [N1]

        inside_mask_K_ge_3 = is_point_in_polygon(points_K_ge_3, hull_points_K_ge_3, lengths_K_ge_3)
        inside_mask[indices_K_ge_3] = inside_mask_K_ge_3

    # Handle samples with K == 1
    mask_K_eq_1 = K == 1
    if mask_K_eq_1.any():
        indices_K_eq_1 = torch.where(mask_K_eq_1)[0]
        points_K_eq_1 = points[indices_K_eq_1]  # [N2, 2]
        hull_points_K_eq_1 = hull_points[indices_K_eq_1, 0, :]  # [N2, 2]
        # If distance is zero, point coincides with hull point
        distances_K_eq_1 = min_distances[indices_K_eq_1]  # [N2]
        inside_mask_K_eq_1 = distances_K_eq_1 < 1e-6
        inside_mask[indices_K_eq_1] = inside_mask_K_eq_1

    # Handle samples with K == 2
    mask_K_eq_2 = K == 2
    if mask_K_eq_2.any():
        indices_K_eq_2 = torch.where(mask_K_eq_2)[0]
        points_K_eq_2 = points[indices_K_eq_2]  # [N3, 2]
        # For these samples, we need to check if point lies on the line segment
        # For each sample n in indices_K_eq_2, t_clamped[n, :2] are valid
        t_eq_2 = t_clamped[indices_K_eq_2, :2, 0]  # [N3, 2]
        distances_eq_2 = distances[indices_K_eq_2, :2]  # [N3, 2]
        # Check if any edge has distance < epsilon and t between 0 and 1
        on_segment = ((t_eq_2 >= 0) & (t_eq_2 <= 1) & (distances_eq_2 < 1e-6)).any(dim=1)
        inside_mask[indices_K_eq_2] = on_segment

    # Assign negative sign to distances for points inside the convex hull
    signed_distances = min_distances
    signed_distances[inside_mask] *= -1

    # Handle samples with no valid edges (K[n] == 0)
    no_edges = K == 0
    # signed_distances[no_edges] = float('nan')
    signed_distances[no_edges] = 0.

    return signed_distances

def is_point_in_polygon(points, polygon, lengths):
    """
    Vectorized point-in-polygon test using the winding number method.
    Args:
        points: torch.Tensor [N1, 2] - Query points.
        polygon: torch.Tensor [N1, max_K, 2] - Polygon vertices.
        lengths: torch.Tensor [N1] - Number of valid vertices per sample.
    Returns:
        inside: torch.Tensor [N1] - Boolean mask indicating whether each point is inside its polygon.
    """
    # Step 1: Compute vectors from anchor points to polygon vertices, considering only valid points
    vectors_to_vertices = polygon - points[:, None, :]  # [N, M, 2]

    # Compute the angles of the vectors with respect to the positive x-axis using atan2
    angles = torch.atan2(vectors_to_vertices[..., 1], 
                         vectors_to_vertices[..., 0])  # [N, M]

    # Step 2: Only include valid points using the point mask
    N1, max_K, _ = polygon.shape
    indices = torch.arange(max_K, device=polygon.device).unsqueeze(0).expand(N1, max_K)  # [N1, max_K]
    valid_mask = indices < lengths.unsqueeze(1)  # [N1, max_K]
    valid_angles = torch.where(
        valid_mask,
        angles,
        torch.tensor(float('nan'), device=angles.device)
    )
    # ic(angles, valid_mask)

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
    return ~is_outside














    # #####################
    # N1, max_K, _ = polygon.shape

    # # Create indices for valid vertices
    # indices = torch.arange(max_K, device=polygon.device).unsqueeze(0).expand(N1, max_K)  # [N1, max_K]
    # valid_mask = indices < lengths.unsqueeze(1)  # [N1, max_K]

    # # Shifted polygons for edge calculation
    # shifted_polygon = torch.roll(polygon, shifts=-1, dims=1)  # [N1, max_K, 2]

    # # Edge vectors
    # edges = shifted_polygon - polygon  # [N1, max_K, 2]

    # # Vectors from polygon vertices to points
    # vectors = points.unsqueeze(1) - polygon  # [N1, max_K, 2]

    # # Compute cross and dot products
    # cross_products = edges[..., 0] * vectors[..., 1] - edges[..., 1] * vectors[..., 0]  # [N1, max_K]
    # dot_products = edges[..., 0] * vectors[..., 0] + edges[..., 1] * vectors[..., 1]  # [N1, max_K]

    # # Compute angles
    # angles = torch.atan2(cross_products, dot_products + 1e-8)  # [N1, max_K]

    # # Zero out angles for invalid vertices
    # angles[~valid_mask] = 0
    # ic(angles)

    # # Sum angles
    # angle_sums = angles.sum(dim=1)  # [N1]


    # # Points are inside if the absolute value of angle_sums is greater than pi
    # inside = angle_sums.abs() > 1e-3  # [N1]

    # return inside


import matplotlib.pyplot as plt
import numpy as np



def visualize_convex_hulls(polygon_points, point_mask, reordered_footpoints, convex_hull_idx, points=None):
    N = polygon_points.shape[0]  # Number of samples (batches)
    
    # Set up the plot
    fig, axs = plt.subplots(1, N, figsize=(5 * N, 5))
    
    if N == 1:
        axs = [axs]  # Handle single subplot case as a list for consistency
    
    for i in range(N):
        ax = axs[i]
        
        # Plot all the valid points in the sample (the original valid points)
        valid_points = polygon_points[i][point_mask[i].bool()]
        ax.scatter(valid_points[:, 0], valid_points[:, 1], color='blue', label='Points')
        if points is not None:
            ax.scatter(points[i, 0], points[i, 1], color='green', label='ZMP')
        
        # Get the convex hull indices from convex_hull_idx[i], ignore the -1 values
        hull_indices = convex_hull_idx[i]
        # valid_hull_indices = hull_indices[hull_indices != -1]
        # ic(valid_hull_indices)
        # ic(hull_indices)
        max_idx = torch.max(hull_indices)
        hull_points = np.zeros((max_idx+2, 2))
        for j in range(max_idx+1):
            hull_points[j] = reordered_footpoints[i][torch.where(hull_indices == j)]
        hull_points[max_idx+1] = reordered_footpoints[i][torch.where(hull_indices == 0)]
        
        # Plot the points that make up the convex hull
        ax.scatter(hull_points[:, 0], hull_points[:, 1], color='red', label='Hull Points')
        
        # Draw the convex hull by connecting the points in the exact order of the indices
        ax.plot(hull_points[:, 0], hull_points[:, 1], 'r--', label='Convex Hull')
        # if len(valid_hull_indices) > 0:
        #     # Get the corresponding points from reordered_footpoints using the valid convex hull indices
        #     hull_points = reordered_footpoints[i][valid_hull_indices]
        #     ic(hull_points)
            
        #     # Plot the points that make up the convex hull
        #     ax.scatter(hull_points[:, 0], hull_points[:, 1], color='red', label='Hull Points')
            
        #     # Draw the convex hull by connecting the points in the exact order of the indices
        #     ax.plot(hull_points[:, 0], hull_points[:, 1], 'r--', label='Convex Hull')

        # Set the plot title and labels
        ax.set_title(f'Convex Hull for Sample {i + 1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Define a larger set of polygon points
    # polygon_points = torch.tensor([
    #     # First polygon: A triangle with two extra points
    #     [[0, 0], [2, 0], [1, 1], [0.5, 0.5], [1.5, 0.5]],
    #     # Second polygon: A square
    #     [[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]],
    #     # Third polygon: A pentagon (with all valid points)
    #     [[1, 0], [2, 1], [1.5, 2], [0.5, 2], [0, 1]],
    #     # Fourth polygon: A line (degenerate polygon)
    #     [[1, 1], [3, 1], [0, 0], [0, 0], [0, 0]],
    #     # Fifth polygon: A complex shape with some internal points (for anchor point 5)
    #     [[0, 0], [3, 0], [3, 3], [0, 3], [1.5, 1.5]],
    #     # Fifth polygon duplicate: A complex shape with some internal points (for anchor point 6)
    #     [[0, 0], [3, 0], [3, 3], [0, 3], [1.5, 1.5]]
    # ], dtype=torch.float32)

    # # Define the point mask, indicating the number of valid points for each polygon
    # point_mask = torch.tensor([
    #     [1, 1, 1, 0, 0],  # Only first three points are valid (triangle)
    #     [1, 1, 1, 1, 0],  # First four points form a square
    #     [1, 1, 1, 1, 1],  # All points are valid (pentagon)
    #     [0, 0, 0, 0, 0],  # Only two points are valid (line)
    #     [1, 1, 1, 1, 1],  # All points valid (complex shape for anchor point 5)
    #     [1, 1, 1, 1, 1]   # All points valid (complex shape for anchor point 6)
    # ], dtype=torch.bool)

    # # Define anchor points for testing
    # anchor_points = torch.tensor([
    #     [1, 0.5],  # Inside the first triangle
    #     [3, 1],    # Outside the second square
    #     [1.5, 1.5],# Inside the pentagon
    #     [2, 1],    # On the line (fourth polygon)
    #     [4, 4],    # Far outside the complex shape (fifth polygon for anchor point 5)
    #     [1.9, 1.9] # Inside the complex shape (fifth polygon for anchor point 6)
    # ], dtype=torch.float32)

    # # Run the test
    # min_distances = compute_min_distance(polygon_points, point_mask, anchor_points)
    # print("Min Distances:", min_distances)
    # Example footpoints and footpoints_mask
    N = 4  # Number of samples
    M = 8   # Number of points per sample

    # Generate random footpoints in 2D space
    footpoints = torch.rand(N, M, 2) * 10.0  # Random points in the range [0, 10)

    # Generate random footpoints_mask
    # Ensure that each sample has at least one valid point
    footpoints_mask = torch.zeros(N, M, dtype=torch.bool)
    for i in range(N):
        # Random number of valid points between 1 and M
        num_valid_points = torch.randint(1, M+1, (1,)).item()
        # Random indices to set as valid
        valid_indices = torch.randperm(M)[:num_valid_points]
        footpoints_mask[i, valid_indices] = True

    # Set invalid points to NaN for clarity (optional)
    footpoints[~footpoints_mask] = float('nan')

    # Compute convex hull
    reordered_footpoints, convex_hull_idx = compute_convex_hull(footpoints, footpoints_mask)

    ic(footpoints, reordered_footpoints, convex_hull_idx)


    # Visualize the convex hulls
    points = torch.rand(N, 2) * 10.
    ic(compute_signed_distance(reordered_footpoints, convex_hull_idx, points))
    visualize_convex_hulls(footpoints, footpoints_mask, reordered_footpoints, convex_hull_idx, points)


if __name__ == '__main__':
    main()