from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs.base.zmp import *
from legged_gym.utils.math import *
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict, draw_axis, get_object_size
from legged_gym.envs.g1.g1_full_config import G1GraspCfg
from icecream import ic


class LeggedRobot(BaseTask):
    def __init__(self, cfg: G1GraspCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.iteration = 0

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for i in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            '''
            NOTE(ytcho)
            Why we compute the linear and angular momentum here?
            Since the ZMP is calculated based on the **derivative**
            of the (lin, ang) momentum, in order to accurately 
            estimate the derivative, we should compute it by subtracting
            sim_dt, not a control_dt.

            Computing (lin, ang) momentum derivative by subtracting
            control_dt would result in high instability.
            '''
            if i == self.cfg.control.decimation-1:
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.prev_lin_momentum, self.prev_ang_momentum = \
                    compute_lin_ang_momentum(
                        self.rb_com,
                        self.rb_inertia,
                        self.rb_mass,
                        self.rigid_body_state,
                        self.root_states)
                
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
    
        self._compute_hand_pose()
        self._compute_foot_pose()
        self._compute_obj_pose_base()
        self._compute_foot_pose_base()
        self._compute_foot_contact()

        self._compute_pelvis_pose()
        # Update foot contact information
        self.foot_contacts = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.

        # Update lifted information
        self.check_lifted()

        self.CoM = compute_com(
            self.rb_com,
            self.rb_mass,
            self.total_mass,
            self.rigid_body_state)

        self.lin_momentum, self.ang_momentum = compute_lin_ang_momentum(
            self.rb_com,
            self.rb_inertia,
            self.rb_mass,
            self.rigid_body_state,
            self.root_states)

        self.ZMP = compute_zmp(
            self.total_mass, self.cfg.sim.gravity, 
            self.dt,
            self.lin_momentum, self.prev_lin_momentum,
            self.ang_momentum, self.prev_ang_momentum,
            self.foot_contacts,
            self.CoM,
            self.root_states)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # Track the contact information
        self.last_last_contacts = self.last_contacts
        self.last_contacts = self.foot_contacts

        if self.viewer:
            self.gym.clear_lines(self.viewer)
            self._draw_debug_vis()

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
        """
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom_a = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 0))
        sphere_geom_b = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 0))
        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 1))
        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))

        for i in range(self.num_envs):
            
            # Display the hand position
            sphere_pose_2 = gymapi.Transform(
                gymapi.Vec3(self.left_hand_pos[i, 0],
                            self.left_hand_pos[i, 1], 
                            self.left_hand_pos[i, 2]), 
                r=None)
            # gymutil.draw_lines(sphere_geom_2, self.gym, 
            #                    self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(
                gymapi.Vec3(self.right_hand_pos[i, 0],
                            self.right_hand_pos[i, 1],
                            self.right_hand_pos[i, 2]), 
                r=None)
            # gymutil.draw_lines(sphere_geom_3, self.gym, 
            #                    self.viewer, self.envs[i], sphere_pose_3) 

            # Display the hand axis
            # draw_axis(self.gym, self.viewer, self.envs[i],
            #           self.left_hand_pos[None, i, :], 
            #           self.left_hand_quat[None, i, :], device=self.device)
            # draw_axis(self.gym, self.viewer, self.envs[i],
            #           self.right_hand_pos[None, i, :], 
            #           self.right_hand_quat[None, i, :], device=self.device)
            # Display the foot axis
            # draw_axis(self.gym, self.viewer, self.envs[i],
            #           self.left_feet_pos[None, i, :], 
            #           self.left_feet_quat[None, i, :], device=self.device)
            # draw_axis(self.gym, self.viewer, self.envs[i],
            #           self.right_feet_pos[None, i, :], 
            #           self.right_feet_quat[None, i, :], device=self.device)

            # Display the COM(for debugging purpose)
            # self._compute_com()
            # sphere_pose_a = gymapi.Transform(
            #     gymapi.Vec3(self.CoM[i, 0],
            #                 self.CoM[i, 1], 
            #                 0.), 
            #     r=None)
            sphere_pose_a = gymapi.Transform(
                gymapi.Vec3(self.ZMP[i, 0],
                            self.ZMP[i, 1], 
                            0.), 
                r=None)
            gymutil.draw_lines(sphere_geom_a, self.gym, 
                               self.viewer, self.envs[i], sphere_pose_a) 
            sphere_pose_b = gymapi.Transform(
                gymapi.Vec3((self.left_feet_pos[i, 0]+self.right_feet_pos[i, 0])/2,
                            (self.left_feet_pos[i, 1]+self.right_feet_pos[i, 1])/2, 
                            (self.left_feet_pos[i, 2]+self.right_feet_pos[i, 2])/2), 
                r=None)
            # gymutil.draw_lines(sphere_geom_b, self.gym, 
            #                    self.viewer, self.envs[i], sphere_pose_b) 

            # obj_right = self.obj_root_states[..., :3] + quat_apply(
            #     self.obj_root_states[..., 3:7],
            #     to_torch([0., -0.5 * self.cfg.object.box_size[1], 0.], device=self.device).expand(self.num_envs, -1))
            # sphere_pose_c = gymapi.Transform(
            #     gymapi.Vec3(obj_right[i, 0],
            #                 obj_right[i, 1], 
            #                 obj_right[i, 2]), 
            #     r=None)
            sphere_pose_c = gymapi.Transform(
                gymapi.Vec3(
                            self.CoM[i, 0],
                            self.CoM[i, 1],
                            0.),
                r=None)
            gymutil.draw_lines(sphere_geom_b, self.gym, 
                               self.viewer, self.envs[i], sphere_pose_c) 
            


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.jump_buf = torch.logical_and(
            ~(self.left_feet_contact | self.right_feet_contact), 
            self.episode_length_buf > 10)

        self.contact_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) \
                > self.cfg.env.termination.termination_force, dim=1)
        # Termination decided by the base's rpy
        self.rpy_buf = torch.logical_or(
            torch.abs(self.rpy[:,0]) > self.cfg.env.termination.rpy_thresh[0], 
            torch.abs(self.rpy[:,1]) > self.cfg.env.termination.rpy_thresh[1])

        # Termination decided by the base height
        self.height_buf = torch.logical_or(
            self.base_pos[..., 2]>1.0, self.base_pos[..., 2]<0.2)

        # Termination in case of object falls
        self.obj_buf = self.obj_root_states[..., 2] < 0

        self.completion_buf = self.lifted_time > self.cfg.object.holding_time_threshold

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf = self.jump_buf | self.contact_buf | self.rpy_buf | self.height_buf | self.obj_buf | self.time_out_buf | self.completion_buf
        # if torch.any(self.reset_buf):
        #     ic(self.jump_buf, self.contact_buf, self.rpy_buf, self.height_buf, self.obj_buf)
        #     ic(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1))
        #     time.sleep(2)
        # if torch.any(self.contact_buf):
        #     ic(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1))
            # ic(self.jump_buf, self.contact_buf, self.rpy_buf, self.height_buf, self.obj_buf)
        #     ic(self.rpy)
        #     ic(self.base_pos[..., 2])
            # ic(self.last_last_contacts, self.last_contacts, self.foot_contacts, self.episode_length_buf)

        '''
        For debugging
        '''
        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # if torch.any(self.episode_length_buf[env_ids] < 3):
        #     ic(self.episode_length_buf[env_ids])
        #     ic(self.rpy[env_ids, :])
        #     ic(self_contact_buf[env_ids])
        #     ic(rpy_buf[env_ids])
        #     ic(height_buf[env_ids])
        #     ic(obj_buf[env_ids])
        #     raise NotImplementedError

    def check_lifted(self):
        '''
        Check if the robot lifted the object above the succeed threshold
        Hand position need to be refreshed before calling this function
        '''
        left_close_enough = torch.norm(self.left_hand_pos - self.obj_root_states[..., :3], dim=-1) < self.cfg.object.box_size[0]
        right_close_enough = torch.norm(self.right_hand_pos - self.obj_root_states[..., :3], dim=-1)< self.cfg.object.box_size[0]
        close_enough = torch.logical_and(left_close_enough, right_close_enough)
        self.is_lifted = torch.logical_and(
                self.obj_root_states[..., 2] > self.cfg.object.succeed_threshold, 
                close_enough)
        
        # Reset the lifted time buffer if not lifted anymore
        self.lifted_time *= self.is_lifted
        # Accumulate the lifted time
        self.lifted_time += self.is_lifted
            


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        if self.cfg.object.use_curriculum:
            self.update_curriculum()


        # self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_contacts[env_ids] = True
        self.last_last_contacts[env_ids] = True
        self.closest_left[env_ids] = self.cfg.object.initial_hand_obj_dist
        self.closest_right[env_ids] = self.cfg.object.initial_hand_obj_dist
        self.feet_air_time[env_ids] = 0.
        self.prev_lin_momentum[env_ids] = 0.
        self.prev_ang_momentum[env_ids] = 0.
        self.is_lifted[env_ids] = False
        self.lifted_time[env_ids] = 0.
        # self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        self.extras["stats"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["stats"]['mean_rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / self.episode_length_buf[env_ids])
            self.episode_sums[key][env_ids] = 0.

        for key in self.episode_metric_sums.keys():
            self.extras["episode"]['metric_' + key] = torch.mean(self.episode_metric_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["stats"]['mean_metric_' + key] = torch.mean(self.episode_metric_sums[key][env_ids] / self.episode_length_buf[env_ids])

            self.episode_metric_sums[key][env_ids] = 0.
        
        self.extras["stats"]['mean_eplen'] = torch.mean(self.episode_length_buf[env_ids])


        # Log the **cause** that lead into the termination
        # The sum of termination ratio might exceed 1, in case of
        # multiple termination conditions are activated at once

        self.extras["stats"]["reset_jump"] = torch.mean(self.jump_buf[env_ids].float())
        self.extras["stats"]["reset_contact"] = torch.mean(self.contact_buf[env_ids].float())
        self.extras["stats"]["reset_base_rpy"] = torch.mean(self.rpy_buf[env_ids].float())
        self.extras["stats"]["reset_base_height"] = torch.mean(self.height_buf[env_ids].float())
        self.extras["stats"]["reset_obj_fall"] = torch.mean(self.obj_buf[env_ids].float())
        self.extras["stats"]["reset_time_out"] = torch.mean(self.time_out_buf[env_ids].float())
        self.extras["stats"]["reset_completion"] = torch.mean(self.completion_buf[env_ids].float())

        # Log the maximum height of the object before termination
        self.extras["stats"]["max_obj_height"] = torch.mean(self.highest_object[env_ids].float())

        # if self.cfg.commands.curriculum:
        #     self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
        self.highest_object[env_ids] = self.obj_root_states[env_ids, 2]
        # Pelvis height reward only activated during lifting
        self.highest_pelvis[env_ids] = 0.1
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            # For debugging: detecting NaNs
            if torch.isnan(rew).any():
                ic(name, rew)
                raise ValueError
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  
            self.base_lin_vel * self.obs_scales.lin_vel, # 3
            self.base_ang_vel  * self.obs_scales.ang_vel, # 3
            self.projected_gravity, # 3
            # self.get_body_orientation(), # 2
            # ZMP, foot avg position
            # self.avg_foot_pos_base, #3
            # self.ZMP_base, #3
            # Foot pos, ori, and contact
            # self.left_feet_pos,
            # self.left_feet_ori_base,
            # self.right_feet_pos,
            # self.right_feet_ori_base,
            # self.foot_contacts,
            # Left hand delta
            self.obj_pos_wrt_left, # 3
            self.obj_ori_wrt_left, # 3
            # Right hand delta
            self.obj_pos_wrt_right, # 3
            self.obj_ori_wrt_right, # 3
            # Binary fingertip contact information
            torch.norm(self.contact_forces[:, self.fingertip_indices, :], dim=-1) > 0.5, #6

            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 37
            self.dof_vel * self.obs_scales.dof_vel, # 37
            self.actions # 37
        ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        # Check for Nans
        if torch.isnan(self.obs_buf).any():
            nan_indices = torch.nonzero(torch.isnan(self.obs_buf), as_tuple=False)
            raise ValueError(f"NaN values found at indices: {nan_indices}")
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                # Modify dof limit according to the config file
                for dof_name in self.cfg.init_state.limits.keys():
                    if dof_name in self.dof_names[i]:
                        props["lower"][i] = self.cfg.init_state.limits[dof_name][0]
                        props["upper"][i] = self.cfg.init_state.limits[dof_name][1]

                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                self.rigid_body_mass[i] = p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
            self.total_mass = sum
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            raise NotImplementedError
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _obj_process_rigid_body_props(self, props, env_id):
        if self.cfg.object.use_curriculum:
            # rng_mass = self.cfg.object.added_mass_range
            # rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
            props[0].mass = self.initial_obj_mass * self.mass_multiplier
        else:
            pass
        
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # Disable command relate code
        '''
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
        '''

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        # self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        # self.dof_vel[env_ids] = 0.
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.3, 0.3, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)

        # NOTE(ytcho): Given the environment index k, corresponding
        # robot agent index is 2*k, and corresponding object index 
        # is 2*k+1
        env_ids_int32 = 2 * env_ids.clone().to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # Should be avoided
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        if self.cfg.domain_rand.randomize_init_orn:
            # Initialize the robot root state in random orientation
            self.root_states[env_ids, 3:7] = quat_from_euler(
                torch.cat([
                    torch.zeros(len(env_ids), 2, device=self.device),
                    # torch_rand_float(0, 2* torch.pi, (len(env_ids), 1), device=self.device)],
                    # Decrease the range
                    torch_rand_float(-0.5*torch.pi, 0.5* torch.pi, (len(env_ids), 1), device=self.device)],
                    # torch_rand_float(-torch.pi/4, torch.pi/4, (len(env_ids), 1), device=self.device)],
                    dim=-1)
            )
        # base velocities
        # Disable random vel
        # NOTE(ytcho): Temporarily Disable random vel at init!!!

        # self.root_states[env_ids, 7:13] = torch_rand_float(-1., 1., (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        self.obj_root_states[env_ids, :3] = self.root_states[env_ids, :3].clone()

        self.obj_root_states[env_ids, 0] += 0.6
        self.obj_root_states[env_ids, 2] = 0.5 * self.cfg.object.box_size[2] + 0.01
        self.obj_root_states[env_ids, 3:7] = torch.tensor([0., 0., 0., 1.], device=self.device, requires_grad=False,)
        self.obj_root_states[env_ids, 7:13] = 0.

        # Should be avoided
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))

        # NOTE(ytcho): Given the environment index k, corresponding
        # robot agent index is 2*k, and corresponding object index 
        # is 2*k+1
        env_ids_int32 = torch.cat(
            (2*env_ids.clone().to(dtype=torch.int32), 
                2*env_ids.clone().to(dtype=torch.int32)+1), 
            dim=-1)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
    
    def _compute_foot_pose(self):
        self.foot_pose = self.rigid_body_state[..., self.feet_indices, :] 
        self.left_feet_pos = self.foot_pose[:, 0, :3]
        self.left_feet_quat = self.foot_pose[:, 0, 3:7]
        self.right_feet_pos = self.foot_pose[:, 1, :3]
        self.right_feet_quat = self.foot_pose[:, 1, 3:7]

    def _compute_hand_pose(self):
        '''
        Computes the (pos, quat) of both hands. Need to be called after 
        refreshing the rigid_body_state_tensor
        '''
        # [N, 2, 4] -> [2*N, 4]
        self.hands_quat = self.rigid_body_state[..., self.elbow_indices, 3:7] 
        self.hands_pos = \
            self.rigid_body_state[..., self.elbow_indices, :3].view(-1, 3) + \
            quat_apply(self.hands_quat.view(-1, 4), 
                       to_torch([self.cfg.asset.elbow_hand_offset, 0., 0.], 
                                device=self.device
                        ).expand(2*self.num_envs, -1)
            )
        self.hands_pos = self.hands_pos.view(-1, len(self.elbow_indices), 3)

        self.left_hand_pos = self.hands_pos[:, 0, :]
        self.right_hand_pos = self.hands_pos[:, 1, :]
        self.left_hand_quat = self.hands_quat[:, 0, :]
        self.right_hand_quat = self.hands_quat[:, 1, :]
    
    def _compute_foot_pose_base(self):
        self.left_feet_pos_base = quat_rotate_inverse(
            self.left_feet_quat,
            self.left_feet_pos - self.root_states[..., :3]
        )
        self.left_feet_ori_base = wrap_to_pi_minuspi(axis_angle_from_quat(quat_multiply(
            self.left_feet_quat, quat_inverse(self.base_quat)))
        )
        self.right_feet_pos_base = quat_rotate_inverse(
            self.right_feet_quat,
            self.right_feet_pos - self.root_states[..., :3]
        )
        self.right_feet_ori_base = wrap_to_pi_minuspi(axis_angle_from_quat(quat_multiply(
            self.right_feet_quat, quat_inverse(self.base_quat)))
        )
        # ic(self.left_feet_pos_base, self.left_feet_ori_base)

    def _compute_pelvis_pose(self):
        self.pelvis_pos = torch.mean(self.rigid_body_state[..., self.hip_indices, :3], dim=1)
    
    def _compute_foot_contact(self):
        '''
        Compute if there is contact with the foot <-> ground
        '''
        self.left_feet_contact = torch.logical_and(
            (self.last_contacts[..., 0] | self.foot_contacts[..., 0]),
            self.left_feet_pos[..., 2] < 0.1)

        self.right_feet_contact = torch.logical_and(
            (self.last_contacts[..., 1] | self.foot_contacts[..., 1]),
            self.right_feet_pos[..., 2] < 0.1)
   
    def _compute_obj_pose_base(self):
        '''
        Computes the (pos, quat) of the object wrt the base frame 
        of the robot. Need to be called after refreshing the 
        root_state_tensor, and computing the hand states
        '''
        self.obj_pos_base = quat_rotate_inverse(
            self.base_quat,
            self.obj_root_states[..., :3] - self.root_states[..., :3])

        self.obj_quat_base = quat_multiply(
            quat_inverse(self.base_quat),
            self.obj_root_states[..., 3:7])
        
        # Compute the object pose wrt to the hand, in the base frame
        self.obj_pos_wrt_left = quat_rotate_inverse(
            self.base_quat,
            self.obj_root_states[..., :3] - self.left_hand_pos)

        self.obj_ori_wrt_left = wrap_to_pi_minuspi(axis_angle_from_quat(quat_multiply(
            self.obj_root_states[..., 3:7], quat_inverse(self.left_hand_quat))))

        self.obj_pos_wrt_right = quat_rotate_inverse(
            self.base_quat,
            self.obj_root_states[..., :3] - self.right_hand_pos)
        self.obj_ori_wrt_right = wrap_to_pi_minuspi(axis_angle_from_quat(quat_multiply(
            self.obj_root_states[..., 3:7], quat_inverse(self.right_hand_quat))))

    def get_body_orientation(self, return_yaw=False):
        body_angles = wrap_to_pi_minuspi(axis_angle_from_quat(self.base_quat))

        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        if self.add_noise:
            raise NotImplementedError
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec


    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        # create some wrapper tensors for different slices
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, 2, 13)
        self.root_states = self._root_states[..., 0, :]
        self.obj_root_states = self._root_states[..., 1, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self._contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies+1, 3) # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = self._contact_forces[..., :-1, :]
        self.obj_contact_forces = self._contact_forces[..., -1, :]
        # self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies + 1, 13)
        self.rigid_body_state = self._rigid_body_state[:, :-1, :]
        self.obj_rigid_body_state = self._rigid_body_state[:, -1, :]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # Momentums
        self.lin_momentum = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.ang_momentum = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_lin_momentum = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_ang_momentum = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # ZMPs
        self.ZMP = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        # self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.completion_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.jump_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.contact_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.rpy_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.height_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.obj_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.CoM = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.left_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.right_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_left_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_right_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.closest_left = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * self.cfg.object.initial_hand_obj_dist
        self.closest_right= torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * self.cfg.object.initial_hand_obj_dist
        self.highest_object= torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.highest_pelvis= torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Lifted time-related buffers
        self.is_lifted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.lifted_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)


        self._compute_hand_pose()
        self._compute_foot_pose()
        self._compute_obj_pose_base()
        self._compute_foot_pose_base()
        self._compute_foot_contact()

      

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if True:
            # Prepare jacobians and rb states
            self.gym.refresh_rigid_body_state_tensor(self.sim)        

            self.rb_inertia = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3, 3), device=self.device) # [comX, comY, comZ], [Ix, Iy, Iz]
            self.rb_mass = gymtorch.torch.zeros((self.num_envs, self.num_bodies), device=self.device) # link mass
            self.rb_com = gymtorch.torch.zeros((self.num_envs, self.num_bodies, 3), device = self.device) # [comX, comY, comZ] in link's origin frame 

            self.obj_inertia = gymtorch.torch.zeros((self.num_envs, 3, 3), device=self.device) # [comX, comY, comZ], [Ix, Iy, Iz]
            self.obj_mass = gymtorch.torch.zeros((self.num_envs), device=self.device) # link mass
            self.obj_com = gymtorch.torch.zeros((self.num_envs, 3), device = self.device) # [comX, comY, comZ] in link's origin frame 
            
            # Reconstruct rb_props as tensor        
            for i in range(self.num_envs):
                for key, N in self.body_names_dict.items():
                    rb_props = self.gym.get_actor_rigid_body_properties(self.envs[i], self.actor_handles[i])[N]
                    # inertia tensors are about link's CoM frame
                    self.rb_com[i, N, :] = gymtorch.torch.tensor([rb_props.com.x, rb_props.com.y, rb_props.com.z], device=self.device)
                    self.rb_inertia[i, N, 0, :] = gymtorch.torch.tensor([rb_props.inertia.x.x, -rb_props.inertia.x.y, -rb_props.inertia.x.z], device=self.device)
                    self.rb_inertia[i, N, 1, :] = gymtorch.torch.tensor([-rb_props.inertia.y.x, rb_props.inertia.y.y, -rb_props.inertia.y.z], device=self.device)
                    self.rb_inertia[i, N, 2, :] = gymtorch.torch.tensor([-rb_props.inertia.z.x, -rb_props.inertia.z.y, rb_props.inertia.z.z], device=self.device)
                    # see how inertia tensor is made : https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf
                    self.rb_mass[i, N] = rb_props.mass

                obj_props = self.gym.get_actor_rigid_body_properties(self.envs[i], self.obj_actor_handles[i])[0]
                self.obj_com[i, :] = gymtorch.torch.tensor([obj_props.com.x, obj_props.com.y, obj_props.com.z], device=self.device)
                self.obj_mass[i] = obj_props.mass
                self.obj_inertia[i, 0, :] = gymtorch.torch.tensor([obj_props.inertia.x.x, -obj_props.inertia.x.y, -obj_props.inertia.x.z], device=self.device)
                self.obj_inertia[i, 1, :] = gymtorch.torch.tensor([-obj_props.inertia.y.x, obj_props.inertia.y.y, -obj_props.inertia.y.z], device=self.device)
                self.obj_inertia[i, 2, :] = gymtorch.torch.tensor([-obj_props.inertia.z.x, -obj_props.inertia.z.y, obj_props.inertia.z.z], device=self.device)

    def _get_curriculum_value(self, schedule, init_range, final_range, counter):
        return np.clip((counter - schedule[0]) / (schedule[1] - schedule[0]), 0, 1) * (final_range - init_range) + init_range
    
    def update_curriculum(self):
        '''
        Update curriculum values
        '''

        self.mass_multiplier = self._get_curriculum_value(
            self.cfg.object.schedule_steps, 
            self.cfg.object.mass_range[0], 
            self.cfg.object.mass_range[1], 
            self.iteration)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            # else:
            #     self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

        # metric episode sums
        self.metric_names = [
                             'left_hand_obj_dist', 
                             'right_hand_obj_dist', 
                             'obj_height',
                             'left_contact',
                             'right_contact',
                             'completion',
                             'base_lin_acc',
                             'com_avgfoot_dist',
                             'zmp_avgfoot_dist',
                             'obj_base_delta_ori',
                             'dof_vel',
                             'obj_base_delta_ori',
                             'left_foot_delta_ori',
                             'right_foot_delta_ori',
                             'left_foot_delta_z_axis',
                             'right_foot_delta_z_axis',
                             'left_foot_vel',
                             'right_foot_vel',
                             'torque',
                             'obj_zvel',
                             'obj_xyvel',
                             'is_lifted',
                             'single_foot_contact',
                             'left_foot_height',
                             'right_foot_height',
                             'pelvis_height',
                             'base_obj_dist',
                             ]

        self.episode_metric_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) \
            for name in self.metric_names}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)

        self.rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_dict = self.gym.get_asset_rigid_body_dict(robot_asset)

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        elbow_names = [s for s in body_names if self.cfg.asset.elbow_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        fingertip_names = []
        for name in self.cfg.asset.fingertip_links:
            fingertip_names.extend([s for s in body_names if name in s])
        left_finger_names = []
        for name in self.cfg.asset.left_fingers:
            left_finger_names.extend([s for s in body_names if name in s])
        right_finger_names = []
        for name in self.cfg.asset.right_fingers:
            right_finger_names.extend([s for s in body_names if name in s])
        hip_names =[]
        for name in self.cfg.asset.hip:
            hip_names.extend([s for s in body_names if name in s])


        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # Object assets
        asset_options = gymapi.AssetOptions()
        # asset_options.density = 1000
        asset_options.density = self.cfg.object.density
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        obj_asset = self.gym.create_box(
            self.sim, 
            self.cfg.object.box_size[0], 
            self.cfg.object.box_size[1], 
            self.cfg.object.box_size[2], 
            asset_options)

        obj_start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.obj_actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # NOTE(ytcho): Aggretation(to avoid OOM), IMPORTANT!!!
            self.gym.begin_aggregate(env_handle, self.num_bodies+1, self.num_shapes+1, True)

            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)


            # Create object instance
            obj_pos = pos.clone()
            obj_pos[0] += 1.
            obj_start_pose.p = gymapi.Vec3(*obj_pos)
            obj_handle = self.gym.create_actor(env_handle, obj_asset, obj_start_pose, "object", i, self.cfg.asset.self_collisions, 0)
            self.obj_actor_handles.append(obj_handle)
            # For segmentation mask(box_id = 1)
            # self.gym.set_rigid_body_segmentation_id(env_handle, obj_handle, 0, self.cfg.box.box_seg_idx)
            box_body_props = self.gym.get_actor_rigid_body_properties(env_handle, obj_handle)
            if i==0:
                self.initial_obj_mass = box_body_props[0].mass
            box_body_props = self._obj_process_rigid_body_props(box_body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, obj_handle, box_body_props, recomputeInertia=True)

            # Finish the aggretation
            self.gym.end_aggregate(env_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.elbow_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(elbow_names)):
            self.elbow_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], elbow_names[i])
        
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        self.fingertip_indices = torch.zeros(len(fingertip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(fingertip_names)):
            self.fingertip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], fingertip_names[i])

        self.left_finger_indices = torch.zeros(len(left_finger_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_finger_names)):
            self.left_finger_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_finger_names[i])
        self.right_finger_indices = torch.zeros(len(right_finger_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_finger_names)):
            self.right_finger_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_finger_names[i])
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], hip_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
      
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        # For mass curriculum
        self.mass_multiplier = 1.


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        torque = torch.norm(self.torques, dim=-1)
        self.episode_metric_sums['torque'] += torque
        # return torch.sum(torch.square(self.torques), dim=1)
        return torque

    def _reward_dof_vel(self):
        # Penalize dof velocities
        # return torch.sum(torch.square(self.dof_vel), dim=1)
        dof_vel = torch.norm(self.dof_vel, dim=-1)
        self.episode_metric_sums['dof_vel'] += dof_vel
        return torch.square(dof_vel)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        rew_smooth = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return rew_smooth

    def _reward_action_double_rate(self):
        # Penalize changes in actions
        rew_smooth = torch.sum(torch.square(self.actions 
                                            -2*self.last_actions
                                            +self.last_last_actions), 
                               dim=1)
        return rew_smooth
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_completion(self):
        '''
        Task completion reward
        '''
        self.episode_metric_sums['completion'] += self.completion_buf

        return self.completion_buf

    def _reward_termination(self):
        # Terminal reward / penalty
        # Additionaly don't penalize completion case
        return self.reset_buf * ~self.time_out_buf * ~self.completion_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_dist_left(self):
        # Reward based on the hand <-> object distance
        left_distance = torch.norm(self.left_hand_pos - self.obj_root_states[..., :3], dim=-1)
        self.episode_metric_sums['left_hand_obj_dist'] += left_distance
        dist_rew = torch.exp(-5 * torch.square(left_distance))
        return dist_rew

    def _reward_dist_right(self):
        # Reward based on the hand <-> object distance
        right_distance = torch.norm(self.right_hand_pos - self.obj_root_states[..., :3], dim=-1)
        self.episode_metric_sums['right_hand_obj_dist'] += right_distance
        dist_rew = torch.exp(-5 * torch.square(right_distance))
        return dist_rew

    def _reward_dist_left_v2(self):
        '''
        Compute the distance-based reward between the left face of
        the object and the left hand
        '''
        obj_left = self.obj_root_states[..., :3] + quat_apply(
            self.obj_root_states[..., 3:7],
            to_torch([0., 0.5 * self.cfg.object.box_size[1], 0.], device=self.device).expand(self.num_envs, -1))

        self.left_distance = torch.norm(self.left_hand_pos - obj_left, dim=-1)
        self.episode_metric_sums['left_hand_obj_dist'] += self.left_distance

        dist_delta = self.closest_left - self.left_distance
        self.closest_left = torch.minimum(self.closest_left, self.left_distance)
        dist_delta = torch.clip(dist_delta, 0., 3.)
        reward = torch.tanh(30.0 * dist_delta)
        # ic(self.closest_left, self.left_distance)
        # ic(dist_delta, reward)
        return reward

    def _reward_dist_right_v2(self):
        '''
        Compute the distance-based reward between the left face of
        the object and the right hand
        '''
        obj_right = self.obj_root_states[..., :3] + quat_apply(
            self.obj_root_states[..., 3:7],
            to_torch([0., -0.5 * self.cfg.object.box_size[1], 0.], device=self.device).expand(self.num_envs, -1))

        self.right_distance = torch.norm(self.right_hand_pos - obj_right, dim=-1)
        self.episode_metric_sums['right_hand_obj_dist'] += self.right_distance

        dist_delta = self.closest_right- self.right_distance
        self.closest_right = torch.minimum(self.closest_right, self.right_distance)
        dist_delta = torch.clip(dist_delta, 0., 3.)
        reward = torch.tanh(30.0 * dist_delta)
        # ic(self.right_distance, reward)
        # ic(reward, reward.mean())
        return reward

    def _reward_dist_left_v3(self):
        '''
        Compute the distance-based reward between the left face of
        the object and the left hand
        '''
        obj_left = self.obj_root_states[..., :3] + quat_apply(
            self.obj_root_states[..., 3:7],
            to_torch([0., 0.5 * self.cfg.object.box_size[1], 0.], device=self.device).expand(self.num_envs, -1))

        self.left_distance = torch.norm(self.left_hand_pos - obj_left, dim=-1)
        self.episode_metric_sums['left_hand_obj_dist'] += self.left_distance

        dist_delta = self.prev_left_distance - self.left_distance
        self.prev_left_distance = self.left_distance
        dist_delta = torch.clip(dist_delta, -0.5, 0.5)
        # ic(dist_delta)
        reward = torch.tanh(30.0 * dist_delta)
        # ic(reward, reward.mean())
        return reward

    def _reward_dist_right_v3(self):
        '''
        Compute the distance-based reward between the right face of
        the object and the right hand
        '''
        obj_right = self.obj_root_states[..., :3] + quat_apply(
            self.obj_root_states[..., 3:7],
            to_torch([0., -0.5 * self.cfg.object.box_size[1], 0.], device=self.device).expand(self.num_envs, -1))

        self.right_distance = torch.norm(self.right_hand_pos - obj_right, dim=-1)
        self.episode_metric_sums['right_hand_obj_dist'] += self.right_distance

        dist_delta = self.prev_right_distance - self.right_distance
        self.prev_right_distance = self.right_distance
        dist_delta = torch.clip(dist_delta, -0.5, 0.5)
        # ic(dist_delta)
        reward = torch.tanh(30.0 * dist_delta)
        return reward

    def _reward_pickup(self):
        # Reward based on the object's z height
        # Activated only if the fingertip has contact with the object
        left_close_enough = torch.norm(self.left_hand_pos - self.obj_root_states[..., :3], dim=-1) < self.cfg.object.box_size[0]
        right_close_enough = torch.norm(self.right_hand_pos - self.obj_root_states[..., :3], dim=-1)< self.cfg.object.box_size[0]
        close_enough = torch.logical_and(left_close_enough, right_close_enough)

        # fingertip_contact = torch.any(
        #     torch.norm(self.contact_forces[:, self.fingertip_indices, :], dim=-1) > 0.5,
        #     dim=-1)

        obj_height = torch.maximum(
            self.obj_root_states[..., 2] - 0.5 * self.cfg.object.box_size[2], 
            torch.tensor(0, device=self.device))
        self.episode_metric_sums['obj_height'] += obj_height

        return close_enough * obj_height

    def _reward_pickup_v2(self):
        left_close_enough = torch.norm(self.left_hand_pos - self.obj_root_states[..., :3], dim=-1) < self.cfg.object.box_size[0]
        right_close_enough = torch.norm(self.right_hand_pos - self.obj_root_states[..., :3], dim=-1)< self.cfg.object.box_size[0]
        close_enough = torch.logical_and(left_close_enough, right_close_enough)

        obj_height = self.obj_root_states[..., 2]
        self.episode_metric_sums['obj_height'] += obj_height

        height_delta = obj_height - self.highest_object
        self.highest_object = torch.maximum(self.highest_object, obj_height)
        height_delta = torch.clip(height_delta, 0., 10.)
        lifting_rew = torch.tanh(30.0 * height_delta)
        
        return close_enough * lifting_rew * ~(self.is_lifted) * (self.episode_length_buf > 50)

    def _reward_base_lin_acc(self):
        '''
        Penalizes base linear acceleration
        '''
        lin_acc = torch.norm((
            (self.last_root_vel[..., :3] - self.root_states[..., 7:10]) / self.dt), dim=-1)
        self.episode_metric_sums['base_lin_acc'] += lin_acc

        return torch.square(lin_acc)

    

    def _reward_left_contact(self):
        left_close_enough = torch.norm(self.left_hand_pos - self.obj_root_states[..., :3], dim=-1) < self.cfg.object.box_size[0]
        # Reward if there is a contact between the fingertip <-> object
        left_fingertip_contact = torch.any(
            torch.norm(self.contact_forces[:, self.left_finger_indices, :], dim=-1) > 0.5,
            dim=-1)
        obj_contact = torch.norm(self.obj_contact_forces[..., :2], dim=-1) > 0.5 
        left_contact_rew = left_close_enough * obj_contact * left_fingertip_contact
        self.episode_metric_sums['left_contact'] += left_contact_rew
        return left_contact_rew * ~(self.is_lifted) * (self.episode_length_buf > 50)

    def _reward_right_contact(self):
        right_close_enough = torch.norm(self.right_hand_pos - self.obj_root_states[..., :3], dim=-1) < self.cfg.object.box_size[0]
        # Reward if there is a contact between the fingertip <-> object
        right_fingertip_contact = torch.any(
            torch.norm(self.contact_forces[:, self.right_finger_indices, :], dim=-1) > 0.5,
            dim=-1)
        obj_contact = torch.norm(self.obj_contact_forces[..., :2], dim=-1) > 0.5 
        right_contact_rew = right_close_enough * obj_contact * right_fingertip_contact
        self.episode_metric_sums['right_contact'] += right_contact_rew
        return right_contact_rew * ~(self.is_lifted) * (self.episode_length_buf > 50)

    def _reward_survive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _reward_com_avgfoot_dist(self):
        '''
        Computes reward based on the distance between the 
        average foot position <-> COM
        In theory, it should be the ZMP, but calculating ZMP
        requires a lot of computation resources. Thus we used
        CoM instead...
        '''

        # self._compute_com() 
        avg_foot_pos = 0.5 * (self.left_feet_pos + self.right_feet_pos)
        com_avgfoot_dist = torch.norm((self.CoM-avg_foot_pos)[..., :2], dim=-1)
        self.episode_metric_sums['com_avgfoot_dist'] += com_avgfoot_dist
        # ic(1.0 * com_avgfoot_dist)
        # ic(5.0 * torch.square(com_avgfoot_dist))
        return torch.square(com_avgfoot_dist)

    def _reward_zmp_avgfoot_dist(self):
        '''
        Computes reward based on the distance between the 
        average foot position <-> ZMP
        # In theory, it should be the ZMP, but calculating ZMP
        # requires a lot of computation resources. Thus we used
        # CoM instead...
        '''

        # self._compute_com() 
        avg_foot_pos = 0.5 * (self.left_feet_pos + self.right_feet_pos)
        zmp_avgfoot_dist = torch.norm((self.ZMP-avg_foot_pos[..., :2]), dim=-1)
        # Clip in case of invalid value
        zmp_avgfoot_dist = torch.clip(zmp_avgfoot_dist, -2., 2.)
        # self.episode_metric_sums['com_avgfoot_dist'] += com_avgfoot_dist
        self.episode_metric_sums['zmp_avgfoot_dist'] += zmp_avgfoot_dist
        # ic(1.0 * zmp_avgfoot_dist)
        # ic(1.0 * torch.square(zmp_avgfoot_dist))
        return torch.square(zmp_avgfoot_dist)

    def _reward_zmp_avgfoot_dist_v2(self):
        '''
        Computes reward based on the distance between the 
        average foot position <-> ZMP
        # In theory, it should be the ZMP, but calculating ZMP
        # requires a lot of computation resources. Thus we used
        # CoM instead...
        '''

        # self._compute_com() 
        avg_foot_pos = 0.5 * (self.left_feet_pos + self.right_feet_pos)
        zmp_avgfoot_dist = torch.norm((self.ZMP[..., :2]-avg_foot_pos[..., :2]), dim=-1)
        # Clip in case of invalid value
        # zmp_avgfoot_dist = torch.clip(zmp_avgfoot_dist, -1.2, 1.2)
        zmp_avgfoot_dist = torch.clip(zmp_avgfoot_dist, 0, self.cfg.rewards.zmp.max_dist)
        # self.episode_metric_sums['com_avgfoot_dist'] += com_avgfoot_dist
        self.episode_metric_sums['zmp_avgfoot_dist'] += zmp_avgfoot_dist
        # ic(1.0 * zmp_avgfoot_dist)
        # ic((torch.exp(4 * zmp_avgfoot_dist)-1).mean())
        # return (torch.exp(4 * zmp_avgfoot_dist)-1)
        return (torch.exp(self.cfg.rewards.zmp.sigma * zmp_avgfoot_dist)-1)

    def _reward_com_avgfoot_dist_v2(self):

        avg_foot_pos = 0.5 * (self.left_feet_pos + self.right_feet_pos)
        com_avgfoot_dist = torch.norm((self.CoM[..., :2]-avg_foot_pos[..., :2]), dim=-1)
        com_avgfoot_dist = torch.clip(com_avgfoot_dist, 0, self.cfg.rewards.com.max_dist)
        self.episode_metric_sums['com_avgfoot_dist'] += com_avgfoot_dist
        return (torch.exp(self.cfg.rewards.com.sigma * com_avgfoot_dist)-1)
    
    def _reward_base_orient(self):
        '''
        Computes the orientation difference with the robot base and the object
        only consider pitch and yaw...
        '''

        # Here, we should consider the orientational difference wrt the
        # Global Frame, not wrt the object

        delta_axa = axis_angle_from_quat(self.base_quat)
        delta_ori = torch.norm(delta_axa[..., 1:], dim=-1)
        self.episode_metric_sums['obj_base_delta_ori'] += delta_ori
        return delta_ori

    def _reward_foot_orient(self):
        '''
        Computes the orientation difference with the object and the foot
        '''
        delta_left_quat = quat_multiply(quat_inverse(self.obj_root_states[..., 3:7]), self.left_feet_quat)
        delta_left_axa = axis_angle_from_quat(delta_left_quat)

        delta_right_quat = quat_multiply(quat_inverse(self.obj_root_states[..., 3:7]), self.right_feet_quat)
        delta_right_axa = axis_angle_from_quat(delta_right_quat)

        left_delta_ori = torch.norm(delta_left_axa, dim=-1)
        right_delta_ori = torch.norm(delta_right_axa, dim=-1)

        self.episode_metric_sums['left_foot_delta_ori'] += left_delta_ori
        self.episode_metric_sums['right_foot_delta_ori'] += right_delta_ori
        return left_delta_ori+right_delta_ori

    def _reward_foot_orient_v2(self):
        '''
        Should we constraint our foot to face the forward direction during 
        the pickup process? Probably not....
        Computes the z-axis difference with the foot and the global frame
        '''
        # Z axis of the left foot frame, measured in the global frame
        left_foot_z = quat_apply(
            self.left_feet_quat, 
            to_torch([0., 0., 1.], device=self.device).expand(self.num_envs, -1))

        # To avoid NaNs
        epsilon=1e-7
        left_delta_z_axis = torch.acos(torch.clamp(left_foot_z[..., 2], -1 + epsilon, 1 - epsilon))

        # Z axis of the right foot frame, measured in the global frame
        right_foot_z = quat_apply(
            self.right_feet_quat, 
            to_torch([0., 0., 1.], device=self.device).expand(self.num_envs, -1))
        right_delta_z_axis = torch.acos(torch.clamp(right_foot_z[..., 2], -1 + epsilon, 1 - epsilon))

        self.episode_metric_sums['left_foot_delta_z_axis'] += left_delta_z_axis
        self.episode_metric_sums['right_foot_delta_z_axis'] += right_delta_z_axis
        # ic(left_delta_z_axis, right_delta_z_axis)

        return left_delta_z_axis+right_delta_z_axis

    def _reward_foot_orient_v3(self):
        '''
        Should we constraint our foot to face the forward direction during 
        the pickup process? Probably not....
        Computes the z-axis difference with the foot and the global frame
        '''
        # Z axis of the left foot frame, measured in the global frame
        left_foot_z = quat_apply(
            self.left_feet_quat, 
            to_torch([0., 0., 1.], device=self.device).expand(self.num_envs, -1))

        # To avoid NaNs
        epsilon=1e-7
        left_delta_z_axis = torch.acos(torch.clamp(left_foot_z[..., 2], -1 + epsilon, 1 - epsilon))

        # Z axis of the right foot frame, measured in the global frame
        right_foot_z = quat_apply(
            self.right_feet_quat, 
            to_torch([0., 0., 1.], device=self.device).expand(self.num_envs, -1))
        right_delta_z_axis = torch.acos(torch.clamp(right_foot_z[..., 2], -1 + epsilon, 1 - epsilon))

        self.episode_metric_sums['left_foot_delta_z_axis'] += left_delta_z_axis
        self.episode_metric_sums['right_foot_delta_z_axis'] += right_delta_z_axis

        return torch.square(left_delta_z_axis)+torch.square(right_delta_z_axis)

    def _reward_foot_vel(self):
        '''
        Penalizes the foot velocities
        '''
        # ic(foot_vel)
        foot_vel = torch.norm(self.rigid_body_state[..., self.feet_indices, 7:10], dim=-1)
        left_foot_vel = foot_vel[..., 0]
        right_foot_vel = foot_vel[..., 1]
        self.episode_metric_sums['left_foot_vel'] += left_foot_vel
        self.episode_metric_sums['right_foot_vel'] += right_foot_vel
        return left_foot_vel + right_foot_vel

    def _reward_foot_vel_v2(self):
        '''
        Penalizes the foot velocities
        Changed to square
        '''
        # ic(foot_vel)
        foot_vel = torch.norm(self.rigid_body_state[..., self.feet_indices, 7:10], dim=-1)
        left_foot_vel = foot_vel[..., 0]
        right_foot_vel = foot_vel[..., 1]
        self.episode_metric_sums['left_foot_vel'] += left_foot_vel
        self.episode_metric_sums['right_foot_vel'] += right_foot_vel

        return torch.square(left_foot_vel) + torch.square(right_foot_vel)

    def _reward_foot_height(self):
        '''
        Penalizes the foot height
        Changed to square
        '''
        # ic(foot_vel)
        foot_height = self.rigid_body_state[..., self.feet_indices, 2]
        left_foot_height = foot_height[..., 0]
        right_foot_height = foot_height[..., 1]
        self.episode_metric_sums['left_foot_height'] += left_foot_height
        self.episode_metric_sums['right_foot_height'] += right_foot_height

        # Clamp foot height
        height_rew = torch.where(foot_height > 0.1, 
                                 torch.tensor(0.1, device=self.device),
                                 torch.tensor(0., device=self.device))

        height_rew += torch.clip(foot_height, min=0., max=1.)
        height_rew = height_rew.sum(dim=-1)
        # ic(foot_height, height_rew)
        return height_rew



    def _reward_obj_zvel(self):
        '''
        Reward based on the object's z-directional velocity
        Incentize the lifting behavior
        '''
        left_close_enough = torch.norm(self.left_hand_pos - self.obj_root_states[..., :3], dim=-1) < self.cfg.object.box_size[0]
        right_close_enough = torch.norm(self.right_hand_pos - self.obj_root_states[..., :3], dim=-1)< self.cfg.object.box_size[0]
        close_enough = torch.logical_and(left_close_enough, right_close_enough)

        obj_zvel = torch.clip(self.obj_root_states[..., 9], min=0.)
        self.episode_metric_sums['obj_zvel'] += obj_zvel

        return close_enough * torch.clip(obj_zvel, max=0.1) * ~(self.is_lifted) * (self.episode_length_buf > 50)

    def _reward_obj_xyvel(self):
        '''
        Reward penelizes the object's xy velocity
        '''

        obj_xyvel = torch.norm(self.obj_root_states[..., 8:10], dim=-1)
        self.episode_metric_sums['obj_xyvel'] += obj_xyvel

        return torch.clip(obj_xyvel, max=1.0)
    
    def _reward_lifted(self):
        '''
        Reward if the object lifted over the threshold
        Enforces the robot to be stable after the lifting
        '''
        self.episode_metric_sums['is_lifted'] += self.is_lifted
        return self.is_lifted * (self.episode_length_buf > 50)

    def _reward_single_foot_contact(self):
        '''
        Penalizes the single foot contact
        '''
        left_foot_contact = self.last_last_contacts[..., 0] | self.last_contacts[..., 0] | self.foot_contacts[..., 0]
        left_foot_contact *= self.left_feet_pos[..., 2] < 0.15
        right_foot_contact = self.last_last_contacts[..., 1] | self.last_contacts[..., 1] | self.foot_contacts[..., 1]
        right_foot_contact *= self.right_feet_pos[..., 2] < 0.15

        # Single foot contact or zero foot contact case
        single_contact = ~torch.logical_and(left_foot_contact, right_foot_contact)
        single_contact *= self.episode_length_buf > 10
        self.episode_metric_sums['single_foot_contact'] += single_contact

        return single_contact

    def _reward_pelvis_height(self):
        # Reward based on the pelvis' z height
        left_close_enough = torch.norm(self.left_hand_pos - self.obj_root_states[..., :3], dim=-1) < self.cfg.object.box_size[0]
        right_close_enough = torch.norm(self.right_hand_pos - self.obj_root_states[..., :3], dim=-1)< self.cfg.object.box_size[0]
        close_enough = torch.logical_and(left_close_enough, right_close_enough)

        
        self.episode_metric_sums['pelvis_height'] += self.pelvis_pos[..., 2]

        height_delta = self.pelvis_pos[..., 2] - self.highest_pelvis
        # Update highest pelvis only for pickup phase
        self.highest_pelvis= torch.where(close_enough,
            torch.maximum(self.highest_pelvis, self.pelvis_pos[..., 2]),
            self.highest_pelvis)
        height_delta = torch.clip(height_delta, 0., 10.)
        pelvis_rew = torch.tanh(30.0 * height_delta)
        
        return close_enough * pelvis_rew * (self.episode_length_buf > 50)

    def _reward_base_obj_dist(self):
        # Reward based on the base <-> object distance
        left_close_enough = torch.norm(self.left_hand_pos - self.obj_root_states[..., :3], dim=-1) < self.cfg.object.box_size[0]
        right_close_enough = torch.norm(self.right_hand_pos - self.obj_root_states[..., :3], dim=-1)< self.cfg.object.box_size[0]
        close_enough = torch.logical_and(left_close_enough, right_close_enough)

        distance = torch.norm(self.root_states[..., :3]- self.obj_root_states[..., :3], dim=-1)
        self.episode_metric_sums['base_obj_dist'] += distance
        dist_rew = torch.exp(-10 * torch.square(distance))
        # ic(dist_rew)
        return close_enough * dist_rew * (self.episode_length_buf > 50)


    '''

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    '''
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
