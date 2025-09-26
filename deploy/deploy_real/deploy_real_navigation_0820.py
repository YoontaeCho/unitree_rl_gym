import os
import torch as th
import numpy as np
import math_utils
from pathlib import Path
from typing import Union
from legged_gym import LEGGED_GYM_ROOT_DIR

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from common.command_helper_ros import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.remote_controller import RemoteController, KeyMap
from config_0820 import Config
from common.crc import CRC
from enum import Enum
from ikctrl import IKCtrl

import utils_robot as ur
import utils_locomotion as ul


class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    finish = 5
    null = 6

yaw_quat = math_utils.as_np(math_utils.yaw_quat)
quat_apply = math_utils.as_np(math_utils.quat_apply)
axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)


class GlobalClock:
    def __init__(self, node):
        self.node = node

    def get_time(self):
        return self.node.get_clock().now()


clock = None


def body_pose(
        tf_buffer,
        frame: str,
        ref_frame: str = 'pelvis',
        stamp=None,
        rot_type: str = 'axa'):
    """ --> tf does not exist """
    if stamp is None:
        stamp = rp.time.Time()
        # stamp = clock.get_time()
    try:
        # t = "ref{=pelvis}_from_frame" transform
        t = tf_buffer.lookup_transform(
            ref_frame,  # to
            frame,  # from
            stamp)
    except TransformException as ex:
        print(f'Could not transform {frame} to {ref_frame}: {ex}')
        raise

    txn = t.transform.translation
    rxn = t.transform.rotation

    xyz = np.array([txn.x, txn.y, txn.z])
    quat_wxyz = np.array([rxn.w, rxn.x, rxn.y, rxn.z])

    xyz = np.array(xyz)
    if rot_type == 'axa':
        axa = axis_angle_from_quat(quat_wxyz)
        axa = wrap_to_pi(axa)
        return (xyz, axa)
    elif rot_type == 'quat':
        return (xyz, quat_wxyz)
    raise ValueError(f"Unknown rot_type: {rot_type}")


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

from common.rotation_helper import get_gravity_orientation, transform_imu_data

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        
        # Load policy

        # smoothing
        self.prev_joint_pos_target = None
        
        # log path
        self.logpath = Path('/tmp/e2e/')
        self.logpath.mkdir(parents=True, exist_ok=True)

        # == build index map ==
        self.mot_from_lab = index_map(self.config.motor_joint, self.config.lab_joint)
        self.lab_from_mot = index_map(self.config.lab_joint, self.config.motor_joint)
        
        # num joints
        self.num_joints = len(self.config.motor_joint)
        
        # log trajectory (1000Hz)
        self.timestamp_high_freq = np.array([])
        self.q_traj = np.zeros((0, self.num_joints))
        self.dq_traj = np.zeros((0, self.num_joints))
        self.tau_traj = np.zeros((0, self.num_joints))

        # log trajectory (50Hz)
        self.timestamp_low_freq = np.array([])
        self.locomotion_observations = np.zeros((0, self.config.locomotion_obs_dim)) # TODO
        # self.eetrack_observations = np.zeros((0, self.config.eetrack_obs_dim))
        self.locomotion_vel_traj = np.zeros((0, 3))
        self.pos_command_bs = np.zeros((0,3))
        self.pos_command_b = np.zeros(3)

        # self.locomotion_actions = np.zeros((0, 15))
        self.locomotion_actions = np.zeros((0, 14))
        self.raw_joint_pos_targets = np.zeros((0, self.num_joints))
        self.joint_pos_targets = np.zeros((0, self.num_joints))

        self.current_joint_pos = np.zeros(self.num_joints)

        # counter
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self._node)

        # locomotion
        self.locomotion_robot = ur.Robot(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0_replace_with_welder.urdf')
        # self.locomotion_actmap = ul.LocomotionAction_0820(config, self.locomotion_robot)
        self.locomotion_actmap = ul.LocomotionAction_14dof(config, self.locomotion_robot)
        self.locomotion_policy = th.jit.load(config.locomotion_policy_path) # TODO
        self.locomotion_policy.eval()
        self.locomotion_obsmap = ul.LocomotionObservation_14dof(config, self.tf_buffer # TODO
        )
        self.locomotion_velocity_command = None
        self.locomotion_phase_command = None
        # self.locomotion_last_action = np.zeros(15)
        self.locomotion_last_action = np.zeros(14)

        self.stop_locomotion = False

        ########################## Navigation ##########################
        self.navigation_pos_target = np.array([1.0, 0.0, 0.0])
        self.navigation_heading_target = 0.0 # radians
        self.pos_error_bs = np.zeros((1,1))
        self.heading_error_bs = np.zeros((1,1))
        self.pelvis_to_midsole_offset_after_locomotion = {
            "x": -0.0069,
            "y": -0.0194,
            "yaw": 0.0061,
        }

        # self.pelvis_to_midsole_offset_after_locomotion = {
        #     "x":0.0,
        #     "y": -0.0094,
        #     "yaw": 0.0061,
        # }
        # self.target_midsole_quat = yaw_quat(np.array([np.sin(self.navigation_heading_target/2), 0.0, 0.0, np.cos(self.navigation_heading_target/2)]).astype(np.float32)).astype(np.float32)

        # self.offset = quat_apply(self.target_midsole_quat, 
        #             np.array(
        #                [self.pelvis_to_midsole_offset_after_locomotion["x"], 
        #                 self.pelvis_to_midsole_offset_after_locomotion["y"], 
        #                 0.0]
        #                 ).astype(np.float32)
        #             )

        # self.pelvis_pos_target = self.navigation_pos_target + self.offset
        # self.pelvis_heading_target = self.navigation_heading_target + self.pelvis_to_midsole_offset_after_locomotion["yaw"]
        
        self.navigation_command = None
        self.pelvis_pos_target = None
        self.pelvis_heading_target = None
        self.navigation_mode = False
        self.locomotion_counter = 0
        self.stop_state = False
        self.minimum_locomotion_iter = 0
        

    
        # HYPERPARAMETERS
        self.locomotion_vel_command = np.array([0., 0., 0.])

        self.SLOW_BOUND = 0.4
        self.MAX_LIN_VEL =  0.15
        self.MAX_ANG_VEL = 0.3
        self.NAV_HZ = 5
        self.ERROR_THRESHOLD = 0.04  # m
        self.NUM_AVG = 30

        ########################## Navigation ##########################

        self.target_vec_b_log = np.zeros((0, 4))
        self.target_vec_bs = np.zeros((0, 4)) # 4 dim
        self.navigation_obsmap = ul.NavigationObservation(config, self.tf_buffer)
        self.navigation_actmap = ul.NavigationAction(config, self.locomotion_robot)
        print(config.navigation_policy_path)
        self.navigation_policy = th.jit.load(config.navigation_policy_path) # TODO
        self.navigation_policy.eval()
        self.naviation_last_action = np.zeros(3, dtype=np.float32)
        self.locomotion_vel_commands = np.zeros((0, 3))

        self.qj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_actions, dtype=np.float32)


        act_joint = config.ik_joint
        self.ikctrl = IKCtrl('../../resources/robots/g1_description/g1_29dof_rev_1_0_ver4_camera_mount_v2.urdf',
                             config.ik_joint,
                             frame='end_effector')
        self.pin_from_mot = np.zeros(29, dtype=np.int32) # FIXME(ycho): hardcoded
        self.mot_from_pin = np.zeros(43, dtype=np.int32) # FIXME(ycho): hardcoded
        self.mot_from_act = np.zeros(len(act_joint), dtype=np.int32) # FIXME(ycho): hardcoded
        for i_mot, j in enumerate( self.config.motor_joint ):
            i_pin = (self.ikctrl.robot.index(j) - 1)
            self.pin_from_mot[i_mot] = i_pin
            self.mot_from_pin[i_pin] = i_mot
            if j in act_joint:
                i_act = act_joint.index(j)
                self.mot_from_act[i_act] = i_mot
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = LowCmdHG()
            self.low_state = LowStateHG()

            self.lowcmd_publisher_ = self._node.create_publisher(LowCmdHG, 'lowcmd', 10)
            self.lowstate_subscriber = self._node.create_subscription(
                LowStateHG, 'lowstate', self.LowStateHgHandler, 10)
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

        elif config.msg_type == "go":
            raise ValueError(f"{config.msg_type} is not implemented yet.")

        else:
            raise ValueError("Invalid msg_type")

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        print("Waiting for the robot to be ready...")
        self.mode = Mode.wait
        

        self.is_eetrack_first_iter = True
        self.task = "locomotion"
        self.zero_phase = False

        self.sitting = False
        self._mode_change = True
        self._terminate = False
        self._timer = self._node.create_timer(self.config.control_dt, self.run_wrapper)
        self.stop_time = None
        try:
            rp.spin(self._node)
        except KeyboardInterrupt:
            self.log_metrics_and_trajectories()
            print("Log saved.")
        finally:
            self._node.destroy_timer(self._timer)
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self._node.destroy_node()
            rp.shutdown()
            print("Exit")

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)
        
        # log trajectory
        # joint order = lab joint config order
        timestamp_high_freq = clock.get_time().nanoseconds / 1e9
        curr_q = np.zeros(self.num_joints)
        curr_dq = np.zeros(self.num_joints)
        curr_tau = np.zeros(self.num_joints)
        curr_q[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].q for mot_idx in range(self.num_joints)]
        curr_dq[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].dq for mot_idx in range(self.num_joints)]
        curr_tau[self.lab_from_mot] = [self.low_state.motor_state[mot_idx].tau_est for mot_idx in range(self.num_joints)]
        self.timestamp_high_freq = np.append(self.timestamp_high_freq, timestamp_high_freq)
        self.q_traj = np.vstack((self.q_traj, curr_q))
        self.dq_traj = np.vstack((self.dq_traj, curr_dq))
        self.tau_traj = np.vstack((self.tau_traj, curr_tau))

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.mode_machine = self.mode_machine_
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.publish(cmd)

    def zero_torque_state(self):
        if self.remote_controller.button[KeyMap.start] == 1:
            self._mode_change = True
            self.mode = Mode.damping
        else:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)

    def prepare_default_pos(self):
        # move time 2s
        total_time = 2
        self.counter = 0
        self._num_step = int(total_time / self.config.control_dt)

        self._kps = [float(kp) for kp in self.config.kps]
        self._kds = [float(kd) for kd in self.config.kds]

        self._init_dof_pos = np.zeros(self.num_joints)
        for mot_idx in range(self.num_joints):
            self._init_dof_pos[mot_idx] = self.low_state.motor_state[mot_idx].q

    def move_to_default_pos(self):
        # move to default pos with smoothing
        if self.counter < self._num_step:
            alpha = self.counter / self._num_step
            for motor_idx in range(self.num_joints):
                target_pos = self.config.default_angles[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].q = (self._init_dof_pos[motor_idx] * (1 - alpha) + target_pos * alpha)
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            self.counter += 1
        else:
            self._mode_change = True
            self.mode = Mode.policy

    def default_pos_state(self):
        if self.remote_controller.button[KeyMap.Y] != 1:
            for motor_idx in range(self.num_joints):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.locomotion_motor_joint_offsets[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            print("Sending cmd to robot")
            print()
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy

        
    def terminate_by_pelvis_condition(self, xyz, quat, limit_euler_angle=[0.9, 1.0]) -> bool:
        """
        limit euler angle : roll 51.57', pitch 57.3'.
        """
        euler = math_utils.wrap_to_pi(
            th.stack(math_utils.euler_xyz_from_quat(th.as_tensor(quat.reshape(1, quat.shape[0]))), dim=-1)
        )
        out_of_limit = th.logical_or(
            th.abs(euler[..., 0]) > limit_euler_angle[0],
            th.abs(euler[..., 1]) > limit_euler_angle[1],
        )
        if out_of_limit.item() :
            print("Terminated by pelvis condition.")
            print(f"euler: {euler}")
        return out_of_limit.item()
    
    def run_policy(self):
        # If the button A is pressed, then finish the policy.
        if self.remote_controller.button[KeyMap.A] == 1:
            self._mode_change = True
            self.mode = Mode.finish
            return

        self.counter += 1

        world_from_pelvis = body_pose(
            self.tf_buffer,
            'pelvis',
            'world',
            rot_type='quat'
        )
        xyz, quat_wxyz = world_from_pelvis
        root_state_w = np.zeros(7)
        root_state_w[0:3] = xyz
        root_state_w[3:7] = quat_wxyz


        if True:
            nav_target_pos , nav_target_axa = body_pose(
                self.tf_buffer,
                'nav_target',
                'world',
                rot_type='axa',
                stamp=rp.time.Time()
            )

            self.pelvis_pos_target = nav_target_pos
            self.pelvis_heading_target = nav_target_axa[-1]
        else:
            self.pelvis_pos_target = np.array([1., 0., 0.])
            self.pelvis_heading_target = 0.


        # # # ADD offset between pelvis <> midsole after locomotion.
        # self.offset = quat_apply(
        #             yaw_quat(nav_target_quat).astype(np.float32), 
        #             np.array(
        #                [self.pelvis_to_midsole_offset_after_locomotion["x"], 
        #                 self.pelvis_to_midsole_offset_after_locomotion["y"], 
        #                 0.0]
        #                 ).astype(np.float32)
        #             )
        
        # # print(self.offset)
        
        # self.pelvis_pos_target += self.offset
        # self.pelvis_heading_target += self.pelvis_to_midsole_offset_after_locomotion["yaw"]
        
        # Add termination condition.
        if self.terminate_by_pelvis_condition(xyz, quat_wxyz):
            raise ValueError("Terminated by pelvis condition.")
        

        phase = (self.counter * 0.02) % 1.0 / 1.0
        
        
        ###################### Compute state ######################
        forward_w = quat_apply(quat_wxyz.astype(np.float32), np.array([1., 0., 0.]).astype(np.float32))
        heading_w = np.arctan2(forward_w[1], forward_w[0])
        heading_error = wrap_to_pi(np.array([self.pelvis_heading_target]).astype(np.float32) - np.array([heading_w]).astype(np.float32))

        target_vec = self.pelvis_pos_target - xyz
        target_vec[2] = 0.0
        self.pos_command_b = quat_rotate_inverse(yaw_quat(quat_wxyz).astype(np.float32), target_vec.astype(np.float32))
        print(f"pos command b : {self.pos_command_b}")
        print(f"heading_error : {heading_error}")
        self.pos_error_bs = np.vstack((self.pos_error_bs, np.linalg.norm(self.pos_command_b[:2]).reshape((1,1))))
        self.heading_error_bs = np.vstack((self.heading_error_bs, heading_error.reshape(1,1)))


        ############################################ SWITCH MODE FROM LOCOMOTION TO NAVIVATION ############################################
        if self.remote_controller.button[KeyMap.X] == 1:
            self.navigation_mode = True

        ############################################ LOCOMOTION ############################################
        if not self.navigation_mode:
            if np.any(
                np.abs(np.array([self.remote_controller.ly, self.remote_controller.lx, self.remote_controller.rx])) > 0.
            ) or self.minimum_locomotion_iter != 0:

                if self.minimum_locomotion_iter == 0:
                    self.minimum_locomotion_iter = 100
                self.minimum_locomotion_iter -= 1

                v_x = np.clip(self.remote_controller.ly, -0.3, 0.1)
                v_y = np.clip(self.remote_controller.lx * -1, -0.15, 0.1)
                v_z = np.clip(self.remote_controller.rx * -1, -0.3, 0.3)

                self.locomotion_vel_command[0] = v_x
                self.locomotion_vel_command[1] = v_y
                self.locomotion_vel_command[2] = v_z

                phase = (self.locomotion_counter * 0.02) % 1.0 / 1.0
                self.locomotion_counter += 1

                print("LOCOMOTION CMD IS WORKING")
                print(self.locomotion_vel_command)
            else:
                self.locomotion_vel_command = np.zeros(3)
                phase = self.locomotion_counter = 0
        
        ############################################ NAVIGATION ############################################
        else:
            ###################### Hand design navigation ######################
            if True:
                if self.pos_error_bs.shape[0] > self.NUM_AVG:
                    # print("pos command mean : ", np.mean(self.pos_error_bs[-self.NUM_AVG:, :].reshape((self.NUM_AVG,))))
                    # print("heading error mean : ", np.mean(np.abs(self.heading_error_bs[-self.NUM_AVG:, :]).reshape((self.NUM_AVG,))))
                    if np.mean(self.pos_error_bs[-self.NUM_AVG:, :].reshape((self.NUM_AVG,))) < self.ERROR_THRESHOLD \
                        and np.mean(np.abs(self.heading_error_bs[-self.NUM_AVG:, :]).reshape((self.NUM_AVG,))) < 0.05:
                        if self.stop_locomotion is not True :
                            self.stop_time = (self.counter-400) * self.config.control_dt
                        self.stop_locomotion = True
                    
                # # To stop when being initialized
                # if self.counter < 200 * 2 :
                #     self.locomotion_vel_command = np.array([0., 0., 0.])
                #     phase = 0.0
                if self.counter % 10 == 0:
                    # self.locomotion_vel_command[:2] = np.clip(np.sign(self.pos_command_b[:2]) * MAX_LIN_VEL * np.sqrt(np.abs(self.pos_command_b[:2] / SLOW_BOUND)), -MAX_LIN_VEL, MAX_LIN_VEL)
                    # X >= 0
                    if self.pos_command_b[0] >= 0:
                        self.locomotion_vel_command[0] = np.clip(0.1 * np.sqrt(np.abs(self.pos_command_b[0] / 0.8)), 0., 0.1)
                    # X < 0
                    if self.pos_command_b[0] < 0:
                        self.locomotion_vel_command[0] = np.clip(-0.3 * np.sqrt(np.abs(self.pos_command_b[0] / 0.4)), -0.3, 0.)
                    # Y >= 0
                    if self.pos_command_b[1] >= 0:
                        self.locomotion_vel_command[1] = np.clip(0.1 * np.sqrt(np.abs(self.pos_command_b[1] / 0.3)), 0., 0.1)
                    # Y < 0
                    if self.pos_command_b[1] < 0:
                        self.locomotion_vel_command[1] = np.clip(-0.1 * np.sqrt(np.abs(self.pos_command_b[1] / 0.3)), -0.1, 0.)
                    
                    self.locomotion_vel_command[2] = np.clip(np.sign(heading_error) * 0.2 * np.sqrt(np.abs(heading_error / 0.4)), -0.2, 0.2)

                if self.stop_locomotion:
                    self.locomotion_vel_command = np.array([0., 0., 0.])
                    phase = 0.0
            ###################### RL Trained navigation ######################
            else:
                mid_sole_pos_w, mid_sole_quat_w = body_pose(
                    self.tf_buffer,
                    'mid_sole_link',
                    'world',
                    rot_type='quat'
                )
                if self.navigation_command == None:
                    self.navigation_command = ul.NavigationCommand(
                        x = self.pelvis_pos_target[0],
                        y = self.pelvis_pos_target[1],
                        heading = self.pelvis_heading_target,
                    )
                if True: # Mid sole target
                    self.target_vec_b = self.navigation_command.command(mid_sole_pos_w, mid_sole_quat_w)
                else: # Plvis Target
                    self.target_vec_b = self.navigation_command.command(xyz, quat_wxyz)
                self.target_vec_b_log = np.vstack([self.target_vec_b_log, self.target_vec_b])
                if self.counter < 200 * 2 :
                    self.locomotion_vel_command = np.array([0., 0., 0.])
                    phase = 0.0
                elif self.counter % 10 == 0:
                    if True:
                        POS_WINDOW = 10
                        n = min(len(self.target_vec_b_log), POS_WINDOW)
                        if n > 0:
                            self.target_vec_b = np.mean(self.target_vec_b_log[-n:], axis=0)  # shape (4,)
                    self.target_vec_bs = np.vstack([self.target_vec_bs, self.target_vec_b])

                    self.nav_obs = self.navigation_obsmap(
                        self.low_state,
                        self.target_vec_b, # 4 dim
                        phase,
                        last_action=self.naviation_last_action
                    )
                    obs_tensor = th.from_numpy(self.nav_obs).unsqueeze(0)
                    obs_tensor = obs_tensor.detach().clone().float()

                    self.navigation_action = self.navigation_policy(obs_tensor).detach().numpy().squeeze()
                    self.locomotion_vel_command = self.navigation_actmap(self.navigation_action)
                    self.naviation_last_action = self.locomotion_vel_command
                    self.locomotion_vel_commands = np.vstack([self.locomotion_vel_commands, self.locomotion_vel_command])

                if self.counter > 200 * 2 and np.linalg.norm(self.locomotion_vel_command) < 0.05:
                    self.stop_locomotion = True
                
                if self.stop_locomotion:
                    self.locomotion_vel_command = np.array([0., 0., 0.])
                    phase = 0.0

        print(f"locomotion vel command : {self.locomotion_vel_command}")
        print()
        print()
        
        # phase = 0.0

        self.obs = self.locomotion_obsmap(self.low_state, self.locomotion_vel_command, phase, last_action=self.locomotion_last_action)
        obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone().float()
        self.locomotion_action = self.locomotion_policy(obs_tensor).detach().numpy().squeeze()
        self.locomotion_last_action = self.locomotion_action

        # target_dof_pos : motor joint ordered
        self.locomotion_target_dof_pos = self.locomotion_actmap(self.locomotion_action)
        target_dof_pos = self.locomotion_target_dof_pos

        if True:
            if self.prev_joint_pos_target is not None:
                if self.counter < 100:
                    target_dof_pos = self.config.initial_smoothing * target_dof_pos + \
                                    (1-self.config.initial_smoothing) * self.prev_joint_pos_target
                else:
                    target_dof_pos = self.config.later_smoothing * target_dof_pos + \
                                    (1-self.config.later_smoothing) * self.prev_joint_pos_target

            self.prev_joint_pos_target = target_dof_pos

        # self.print_locomotion_status()
        for mot_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[mot_idx].q =  float(target_dof_pos[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].dq = 0.0
            self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(self.config.kps[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(self.config.kds[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].tau = 0.0

        
        # observation dumping
        self.dump_observations_and_joint_pos_target(target_dof_pos)

         
        # send the command
        self.send_cmd(self.low_cmd)

    def dump_observations_and_joint_pos_target(self, target_dof_pos):
        # log timestamp
        timestamp_low_freq = clock.get_time().nanoseconds / 1e9
        self.timestamp_low_freq = np.append(self.timestamp_low_freq, timestamp_low_freq)
        
        self.locomotion_observations = np.vstack((self.locomotion_observations, self.obs))
        self.locomotion_actions = np.vstack((self.locomotion_actions, self.locomotion_action))
        self.locomotion_vel_traj = np.vstack((self.locomotion_vel_traj, self.locomotion_vel_command))
        self.pos_command_bs = np.vstack((self.pos_command_bs, self.pos_command_b))

        
            
    def log_metrics_and_trajectories(self):     
        # Calculate pos diff & torque diff metrics
        pos_diff = np.average(np.abs(self.q_traj[1:] - self.q_traj[:-1]))
        torque_diff = np.average(np.abs(self.tau_traj[1:] - self.tau_traj[:-1]))
        print("\n--------------------------METRICS--------------------------")
        print("total_time", self.counter * self.config.control_dt)
        print("pos_diff", pos_diff)
        print("torque_diff", torque_diff)
        print("stop time ", self.stop_time)
        print("----------------------------------------------------------")
        
        # Normalize timestamps with respect to the first timestamp_high_freq
        self.timestamp_low_freq = self.timestamp_low_freq - self.timestamp_high_freq[0]
        self.timestamp_high_freq = self.timestamp_high_freq - self.timestamp_high_freq[0]
        
        # Save metrics and trajectories
        metrics = {
            "pos_diff": pos_diff,
            "torque_diff": torque_diff
        }
        trajectories = {
            "timestamp_high_freq": self.timestamp_high_freq,
            "q_traj": self.q_traj,
            "dq_traj": self.dq_traj,
            "tau_traj": self.tau_traj,
            "timestamp_low_freq": self.timestamp_low_freq,
            # "sit_observations": self.sit_observations,
            "locomotion_observations": self.locomotion_observations,
            # "eetrack_observations": self.eetrack_observations,
            # "sit_actions": self.sit_actions,
            "locomotion_actions": self.locomotion_actions,
            # "eetrack_actions": self.eetrack_actions,
            "joint_pos_targets" : self.joint_pos_targets,
            "locomotion_vel_traj" : self.locomotion_vel_traj,
            "pos_errors" : self.pos_error_bs,
            "heading_errors" : self.heading_error_bs,
            "pos_command_bs" : self.pos_command_bs
        }
        log_data = {
            "metrics": metrics,
            "trajectories": trajectories
        }

        # Save the log with experiment name
        # sit_model = os.path.basename(self.config.sit_policy_path).split('.')[0]
        # eetrack_model = os.path.basename(self.config.locomotion_policy_path).split('.')[0]
        locomotion_model = os.path.basename(self.config.locomotion_policy_path).split('.')[0]
        timestamp = clock.get_time().nanoseconds / 1e9
        
        file_name = f"{self.logpath}/log_{locomotion_model}_{str(timestamp).split('.')[0]}.npy"
        np.save(file_name, log_data)
        print(f"Log saved at {file_name}")
        
        # totally terminate
        self._mode_change = True
        self.mode = Mode.null
        

    def run_wrapper(self):
        if self.mode == Mode.wait:
            print("Waiting for the robot to be ready...")
            if self.low_state.crc != 0:
                self.mode = Mode.zero_torque
                self.low_cmd.mode_machine = self.mode_machine_
                print("Successfully connected to the robot.")
        elif self.mode == Mode.zero_torque:
            if self._mode_change:
                print("Enter zero torque state.")
                print("Waiting for the start signal...")
                self._mode_change = False
            self.zero_torque_state()
        elif self.mode == Mode.default_pos:
            if self._mode_change:
                print("Moving to default pos.")
                self._mode_change = False
                self.prepare_default_pos()
            self.move_to_default_pos()
        elif self.mode == Mode.damping:
            if self._mode_change:
                print("Enter default pos state.")
                print("Waiting for the Button Y signal...")
                self._mode_change = False
            self.default_pos_state()
        elif self.mode == Mode.policy:
            if self._mode_change:
                print("Run Policy.\n")
                print("--------------[ Basic Guidelines ]---------------")
                print("[Navigation] Press Button {X} to switch mode into **Navigation**.")
                print("-------------------------------------------------")
                print("[Exit] Press Button {A} to finish.")
                print("-------------------------------------------------")

                self._mode_change = False
                self.counter = 0
            self.run_policy()
        elif self.mode == Mode.finish:
            if self._mode_change:
                print("Finish.")
                self._mode_change = False
            self.log_metrics_and_trajectories()
        elif self.mode == Mode.null:
            self._terminate = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="config file name in the configs folder",
        default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    controller = Controller(config)
