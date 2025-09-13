import os
import torch as th
import numpy as np
import math_utils
from pathlib import Path
from typing import Union
from legged_gym import LEGGED_GYM_ROOT_DIR
from scipy.spatial.transform import Rotation as R

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from common.command_helper_ros import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.remote_controller import RemoteController, KeyMap
from config_e2e import E2EConfig as Config
from common.crc import CRC
from enum import Enum
from ikctrl import IKCtrl
from std_msgs.msg import Float32

import utils_stage as us
# import utils_eetrack as ue
import utils_eetrack_tag as ue
import utils_robot as ur
import utils_locomotion as ul

# For trajopt action client
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from pyroki_ros.action import TrajOpt

class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    finish = 5
    null = 6


axis_angle_from_quat = math_utils.as_np(math_utils.axis_angle_from_quat)
quat_conjugate = math_utils.as_np(math_utils.quat_conjugate)
quat_mul = math_utils.as_np(math_utils.quat_mul)
quat_rotate = math_utils.as_np(math_utils.quat_rotate)
quat_rotate_inverse = math_utils.as_np(math_utils.quat_rotate_inverse)
wrap_to_pi = math_utils.as_np(math_utils.wrap_to_pi)
combine_frame_transforms = math_utils.as_np(
    math_utils.combine_frame_transforms)
quat_from_euler_xyz = math_utils.as_np(math_utils.quat_from_euler_xyz)
euler_xyz_from_quat = math_utils.as_np(math_utils.euler_xyz_from_quat)
yaw_quat = math_utils.as_np(math_utils.yaw_quat)
matrix_from_quat = math_utils.as_np(math_utils.matrix_from_quat)
subtract_frame_transforms = math_utils.as_np(math_utils.subtract_frame_transforms)


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
        rot_type: str = 'axa',
        do_not_raise: bool = False):
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
        if do_not_raise:
            t = TransformStamped()
        else:
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


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        
        # Load policy
        print(config.policy_path)
        self.sit_policy = th.jit.load(config.sit_policy_path) # FIXME sit_policy_path
        self.sit_policy.eval()

        # smoothing
        self.prev_joint_pos_target = None
        
        # log path
        self.logpath = Path('/home/unitree/logs/e2e/')
        self.logpath.mkdir(parents=True, exist_ok=True)

        # == build index map ==
        self.mot_from_lab = index_map(self.config.motor_joint, self.config.lab_joint)
        self.lab_from_mot = index_map(self.config.lab_joint, self.config.motor_joint)
        self.mot_from_lower = index_map(self.config.motor_joint, self.config.lower_joint)
        
        self.mot_from_lab_eetrack = index_map(self.config.motor_joint, self.config.lab_joint_eetrack)
        self.lab_from_mot_eetrack = index_map(self.config.lab_joint_eetrack, self.config.motor_joint)
        
        # num joints
        self.num_joints = len(self.config.motor_joint)
        
        # log trajectory (1000Hz)
        self.timestamp_high_freq = np.array([])
        self.q_traj = np.zeros((0, self.num_joints))
        self.dq_traj = np.zeros((0, self.num_joints))
        self.tau_traj = np.zeros((0, self.num_joints))

        # log trajectory (50Hz)
        self.timestamp_low_freq = np.array([])
        self.sit_observations = np.zeros((0, self.config.sit_obs_dim))
        self.locomotion_observations = np.zeros((0, self.config.locomotion_obs_dim))

        self.sit_actions = np.zeros((0, self.num_joints))
        self.locomotion_actions = np.zeros((0, self.num_joints))
        self.raw_joint_pos_targets = np.zeros((0, self.num_joints))
        self.joint_pos_targets = np.zeros((0, self.num_joints))
        self.locomotion_cmd_traj = np.zeros((0, 3))

        self.current_joint_pos = np.zeros(self.num_joints)

        self.is_initial_goals = np.zeros((0,), dtype=bool)
        self.is_initial_eetracks = np.zeros((0,), dtype=bool)
        self.is_go_starts = np.zeros((0,), dtype=bool)
        self.target_poses_w = np.zeros((0, 7))
        self.target_poses_b = np.zeros((0, 7))
        self.ee_poses_w = np.zeros((0, 7))
        self.ee_poses_b = np.zeros((0, 7))

        self.task_traj = []

        # counter
        self.counter = 0
        self.eetrack_initial_counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self._node)


        self.imu_quat = np.zeros((0, 4))
        self.slam_quat = np.zeros((0, 4))

        ### Mapping helpers.
        self.sit_obsmap = us.SitObservation(config, self.tf_buffer)
        
        self.sit_robot = ur.Robot(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0_ver4_camera_mount_v2.urdf')
        self.sit_actmap = us.SitActionVer2(config, self.sit_robot)
        self.vhcommand = us.VelocityHeightCommandV2(config)
        self.target_height = 0.7

        self.sit_kps = np.array(self.config.sit_kps).astype(np.float32)
        self.sit_kds = np.array(self.config.sit_kds).astype(np.float32)
        self.delta_kds = 0.0
        self.last_sit_dof_pos = np.zeros(29, dtype=np.float32)

        # locomotion
        self.locomotion_robot = ur.Robot(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0_replace_with_welder.urdf')
        self.locomotion_actmap = ul.LocomotionAction(config, self.locomotion_robot)
        self.locomotion_policy = th.jit.load(config.locomotion_policy_path) # TODO
        self.locomotion_policy.eval()
        self.locomotion_obsmap = ul.LocomotionObservation(config, self.tf_buffer # TODO
        )
        self.locomotion_velocity_command = None
        self.locomotion_phase_command = None
        self.locomotion_last_action = np.zeros(29)
        self.zero_phase = False

        self.navigation_obsmap = ul.NavigationObservation(config, self.tf_buffer)
        self.navigation_actmap = ul.NavigationAction(config, self.locomotion_robot)
        self.navigation_policy = th.jit.load(config.navigation_policy_path) # TODO
        self.navigation_policy.eval()
        self.naviation_last_action = np.zeros(3, dtype=np.float32)

        self.navigation_target_location = np.array([-0.5, 0.5, 0., 0.])
        self.locomotion_vel_commands = np.zeros((0, 3))
        self.navigation_counter = 0
        self.navigation_command = None

        self.target_vec_bs = np.zeros((0, 4)) # 4 dim
        self.navigation_target_cmds = np.zeros((0, 3)) # 2 dim

        # eetrack
        self.eetrack_command = None

        self.act_joint = config.ik_joint
        self.ikctrl = IKCtrl('../../resources/robots/g1_description/g1_29dof_rev_1_0_ver4_camera_mount_v4.urdf',
                             self.act_joint,
                             frame='end_effector')
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit
        
        # == build index map ==
        self.pin_from_mot = np.zeros(29, dtype=np.int32) # FIXME(ycho): hardcoded
        self.mot_from_pin = np.zeros(43, dtype=np.int32) # FIXME(ycho): hardcoded
        self.mot_from_act = np.zeros(len(self.act_joint), dtype=np.int32) # FIXME(ycho): hardcoded
        for i_mot, j in enumerate( self.config.motor_joint):
            i_pin = (self.ikctrl.robot.index(j) - 1)
            self.pin_from_mot[i_mot] = i_pin
            self.mot_from_pin[i_pin] = i_mot
            if j in self.act_joint:
                i_act = self.act_joint.index(j)
                self.mot_from_act[i_act] = i_mot

        q_mot = np.array(config.default_angles)
        q_pin = np.zeros_like(self.ikctrl.cfg.q)
        q_pin[self.pin_from_mot] = q_mot

        controller_joints_name = ['waist_roll_joint', 'waist_yaw_joint', 'waist_pitch_joint']
        self.waist_res_q = np.zeros(len(controller_joints_name), dtype=np.float32)
        self.mot_from_controller = index_map(self.config.motor_joint, controller_joints_name)
        self.joint_name_to_idx = {name: i for i, name in enumerate(controller_joints_name)}

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
        
        # self.welding_vel = 0.01
        # self.welding_vel_subscriber = self._node.create_subscription(
        #     Float32, 'weldingvel', self.WeldingVel, 10)
        self.is_go_start = False

        self.weld_dx = 0.0
        self.weld_dy = 0.0
        self.weld_dz = 0.0
        self.weld_dpitch = 0.0
        self.weld_dyaw = 0.0

        # self._trajopt_action_client = ActionClient(
        #     self._node,
        #     TrajOpt,
        #     'trajopt',
        # )
        # self.trajopt_joint_traj = None
        # self.trajopt_success = False
        # self._trajopt_action_client.wait_for_server()

        self.mode = Mode.wait

        self.is_eetrack_first_iter = True
        self.task = "sit"

        self._mode_change = True
        self._terminate = False
        self._timer = self._node.create_timer(self.config.control_dt, self.run_wrapper)
        
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

    # def WeldingVel(self, msg: Float32):
    #     self.

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.mode_machine = self.mode_machine_
        cmd.crc = CRC().Crc(cmd)
        return
        self.lowcmd_publisher_.publish(cmd)

    def zero_torque_state(self):
        if self.remote_controller.button[KeyMap.start] == 1:
            self._mode_change = True
            self.mode = Mode.damping
        else:
            pass
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)

    def prepare_default_pos(self):
        # move time 2s
        total_time = 2
        self.counter = 0
        self._num_step = int(total_time / self.config.control_dt)

        self._kps = [float(kp) for kp in self.config.sit_kps]
        self._kds = [float(kd) for kd in self.config.sit_kds]

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
        if self.remote_controller.button[KeyMap.F2] != 1: # F2 is F3 button in controller
            for motor_idx in range(self.num_joints):
                self.low_cmd.motor_cmd[motor_idx].q = 0.0#self.config.locomotion_motor_joint_offsets[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = 0.0#self.config.locomotion_kps[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].kd = 0.0#self.config.locomotion_kds[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy

    def publish_hand_target(self):
        t = TransformStamped()

        # Format header
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'target'

        # Populate translation
        t.transform.translation.x = float(self.target_pose_w[0])
        t.transform.translation.y = float(self.target_pose_w[1])
        t.transform.translation.z = float(self.target_pose_w[2])

        # Set world_from_pelvis quaternion based on IMU state
        qw, qx, qy, qz = [float(x) for x in self.target_pose_w[3:7]]
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)
    
    def publish_navigation_target(self, pos, quat):
        t = TransformStamped()

        # Format header
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'navigation_target'

        # Populate translation
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])

        # Set world_from_pelvis quaternion based on IMU state
        qw, qx, qy, qz = [float(x) for x in quat]
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)


        
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
    
    def print_sit_status(self): 
        height_error = self.vhcommand.pelvis_height_w - self.sit_obsmap._pelvis_height()[0]
        print(f"Vel command : {self.vhcommand.pelvis_lin_vel_z_w}")
        print(f"Height error : {height_error}")


    def send_ee_traj(self, welding_xyz_wxyz_path):
        self._node.get_logger().info('Start sending end effector trajectories...')

        stamp = self._node.get_clock().now().to_msg()
        frame_id = "world"

        goal_msg = TrajOpt.Goal()
        goal_msg.ee_traj.header.stamp = stamp
        goal_msg.ee_traj.header.frame_id = frame_id

        for xyz_wxyz in welding_xyz_wxyz_path:
            pose_stamped = PoseStamped()

            pose_stamped.header.stamp = stamp
            pose_stamped.header.frame_id = frame_id

            pose_stamped.pose.position.x = xyz_wxyz[0]
            pose_stamped.pose.position.y = xyz_wxyz[1]
            pose_stamped.pose.position.z = xyz_wxyz[2]

            pose_stamped.pose.orientation.w = xyz_wxyz[3]
            pose_stamped.pose.orientation.x = xyz_wxyz[4]
            pose_stamped.pose.orientation.y = xyz_wxyz[5]
            pose_stamped.pose.orientation.z = xyz_wxyz[6]

            goal_msg.ee_traj.poses.append(pose_stamped)

        self._node.get_logger().info("Sending goal end effector trajectories...")
        self._send_goal_future = self._trajopt_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self._node.get_logger().info('Goal rejected :(')
            return

        self._node.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()

        self._get_result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        result = future.result().result

        self.trajopt_joint_traj = result.joint_traj
        self.trajopt_success = result.success
        error_message = result.error_message

        self._node.get_logger().info(f'Success: {self.trajopt_success}')


    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        percent = feedback_msg.percent_complete
        self._node.get_logger().info(f'In progress: {percent:.1f}')
    

    def run_policy(self):
        # If the button A is pressed, then finish the policy.
        if self.remote_controller.button[KeyMap.A] == 1:
            self._mode_change = True
            self.mode = Mode.finish
            return
        
        if self.remote_controller.button[KeyMap.X] == 1:
            self.task = "locomotion"
            # self.counter = 0
            print("Task changed to locomotion.")

        if self.remote_controller.button[KeyMap.Y] == 1:
            self.task = "sit"
            self.sit_kps = np.array(self.config.sit_kps).astype(np.float32)
            self.sit_kds = np.array(self.config.sit_kds).astype(np.float32)
            print("Task changed to sit.")


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
        
        if self.remote_controller.button[KeyMap.B] == 1:
            self.task = "eetrack"
            start_ee_pos, start_ee_quat = body_pose(
                self.tf_buffer,
                'end_effector',
                'world',
                rot_type='quat'   
            )
            self.eetrack_command = ue.eetrack(
                th.from_numpy(root_state_w)[None],
                self.tf_buffer,
                clock, to_start=False, 
                start_ee_pos=start_ee_pos,
                  start_ee_quat=start_ee_quat)
            
            # data = dict(np.load("test_welding_path.npz"))
            # data["pos"][:,2] += 0.01
            # welding_xyz_wxyz_path = np.concatenate([data["pos"], data["wxyz"]], axis=-1).tolist()
            # self.send_ee_traj(welding_xyz_wxyz_path)
            
            # Restore from previous dx, dy, dz, dpitch
            start_T = matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_start_quat_w))
            end_T = matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_end_quat_w))
            self.eetrack_command.eetrack_start_w += self.weld_dx*start_T[:3,0]
            self.eetrack_command.eetrack_start_w += self.weld_dy*start_T[:3,1]
            self.eetrack_command.eetrack_start_w += self.weld_dz*start_T[:3,2]
            self.eetrack_command.eetrack_end_w += self.weld_dx*end_T[:3,0]
            self.eetrack_command.eetrack_end_w += self.weld_dy*end_T[:3,1]
            self.eetrack_command.eetrack_end_w += self.weld_dz*end_T[:3,2]
            eetrack_start_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_start_quat_w[None])
            eetrack_end_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_end_quat_w[None])
            eetrack_start_euler_w[1] += self.weld_dpitch
            eetrack_end_euler_w[1] += self.weld_dpitch
            # eetrack_start_euler_w[2] += self.weld_dyaw
            # eetrack_end_euler_w[2] += self.weld_dyaw
            self.eetrack_command.eetrack_start_quat_w = quat_from_euler_xyz(*eetrack_start_euler_w)[0]
            self.eetrack_command.eetrack_end_quat_w = quat_from_euler_xyz(*eetrack_end_euler_w)[0]
            self.eetrack_command.create_eetrack()
            self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()

            self.sit_kps[-7:] = self.config.eetrack_right_arm_kps
            self.sit_kds[-7:] = self.config.eetrack_right_arm_kds
            self.sit_kds[-7:] += self.delta_kds

            self.start_ee_pos, self.start_ee_quat = body_pose(
                self.tf_buffer,
                'end_effector',
                'world',
                rot_type='quat'   
            )
            self.is_go_start = False
            print("Task changed to eetrack.")
            ########## To disable sitting ##########
            # for i in range(len(self.config.sit_kps[:-7])):
            #     self.config.sit_kps[i] = 0.0
            #     self.config.sit_kps[i] = 0.0
            # self.wasit_target_dof_pos = self.q_traj[-1, self.mot_from_controller]
        
        # if self.remote_controller.button[KeyMap.X] == 1:
        #     self.task = "waist_rot"

        mid_sole_pos_w, mid_sole_quat_w = body_pose(
            self.tf_buffer,
            'mid_sole_link',
            'world',
            rot_type='quat'
        )


        imu_quat_wxyz = np.array([
            float(x) for x in 
            self.low_state.imu_state.quaternion
        ])

        self.slam_quat = np.vstack([self.slam_quat, quat_wxyz])
        self.imu_quat = np.vstack([self.imu_quat, imu_quat_wxyz])
        
        # Add termination condition.
        if self.terminate_by_pelvis_condition(xyz, quat_wxyz):
            raise ValueError("Terminated by pelvis condition.")

        target_tau = np.zeros(29, dtype=np.float32)
        
        if self.task == "locomotion":

            phase = (self.counter * 0.02) % 0.8 / 0.8
            if not self.zero_phase:
                self.locomotion_phase_command = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])
            else:
                self.locomotion_phase_command = self.locomotion_phase_command
            if self.remote_controller.button[KeyMap.down] == "X":
                self.zero_phase == True
                print("PHASE IS KEPT SAME")

            # target_vec = np.zeros(3)
            # x, y, z = xyz
            # target_vec[0] = self.navigation_target_location[0] - x
            # target_vec[1] = self.navigation_target_location[1] - y

            # self.target_vec_b = quat_rotate_inverse(yaw_quat(quat_wxyz), target_vec)
            # self.target_vec_b = np.concatenate([self.target_vec_b, [0.]], axis=0)

            # print("xyz : ",xyz)
            # print("target_vec : ", self.navigation_target_location, )
            # print("target_vec_b : ", self.target_vec_b, quat_wxyz)
            # print()

            goal_base_pos, goal_base_axa = body_pose(
                self.tf_buffer,
                "goal_base_pose",
                "world",
                rot_type='axa',
                do_not_raise=True,
            )

            navigation_target_pos, navigation_target_axa = body_pose(
                self.tf_buffer,
                "navigation_target",
                "world",
                rot_type='axa',
                do_not_raise=True,
            )
            self.navigation_target_cmds = np.vstack([
                self.navigation_target_cmds, 
                np.concatenate([navigation_target_pos[:2], navigation_target_axa[-1:]], axis=0) 
            ])

            if navigation_target_pos.any() or self.navigation_command == None:
            # if goal_base_pos.any() or self.navigation_command == None:
                self.navigation_command = ul.NavigationCommand(
                    # navigation target pose
                    x = navigation_target_pos[0],
                    y = navigation_target_pos[1],
                    heading = navigation_target_axa[2],
                    
                    # goal base pose
                    # x = goal_base_pos[0],
                    # y = goal_base_pos[1],
                    # heading = goal_base_axa[2],

                    # zero pose
                    # x = 0.0,
                    # y = 0.0,
                    # heading = 0.0,
                )

            # self.target_vec_b = self.navigation_command.command(xyz, quat_wxyz)
            self.target_vec_b = self.navigation_command.command(mid_sole_pos_w, mid_sole_quat_w)
            self.target_vec_bs = np.vstack([self.target_vec_bs, self.target_vec_b])

            
                

            self.publish_navigation_target(
                self.navigation_command.pos_command_w,
                quat_from_euler_xyz(np.zeros_like(self.navigation_command.heading_command_w), 
                                    np.zeros_like(self.navigation_command.heading_command_w),
                                    self.navigation_command.heading_command_w).squeeze(0)
            )
            
            if self.navigation_counter % 10 == 0:
                
                self.nav_obs = self.navigation_obsmap(self.low_state, 
                                                    self.target_vec_b, # 4 dim
                                                    self.locomotion_phase_command, 
                                                    last_action=self.naviation_last_action)
                obs_tensor = th.from_numpy(self.nav_obs).unsqueeze(0)
                obs_tensor = obs_tensor.detach().clone().float()

                self.navigation_action = self.navigation_policy(obs_tensor).detach().numpy().squeeze()
                self.locomotion_vel_command = self.navigation_actmap(self.navigation_action)
                self.naviation_last_action = self.locomotion_vel_command
                self.locomotion_vel_commands = np.vstack([self.locomotion_vel_commands, self.locomotion_vel_command])
                
                self.navigation_counter = 0
                
            self.navigation_counter += 1


            if self.remote_controller.ly or self.remote_controller.lx or self.remote_controller.rx:
                self.locomotion_vel_command = np.array([0., 0., 0.])
                self.locomotion_vel_command[0] = np.clip(self.remote_controller.ly, -0.2, 0.3)
                self.locomotion_vel_command[1] = np.clip(self.remote_controller.lx * -1, -0.2, 0.2)
                self.locomotion_vel_command[2] = np.clip(self.remote_controller.rx * -1, -0.5, 0.5)

            # print(self.locomotion_vel_command)

            self.obs = self.locomotion_obsmap(self.low_state, self.locomotion_vel_command, self.locomotion_phase_command, last_action=self.locomotion_last_action)
            obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
            obs_tensor = obs_tensor.detach().clone().float()
            self.locomotion_action = self.locomotion_policy(obs_tensor).detach().numpy().squeeze()
            self.locomotion_last_action = self.locomotion_action

            # target_dof_pos : motor joint ordered
            self.locomotion_target_dof_pos = self.locomotion_actmap(self.locomotion_action)
            target_dof_pos = self.locomotion_target_dof_pos



            

        elif self.task == "sit":
            # Press down button to sit
            if self.remote_controller.button[KeyMap.up] == 1:
                self.target_height = 0.7
                print(f"Target height set to {self.target_height}")
            if self.remote_controller.button[KeyMap.down] == 1:
                self.target_height = 0.38
                print(f"Target height set to {self.target_height}")

            # height_command = self.vhcommand(current_pelvis_height_w = xyz[2] + 0.04, target_height=self.target_height)
            height_command = self.vhcommand(current_pelvis_height_w = xyz[2], target_pelvis_height_w=self.target_height)

            # For stage 1 & 2.
            self.obs = self.sit_obsmap(self.low_state, height_command, xyz)

            obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
            obs_tensor = obs_tensor.detach().clone().float()
            self.sit_action = self.sit_policy(obs_tensor).detach().numpy().squeeze()

            # target_dof_pos : motor joint ordered
            self.sit_target_dof_pos = self.sit_actmap(self.sit_action)
            ####### TO DISABLE SITTING #######
            # self.sit_target_dof_pos = self.q_traj[-1].copy()
            self.last_sit_dof_pos[self.mot_from_lab] = self.q_traj[-1]
        # if self.task == "sit":
            target_dof_pos = self.sit_target_dof_pos
            self.waist_rot_target_dof_pos = self.sit_target_dof_pos

            # self.print_sit_status()
        elif self.task == "waist_rot":
            if self.remote_controller.button[KeyMap.left] == 1: 
                self.waist_res_q[self.joint_name_to_idx['waist_roll_joint']] += + 0.5 * np.pi / 180
            elif self.remote_controller.button[KeyMap.right] == 1:
                self.waist_res_q[self.joint_name_to_idx['waist_roll_joint']] += - 0.5 * np.pi / 180
            elif self.remote_controller.button[KeyMap.up] == 1:
                self.waist_res_q[self.joint_name_to_idx['waist_yaw_joint']] += + 0.5 * np.pi / 180
            elif self.remote_controller.button[KeyMap.down] == 1:
                self.waist_res_q[self.joint_name_to_idx['waist_yaw_joint']] += - 0.5 * np.pi / 180
            elif self.remote_controller.button[KeyMap.select] == 1:
                self.waist_res_q[self.joint_name_to_idx['waist_pitch_joint']] += + 0.5 * np.pi / 180
            elif self.remote_controller.button[KeyMap.F1] == 1:
                self.waist_res_q[self.joint_name_to_idx['waist_pitch_joint']] += - 0.5 * np.pi / 180

            target_dof_pos = self.sit_target_dof_pos.copy()
            for i, i_mot_from_controller in enumerate(self.mot_from_controller):
                # TODO:
                # Does the target_dof_pos have the same order as the motor_state?
                target_dof_pos[i_mot_from_controller] += self.waist_res_q[i]
            
            self.waist_rot_target_dof_pos = target_dof_pos.copy()
            
        elif self.task == "eetrack":
            if self.is_eetrack_first_iter:
                print("\n[EETrack] EETrack has began.")
                self.is_eetrack_first_iter = False
                self.eetrack_initial_counter = self.counter

            if self.eetrack_command is None:
                self.eetrack_command = ue.eetrack(
                    th.from_numpy(root_state_w)[None],
                    self.tf_buffer,
                    clock, to_start=False)

            # Keymap press -> changes is_initial_goal == False
            if self.remote_controller.button[KeyMap.start] == 1:
                print("\n[EETrack] To welding line start Sampling has begun.")
                self.eetrack_command.is_initial_goal = False

            if self.remote_controller.button[KeyMap.F2] == 1: # F2 is F3 button in controller
                print("\n[EETrack] Subgoal Sampling has begun.")
                self.eetrack_command.is_initial_eetrack = False
                self.is_go_start = False
            
            # print(self.eetrack_command.eetrack_start_w)
            if self.remote_controller.button[KeyMap.up] == 1:
                self.weld_dx += 0.001
                # Welding pose
                # self.eetrack_command.eetrack_start_w += 0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_start_quat_w))[:3,0]
                # self.eetrack_command.eetrack_end_w += 0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_end_quat_w))[:3,0]
                # self.eetrack_command.create_eetrack()
                # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
                print("Increase x")
            if self.remote_controller.button[KeyMap.down] == 1:
                self.weld_dx -= 0.001
                # self.eetrack_command.eetrack_start_w += -0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_start_quat_w))[:3,0]
                # self.eetrack_command.eetrack_end_w += -0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_end_quat_w))[:3,0]
                # self.eetrack_command.create_eetrack()
                # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
                print("Decrease x")
            if self.remote_controller.button[KeyMap.left] == 1:
                self.weld_dy += 0.001
                # self.eetrack_command.eetrack_start_w += 0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_start_quat_w))[:3,1]
                # self.eetrack_command.eetrack_end_w += 0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_end_quat_w))[:3,1]
                # self.eetrack_command.create_eetrack()
                # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
                print("Increase y")
            if self.remote_controller.button[KeyMap.right] == 1:
                self.weld_dy -= 0.001
                # self.eetrack_command.eetrack_start_w += -0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_start_quat_w))[:3,1]
                # self.eetrack_command.eetrack_end_w += -0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_end_quat_w))[:3,1]
                # self.eetrack_command.create_eetrack()
                # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
                print("Decrease y")
            if self.remote_controller.button[KeyMap.L1] == 1:
                self.weld_dz += 0.001
                # self.eetrack_command.eetrack_start_w += 0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_start_quat_w))[:3,2]
                # self.eetrack_command.eetrack_end_w += 0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_end_quat_w))[:3,2]
                # self.eetrack_command.create_eetrack()
                # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
                print("Increase z")
            if self.remote_controller.button[KeyMap.L2] == 1:
                self.weld_dz -= 0.001
                # self.eetrack_command.eetrack_start_w += -0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_start_quat_w))[:3,2]
                # self.eetrack_command.eetrack_end_w += -0.001*matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_end_quat_w))[:3,2]
                # self.eetrack_command.create_eetrack()
                # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
                print("Decrease z")
            if self.remote_controller.button[KeyMap.R1] == 1:
                self.weld_dpitch += 0.01
                # eetrack_start_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_start_quat_w[None])
                # eetrack_end_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_end_quat_w[None])
                # eetrack_start_euler_w[1] += 0.01
                # eetrack_end_euler_w[1] += 0.01
                # self.eetrack_command.eetrack_start_quat_w = quat_from_euler_xyz(*eetrack_start_euler_w)[0]
                # self.eetrack_command.eetrack_end_quat_w = quat_from_euler_xyz(*eetrack_end_euler_w)[0]
                # self.eetrack_command.create_eetrack()
                # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
                print(f"Start pitch: {eetrack_start_euler_w[1][0]}, end pitch: {eetrack_end_euler_w[1][0]}")
            if self.remote_controller.button[KeyMap.R2] == 1:
                self.weld_dpitch -= 0.01
                # eetrack_start_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_start_quat_w[None])
                # eetrack_end_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_end_quat_w[None])
                # eetrack_start_euler_w[1] += -0.01
                # eetrack_end_euler_w[1] += -0.01
                # self.eetrack_command.eetrack_start_quat_w = quat_from_euler_xyz(*eetrack_start_euler_w)[0]
                # self.eetrack_command.eetrack_end_quat_w = quat_from_euler_xyz(*eetrack_end_euler_w)[0]
                # self.eetrack_command.create_eetrack()
                # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
                print(f"Start pitch: {eetrack_start_euler_w[1][0]}, end pitch: {eetrack_end_euler_w[1][0]}")
            # if self.remote_controller.button[KeyMap.R1] == 1:
            #     self.weld_dyaw += 0.01
            #     eetrack_start_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_start_quat_w[None])
            #     eetrack_end_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_end_quat_w[None])
            #     eetrack_start_euler_w[2] += 0.01
            #     eetrack_end_euler_w[2] += 0.01
            #     self.eetrack_command.eetrack_start_quat_w = quat_from_euler_xyz(*eetrack_start_euler_w)[0]
            #     self.eetrack_command.eetrack_end_quat_w = quat_from_euler_xyz(*eetrack_end_euler_w)[0]
            #     self.eetrack_command.create_eetrack()
            #     self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
            #     print(f"dyaw: {self.weld_dyaw}")
            # if self.remote_controller.button[KeyMap.R2] == 1:
            #     self.weld_dyaw -= 0.01
            #     eetrack_start_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_start_quat_w[None])
            #     eetrack_end_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_end_quat_w[None])
            #     eetrack_start_euler_w[2] += -0.01
            #     eetrack_end_euler_w[2] += -0.01
            #     self.eetrack_command.eetrack_start_quat_w = quat_from_euler_xyz(*eetrack_start_euler_w)[0]
            #     self.eetrack_command.eetrack_end_quat_w = quat_from_euler_xyz(*eetrack_end_euler_w)[0]
            #     self.eetrack_command.create_eetrack()
            #     self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()
            #     print(f"dyaw: {self.weld_dyaw}")

            if self.remote_controller.button[KeyMap.F1] == 1:
                self.eetrack_command = ue.eetrack(
                    th.from_numpy(root_state_w)[None],
                    self.tf_buffer,
                    clock,
                    to_start=True,
                    start_ee_pos=self.start_ee_pos,
                    start_ee_quat=self.start_ee_quat,
                )
                print("[EETrack] back to start position. Press 'start' button to execute.")
            if self.remote_controller.button[KeyMap.select] == 1:
                self.is_go_start = True
                print("Go back to start hand pose")

            _ = self.eetrack_command.get_command(
                th.from_numpy(root_state_w)[None]
                )[0].detach().cpu().numpy()
            self.target_pose_w = np.copy(
                self.eetrack_command.next_command_s_left.squeeze().detach().cpu().numpy()
            )
            self.target_pose_b = np.concatenate([
                self.eetrack_command.lerp_command_b_left_pos.squeeze().detach().cpu().numpy(),
                self.eetrack_command.lerp_command_b_left_quat.squeeze().detach().cpu().numpy(),
            ])

            if False:
                pos, quat = subtract_frame_transforms(
                root_state_w[..., 0:3],
                root_state_w[..., 3:7],
                self.eetrack_command.eetrack_start_w,
                self.eetrack_command.eetrack_start_quat_w,
                )
                self.target_pose_b = np.concatenate([pos, quat])

            self.publish_hand_target()

            if not self.is_go_start:
                # Get current joint positions
                qj = np.zeros(29, dtype=np.float32)
                for i_mot in range(len(self.config.motor_joint)):
                    i_pin = self.pin_from_mot[i_mot]
                    qj[i_pin] = self.low_state.motor_state[i_mot].q

                gravity_vec = 9.81*self.sit_obsmap._projected_gravity()
                res_q, arm_nle = self.ikctrl(qj,
                                            self.target_pose_b,
                                            rel=False,
                                            gravity_vec=gravity_vec,
                                            )
                res_q = 2*res_q

                target_dof_pos = self.sit_target_dof_pos.copy()
                for i_act in range(len(res_q)):
                    i_mot = self.mot_from_act[i_act]
                    i_pin = self.pin_from_mot[i_mot]
                    target_q_i = (
                            self.low_state.motor_state[i_mot].q + res_q[i_act]
                    )
                    target_q_i = np.clip(target_q_i,
                                    self.lim_lo_pin[i_pin],
                                    self.lim_hi_pin[i_pin])
                    target_dof_pos[i_mot] = target_q_i
                    target_tau[i_mot] = arm_nle[i_act]
            else:
                if self.eetrack_command.to_start:
                    qj = np.zeros(29, dtype=np.float32)
                    for i_mot in range(len(self.config.motor_joint)):
                        qj[i_mot] = self.low_state.motor_state[i_mot].q
                    target_dof_pos = self.sit_target_dof_pos.copy()
                    target_dof_pos[-7:] = qj[-7:] + (self.last_sit_dof_pos[-7:] - qj[-7:]).clip(-0.01,0.01)
                else:
                    if self.trajopt_joint_traj is None:
                        trajopt_init_dof_pos = self.prev_target_dof_pos[-7:]
                    else:
                        trajopt_init_dof_pos = np.array(self.trajopt_joint_traj.points[0].positions)
                    qj = np.zeros(29, dtype=np.float32)
                    for i_mot in range(len(self.config.motor_joint)):
                        qj[i_mot] = self.low_state.motor_state[i_mot].q
                    target_dof_pos = self.sit_target_dof_pos.copy()
                    target_dof_pos[-7:] = qj[-7:] + (trajopt_init_dof_pos - qj[-7:]).clip(-0.01,0.01)

                    # qj_pin = np.zeros(29, dtype=np.float32)
                    # for i_mot in range(len(self.config.motor_joint)):
                    #     i_pin = self.pin_from_mot[i_mot]
                    #     qj_pin[i_pin] = self.low_state.motor_state[i_mot].q
                    # gravity_vec = 9.81*self.sit_obsmap._projected_gravity()
                    # arm_nle = self.ikctrl.get_gravity_compensation(
                    #     qj_pin,
                    #     gravity_vec=gravity_vec,
                    # )
                    # for i_act in range(7):
                    #     i_mot = self.mot_from_act[i_act]
                    #     target_tau[i_mot] = arm_nle[i_act]

                    joint_pos_diff = np.linalg.norm(trajopt_init_dof_pos - qj[-7:])
                    print("Joint pos diff: ", joint_pos_diff)
                    if joint_pos_diff < 0.005:
                        self.is_go_start = False
                        print("is_go_start disabled!!!")

        raw_target_dof_pos = target_dof_pos.copy()
        
        # observation dumping
        self.dump_observations_and_joint_pos_target(raw_target_dof_pos, target_dof_pos)

        # FIXME(hh) kpkd coefficient smoothing
        # Build low cmd
        for mot_idx in range(self.num_joints):
            if self.task == "locomotion":
                if True:
                    if self.prev_joint_pos_target is not None:
                        if self.counter < 10:
                            target_dof_pos = self.config.initial_smoothing * target_dof_pos + \
                                            (1-self.config.initial_smoothing) * self.prev_joint_pos_target
                        else:
                            target_dof_pos = self.config.locomotion_later_smoothing * target_dof_pos + \
                                            (1-self.config.locomotion_later_smoothing) * self.prev_joint_pos_target

                    self.prev_joint_pos_target = target_dof_pos

                self.low_cmd.motor_cmd[mot_idx].q = 0.#float(target_dof_pos[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].dq = 0.0
                self.low_cmd.motor_cmd[mot_idx].kp = 0.#self.config.kpkd_smoothing * float(self.config.locomotion_kps[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].kd = 0.#self.config.kpkd_smoothing * float(self.config.locomotion_kds[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].tau = 0.0


                l2_err = np.linalg.norm(self.target_vec_bs[-50:,:2], axis=-1).mean()
                ori_err = np.abs(self.target_vec_bs[-50:,2]).mean()
                print(f"l2_err: {l2_err}, ori_err: {ori_err}")

                # if l2_err < 0.05 and ori_err < 0.1:
                #     print("Navigation target reached.")
                #     self.task = "sit"

            elif self.task == "sit":
                if True:
                    if self.prev_joint_pos_target is not None:
                        target_dof_pos = self.config.sit_later_smoothing * target_dof_pos + \
                                        (1-self.config.sit_later_smoothing) * self.prev_joint_pos_target

                    self.prev_joint_pos_target = target_dof_pos

                self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].dq = 0.0
                ########## To disable sitting ##########
                # self.low_cmd.motor_cmd[mot_idx].kp = 0.0
                # self.low_cmd.motor_cmd[mot_idx].kd = 0.0
                self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(self.sit_kps[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(self.sit_kds[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].tau = 0.0
            elif self.task == "waist_rot":
                self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].dq = 0.0
                self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(self.sit_kps[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(self.sit_kds[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].tau = 0.0
            elif self.task == "eetrack":
                self.prev_target_dof_pos = target_dof_pos
                self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].dq = 0.0
                self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(self.sit_kps[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(self.sit_kds[mot_idx])
                self.low_cmd.motor_cmd[mot_idx].tau = float(target_tau[mot_idx])

            # For debugging
            # self.low_cmd.motor_cmd[mot_idx].kp = 0.0
            # self.low_cmd.motor_cmd[mot_idx].kd = 0.0
         
        # send the command
        self.send_cmd(self.low_cmd)

        if self.remote_controller.button[KeyMap.select] == 1:
            self.log_metrics_and_trajectories(terminate=False)

    def dump_observations_and_joint_pos_target(self, raw_target_dof_pos, target_dof_pos):
        # log timestamp
        timestamp_low_freq = clock.get_time().nanoseconds / 1e9
        self.timestamp_low_freq = np.append(self.timestamp_low_freq, timestamp_low_freq)
        
        # log raw joint pos target (before smoothing)
        raw_target_dof_pos_lab = np.zeros(29)
        raw_target_dof_pos_lab[self.lab_from_mot] = raw_target_dof_pos
        self.raw_joint_pos_targets = np.vstack((self.raw_joint_pos_targets, raw_target_dof_pos_lab))
        
        # log joint pos target
        target_dof_pos_lab = np.zeros(29)
        target_dof_pos_lab[self.lab_from_mot] = target_dof_pos
        self.joint_pos_targets = np.vstack((self.joint_pos_targets, target_dof_pos_lab))

        self.task_traj.append(self.task)
        
        # log observation
        if self.task == "locomotion":
            self.locomotion_observations = np.vstack((self.locomotion_observations, self.obs))
            self.locomotion_actions = np.vstack((self.locomotion_actions, self.locomotion_action))
            self.locomotion_cmd_traj = np.vstack((self.locomotion_cmd_traj, self.locomotion_vel_command))
        elif self.task == "sit":
            self.sit_observations = np.vstack((self.sit_observations, self.obs))
            self.sit_actions = np.vstack((self.sit_actions, self.sit_action))
        elif self.task == "waist_rot":
            pass
        elif self.task == "eetrack":
            self.is_initial_goals = np.hstack((self.is_initial_goals, self.eetrack_command.is_initial_goal))
            self.is_initial_eetracks = np.hstack((self.is_initial_eetracks, self.eetrack_command.is_initial_eetrack))
            self.is_go_starts = np.hstack((self.is_go_starts, self.is_go_start))
            self.target_poses_w = np.vstack((self.target_poses_w, self.target_pose_w))
            self.target_poses_b = np.vstack((self.target_poses_b, self.target_pose_b))
            ee_pos_w, ee_quat_w = body_pose(self.tf_buffer, "end_effector", "world", rot_type="quat")
            self.ee_poses_w = np.vstack((self.ee_poses_w, np.concatenate((ee_pos_w, ee_quat_w))))
            ee_pos_b, ee_quat_b = body_pose(self.tf_buffer, "end_effector", "pelvis", rot_type="quat")
            self.ee_poses_b = np.vstack((self.ee_poses_b, np.concatenate((ee_pos_b, ee_quat_b))))
        else:
            raise ValueError("Invalid task")
        
            
    def log_metrics_and_trajectories(self, terminate=True):
        # Calculate pos diff & torque diff metrics
        pos_diff = np.average(np.abs(self.q_traj[1:] - self.q_traj[:-1]))
        torque_diff = np.average(np.abs(self.tau_traj[1:] - self.tau_traj[:-1]))
        print("\n--------------------------METRICS--------------------------")
        print("total_time", self.counter * self.config.control_dt)
        print("pos_diff", pos_diff)
        print("torque_diff", torque_diff)
        print("----------------------------------------------------------")
        
        # Normalize timestamps with respect to the first timestamp_high_freq
        timestamp_low_freq = self.timestamp_low_freq - self.timestamp_high_freq[0]
        timestamp_high_freq = self.timestamp_high_freq - self.timestamp_high_freq[0]
        
        # Save metrics and trajectories
        metrics = {
            "pos_diff": pos_diff,
            "torque_diff": torque_diff
        }
        trajectories = {
            "timestamp_high_freq": timestamp_high_freq,
            "q_traj": self.q_traj,
            "dq_traj": self.dq_traj,
            "tau_traj": self.tau_traj,
            "timestamp_low_freq": timestamp_low_freq,
            "locomotion_cmds": self.locomotion_cmd_traj,
            "locomotion_observations": self.locomotion_observations,
            "locomotion_actions": self.locomotion_actions,
            "sit_observations": self.sit_observations,
            "sit_actions": self.sit_actions,
            "raw_joint_pos_targets": self.raw_joint_pos_targets,
            "tasks": self.task_traj,
            "joint_pos_targets" : self.joint_pos_targets,
            "is_initial_goals": self.is_initial_goals,
            "is_initial_eetracks": self.is_initial_eetracks,
            "is_go_starts": self.is_go_starts,
            "target_poses_w": self.target_poses_w,
            "target_poses_b": self.target_poses_b,
            "ee_poses_w": self.ee_poses_w,
            "ee_poses_b": self.ee_poses_b,
            "target_vec_bs": self.target_vec_bs,
            "slam_quat" : self.slam_quat,
            "imu_quat" : self.imu_quat,
            "navigation_target_cmds": self.navigation_target_cmds,
        }
        log_data = {
            "metrics": metrics,
            "trajectories": trajectories
        }

        # Save the log with experiment name
        sit_model = os.path.basename(self.config.sit_policy_path).split('.')[0]

        timestamp = clock.get_time().nanoseconds / 1e9
        
        file_name = f"{self.logpath}/log_{sit_model}_ikctrl_{str(timestamp).split('.')[0]}.npy"
        np.save(file_name, log_data)
        print(f"Log saved at {file_name}")
        
        # totally terminate
        if terminate:
            self._mode_change = True
            self.mode = Mode.null
        

    def run_wrapper(self):
        if self.mode == Mode.wait:
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
                print("Waiting for the Button F3 signal...")
                self._mode_change = False
            self.default_pos_state()
        elif self.mode == Mode.policy:
            if self._mode_change:
                print("Run Policy.\n")
                print("--------------[ Basic Guidelines ]---------------")
                print("[Locomotion] Press Button {X} to start locomotion.")
                print("-------------------------------------------------")
                print("[Sit] Press Button {Y} to start sit (default target height = 0.7m).")
                print("-------------------------------------------------")
                print("[Sit] Press Button {down} to move pelvis to target height 0.3m.")
                print("-------------------------------------------------")
                print("[Sit] Press Button {up} to move pelvis to target height 0.7m.")
                print("-------------------------------------------------")
                print("-------------------------------------------------")
                print("[EETrack] Press Button {B} to change task to EETrack.")
                print("-------------------------------------------------")
                print("[EETrack] Press down {start} to go to welding start pose.")
                print("-------------------------------------------------")
                print("[EETrack] Adjust welding position.\n{up}:+x / {down}:-x\n{left}:+y / {right}:-y\n{L1}:+z/{L2}:-z")
                print("-------------------------------------------------")
                print("[EETrack] Press down {F3} to start welding line following.")
                print("-------------------------------------------------")
                print("[EETrack] Press down {F1} to go back to original right hand pose.")
                print("-------------------------------------------------")
                print("[EETrack] Press down {select} to save log.")
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