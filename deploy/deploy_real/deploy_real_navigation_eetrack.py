import os
import torch as th
import numpy as np
import math_utils
from pathlib import Path
from typing import Union, Literal
from legged_gym import LEGGED_GYM_ROOT_DIR

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo
from rclpy.duration import Duration

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
from std_msgs.msg import Empty

import utils_robot as ur
import utils_locomotion as ul
import utils_stage as us
import utils_eetrack_tag as ue

from std_msgs.msg import MultiArrayLayout

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

from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import TransformException

_last_tf = None  # module/class-level cache

def map_transform(t, rot_type="quat"):
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

    
def body_pose(
        tf_buffer,
        frame: str,
        ref_frame: str = 'pelvis',
        stamp=None,
        rot_type: str = 'axa'):
    """ --> tf does not exist """
    # global _last_tf
    if stamp is None:
        stamp = rp.time.Time()
        # stamp = clock.get_time()
    try:
        # t = "ref{=pelvis}_from_frame" transform
        t = tf_buffer.lookup_transform(
            ref_frame,  # to
            frame,  # from
            stamp)
        # _last_tf = t
    except TransformException as ex:
        print(f'Could not transform {frame} to {ref_frame}: {ex}')
        # return map_transform(_last_tf, rot_type)

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
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

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
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        # self.tf_listener = TransformListener(self.tf_buffer, self._node, spin_thread=True)

        # qos_tf = QoSProfile(depth=100)
        # qos_tf.reliability = ReliabilityPolicy.RELIABLE
        # qos_tf.durability  = DurabilityPolicy.VOLATILE

        # qos_tf_static = QoSProfile(depth=1)
        # qos_tf_static.reliability = ReliabilityPolicy.RELIABLE
        # qos_tf_static.durability  = DurabilityPolicy.TRANSIENT_LOCAL
        # self.tf_listener = TransformListener(self.tf_buffer, self._node, spin_thread=True, qos=qos_tf, static_qos=qos_tf_static)

        self.tf_broadcaster = TransformBroadcaster(self._node)

        ########################## Locomotion ##########################
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


        ########################## Sit ##########################
        self.sit_obsmap = us.SitObservation(config, self.tf_buffer)
        self.sit_robot = self.locomotion_robot
        self.sit_actmap = us.SitActionVer2(config, self.sit_robot)
        self.vhcommand = us.VelocityHeightCommand(config)
        self.sit_policy = th.jit.load(config.sit_policy_path) # FIXME sit_policy_path
        self.sit_policy.eval()

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
        self.navigation_counter = 0
        self.sit_counter = 0
        self.stop_state = False
        self.minimum_locomotion_iter = 0
        self.prev_locomotion_vel_command = np.zeros(3)


        self.image_capture_publisher = self._node.create_publisher(Empty, '/capture_trigger', 10)
        self.bending_counter = 0

        # Subscribe to /eetrack_vision topic (MultiArrayLayout)
        self.eetrack_vision_subscriber = self._node.create_subscription(
            MultiArrayLayout,
            '/eetrack_vision/weldpoints',
            self.eetrack_vision_callback,
            10
        )
        self.eetrack_vision_points = None



        self.bending_offset = 0.
        self.bending_target_dof = None
        
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
        # self.navigation_obsmap = ul.NavigationObservation(config, self.tf_buffer)
        # self.navigation_actmap = ul.NavigationAction(config, self.locomotion_robot)
        # print(config.navigation_policy_path)
        # self.navigation_policy = th.jit.load(config.navigation_policy_path) # TODO
        # self.navigation_policy.eval()
        self.naviation_last_action = np.zeros(3, dtype=np.float32)
        self.locomotion_vel_commands = np.zeros((0, 3))

        self.qj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_actions, dtype=np.float32)

        ######################### eetrack #########################
        self.eetrack_command = None
        self.is_eetrack_first_iter = True
        self.is_go_start = False

        self.weld_dx = 0.0
        self.weld_dy = 0.0
        self.weld_dz = 0.0
        self.weld_dpitch = 0.0
        self.weld_dyaw = 0.0

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

        q_mot = np.array(config.mot_joint_offsets)
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

        print("Waiting for the robot to be ready...")
        self.mode = Mode.wait
        self.task : Literal["locomotion", "navigation", "sit", "eetrack"] = "locomotion"
        # self.task : Literal["locomotion", "navigation", "sit", "eetrack"] = "sit"
        self.zero_phase = False

        self.sitting = False
        self._mode_change = True
        self._terminate = False

        # calls run_wrapper every self.config.control_dt seconds
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

    def eetrack_vision_callback(self, msg: 'MultiArrayLayout'):
    # TODO: Implement handling of the received MultiArrayLayout message
        self.eetrack_vision_points = msg
        print("CALLBACK!!")
        print(msg)
    
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
        if self.counter < self._num_step: # NOTE (bk) what's this if statement for?
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
            # NOTE (bk) why are we switching to policy mode here?
            self._mode_change = True
            self.mode = Mode.policy

    def default_pos_state(self):
        if self.remote_controller.button[KeyMap.Y] != 1:
            # NOTE (bk) what does this code snippet do? perhaps it sends the robot to default pos?
            # return
            for motor_idx in range(self.num_joints):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.locomotion_motor_joint_offsets[motor_idx]
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[motor_idx])
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
        
    def terminate_by_pelvis_condition(self, xyz, quat, limit_euler_angle=[1.7, 1.7]) -> bool:
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
        ############################# MAIN LOOP #############################
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
            rot_type='quat',
            # stamp=rp.time.Time()
        )

        xyz, quat_wxyz = world_from_pelvis
        root_state_w = np.zeros(7)
        root_state_w[0:3] = xyz
        root_state_w[3:7] = quat_wxyz

        in_nav_mode = False
        if in_nav_mode:
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
        pos_command_b = quat_rotate_inverse(yaw_quat(quat_wxyz).astype(np.float32), target_vec.astype(np.float32))

        self.pos_error_bs = np.vstack((self.pos_error_bs, np.linalg.norm(pos_command_b[:2]).reshape((1,1))))
        self.heading_error_bs = np.vstack((self.heading_error_bs, heading_error.reshape(1,1)))


        ############################################ SWITCH MODE FROM LOCOMOTION TO NAVIVATION ############################################
        if self.remote_controller.button[KeyMap.Y] == 1 and self.task not in  ["eetrack", "sit"]:
            print("============== locomotion mode activated ==============")
            self.task = "locomotion" 
        if self.remote_controller.button[KeyMap.X] == 1 and self.task not in  ["eetrack", "sit"]:
            print("============== navigation mode activated ==============")
            self.task = "navigation"
        if self.remote_controller.button[KeyMap.B] == 1 and self.task != "eetrack":
            print("============== Sitting mode activated ==============")
            self.task = "sit"
        if self.remote_controller.button[KeyMap.F1] == 1:
            print("============== eetrack mode activated ==============")
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
                  start_ee_quat=start_ee_quat,
                  eetrack_vision_points = self.eetrack_vision_points)
            
            # start_T = matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_start_quat_w))
            # end_T = matrix_from_quat(yaw_quat(self.eetrack_command.eetrack_end_quat_w))
            # self.eetrack_command.eetrack_start_w += self.weld_dx*start_T[:3,0]
            # self.eetrack_command.eetrack_start_w += self.weld_dy*start_T[:3,1]
            # self.eetrack_command.eetrack_start_w += self.weld_dz*start_T[:3,2]
            # self.eetrack_command.eetrack_end_w += self.weld_dx*end_T[:3,0]
            # self.eetrack_command.eetrack_end_w += self.weld_dy*end_T[:3,1]
            # self.eetrack_command.eetrack_end_w += self.weld_dz*end_T[:3,2]
            # eetrack_start_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_start_quat_w[None])
            # eetrack_end_euler_w = euler_xyz_from_quat(self.eetrack_command.eetrack_end_quat_w[None])
            # # eetrack_start_euler_w[1] += self.weld_dpitch
            # # eetrack_end_euler_w[1] += self.weld_dpitch
            # eetrack_start_euler_w[2] += self.weld_dyaw
            # eetrack_end_euler_w[2] += self.weld_dyaw
            # self.eetrack_command.eetrack_start_quat_w = quat_from_euler_xyz(*eetrack_start_euler_w)[0]
            # self.eetrack_command.eetrack_end_quat_w = quat_from_euler_xyz(*eetrack_end_euler_w)[0]
            # self.eetrack_command.create_eetrack()
            # self.eetrack_command.eetrack_subgoal = self.eetrack_command.create_subgoal()


            # self.start_ee_pos, self.start_ee_quat = body_pose(
            #     self.tf_buffer,
            #     'end_effector',
            #     'world',
            #     rot_type='quat'   
            # )
            # self.is_go_start = False

        if self.remote_controller.button[KeyMap.select] == 1:
            print("Capture ZED images")
            msg = Empty()
            self.image_capture_publisher.publish(msg)
        if self.remote_controller.button[KeyMap.R2] == 1:
            self.task = "bending"


        ############################################ LOCOMOTION ############################################
        if self.task == "locomotion":
            # NOTE (bk): what's this if statement? needs to be re-written
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

            else:
                self.locomotion_vel_command = np.zeros(3)
                phase = self.locomotion_counter = 0
        
        ############################################ NAVIGATION ############################################
        elif self.task == "navigation":
            print("Pelvis height :", xyz[-1])
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
                    # self.locomotion_vel_command[:2] = np.clip(np.sign(pos_command_b[:2]) * MAX_LIN_VEL * np.sqrt(np.abs(pos_command_b[:2] / SLOW_BOUND)), -MAX_LIN_VEL, MAX_LIN_VEL)
                    # X >= 0
                    if pos_command_b[0] >= 0:
                        self.locomotion_vel_command[0] = np.clip(0.08 * np.sqrt(np.abs(pos_command_b[0] / 0.8)), 0., 0.08)
                    # X < 0
                    if pos_command_b[0] < 0:
                        self.locomotion_vel_command[0] = np.clip(-0.3 * np.sqrt(np.abs(pos_command_b[0] / 0.2)), -0.3, 0.)
                    # Y >= 0
                    if pos_command_b[1] >= 0:
                        self.locomotion_vel_command[1] = np.clip(0.1 * np.sqrt(np.abs(pos_command_b[1] / 0.8)), 0., 0.1)
                    # Y < 0
                    if pos_command_b[1] < 0:
                        self.locomotion_vel_command[1] = np.clip(-0.2 * np.sqrt(np.abs(pos_command_b[1] / 0.2)), -0.2, 0.)
                    
                    self.locomotion_vel_command[2] = np.clip(np.sign(heading_error) * self.MAX_ANG_VEL * np.sqrt(np.abs(heading_error / self.SLOW_BOUND)), -self.MAX_ANG_VEL, self.MAX_ANG_VEL)

                if self.stop_locomotion:
                    self.locomotion_vel_command = np.array([0., 0., 0.])
                    phase = 0.0

            if self.navigation_counter < 100:
                # vel cmd smoothing
                alpha = self.navigation_counter / 100
                self.locomotion_vel_command = np.zeros(3) * (1-alpha) + self.locomotion_vel_command * alpha
            
                self.navigation_counter += 1
        
        # phase = 0.0
        target_tau = np.zeros(29, dtype=np.float32)
        if self.task in ["locomotion", "navigation"]:
            self.obs = self.locomotion_obsmap(self.low_state, self.locomotion_vel_command, phase, last_action=self.locomotion_last_action)
            obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
            obs_tensor = obs_tensor.detach().clone().float()
            self.locomotion_action = self.locomotion_policy(obs_tensor).detach().numpy().squeeze()
            self.locomotion_last_action = self.locomotion_action

            # target_dof_pos : motor joint ordered
            if "0901_f2" in self.config.locomotion_policy_path:
                is_arm_action = True
            else:
                is_arm_action = False
            self.locomotion_target_dof_pos = self.locomotion_actmap(self.locomotion_action, arm_action=is_arm_action)
            target_dof_pos = self.locomotion_target_dof_pos

            self.bending_offset = 0
            self.bending_target_dof = None
            self.bending_counter = 0

        elif self.task == "bending":
            target_dof_pos = self.locomotion_target_dof_pos.copy()
            if self.bending_target_dof is None:
                self.bending_target_dof = self.locomotion_target_dof_pos.copy()
            down_button = self.remote_controller.button[KeyMap.down]

            if down_button == 1:
                print(f"Waist pitch bent : {self.bending_offset} [rad]")
                self.bending_offset += 0.01
                # 버튼 누른 상태에서는 현재 관절 그대로 유지
                self.bending_counter = 0
                self.last_target_dof = self.bending_target_dof.copy()

            else:
                self.bending_offset = np.clip(self.bending_offset, a_min= -1.0, a_max= 0.087)
                # 버튼 안 눌렀을 때만 목표 위치로 스무스하게 이동
                if self.bending_counter < 100:
                    alpha = self.bending_counter / 100
                    waist_pos = self.last_target_dof[14] * (1 - alpha) + (self.last_target_dof[14] + self.bending_offset) * alpha
                    target_dof_pos[14] = waist_pos
                    self.bending_target_dof = target_dof_pos
                    self.bending_counter += 1
                elif self.bending_counter == 100:
                    target_dof_pos = self.bending_target_dof



        elif self.task == "sit":
            if self.remote_controller.button[KeyMap.down] == 1:
                self.sitting = True

            height_command = self.vhcommand(current_pelvis_height_w = xyz[2] + 0.00, sitting=self.sitting)

            # For stage 1 & 2.
            self.obs = self.sit_obsmap(self.low_state, height_command, xyz)

            obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
            obs_tensor = obs_tensor.detach().clone().float()
            self.sit_action = self.sit_policy(obs_tensor).detach().numpy().squeeze()

            # target_dof_pos : motor joint ordered
            self.sit_target_dof_pos = self.sit_actmap(self.sit_action)
            target_dof_pos = self.sit_target_dof_pos

            if self.sit_counter < 100:
                alpha = self.sit_counter / 100
                target_dof_pos = self.locomotion_target_dof_pos.copy()
                arm_pos = (
                    np.zeros_like(self.locomotion_actmap.lab_arm_offset) * (1-alpha)
                    + np.array(self.locomotion_actmap.lab_arm_offset) * alpha
                    )
                target_dof_pos[self.locomotion_actmap.mot_from_lab_upper_joints] = arm_pos
                self.sit_counter += 1

        # elif self.task == "to_eetrack_init":
        #     pass
        
        ################################# EETrack #################################
        elif self.task == "eetrack":

            if self.is_eetrack_first_iter:
                print("\n[EETrack] EETrack has began.")
                self.is_eetrack_first_iter = False
                self.eetrack_initial_counter = self.counter

            # if self.eetrack_command is None:
            #     self.eetrack_command = ue.eetrack(
            #         th.from_numpy(root_state_w)[None],
            #         self.tf_buffer,
            #         clock, to_start=False)

            # Keymap press -> changes is_initial_goal == False
            commanded_to_go_to_first_welding_ee_pose = self.remote_controller.button[KeyMap.start] == 1
            if commanded_to_go_to_first_welding_ee_pose:
                print("\n[EETrack] To welding line start Sampling has begun.")
                self.eetrack_command.is_initial_goal = False

            if self.remote_controller.button[KeyMap.F2] == 1: # F2 is F3 button in controller
                print("\n[EETrack] Subgoal Sampling has begun.")
                self.eetrack_command.is_initial_eetrack = False
                # self.is_go_start = False

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

            print(self.target_pose_w)
            print(self.target_pose_b)

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
                if True:
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
            # else:
            #     if self.eetrack_command.to_start:
            #         qj = np.zeros(29, dtype=np.float32)
            #         for i_mot in range(len(self.config.motor_joint)):
            #             qj[i_mot] = self.low_state.motor_state[i_mot].q
            #         target_dof_pos = self.sit_target_dof_pos.copy()
            #         target_dof_pos[-7:] = qj[-7:] + (self.last_sit_dof_pos[-7:] - qj[-7:]).clip(-0.01,0.01)
            #     else:
            #         if self.trajopt_joint_traj is None:
            #             trajopt_init_dof_pos = self.prev_target_dof_pos[-7:]
            #         else:
            #             trajopt_init_dof_pos = np.array(self.trajopt_joint_traj.points[0].positions)
            #         qj = np.zeros(29, dtype=np.float32)
            #         for i_mot in range(len(self.config.motor_joint)):
            #             qj[i_mot] = self.low_state.motor_state[i_mot].q
            #         target_dof_pos = self.sit_target_dof_pos.copy()
            #         target_dof_pos[-7:] = qj[-7:] + (trajopt_init_dof_pos - qj[-7:]).clip(-0.01,0.01)

            #         joint_pos_diff = np.linalg.norm(trajopt_init_dof_pos - qj[-7:])
            #         print("Joint pos diff: ", joint_pos_diff)
            #         if joint_pos_diff < 0.005:
            #             self.is_go_start = False
            #             print("is_go_start disabled!!!")
                
        kps = np.array(self.config.kps).astype(np.float32).copy()
        kds = np.array(self.config.kds).astype(np.float32).copy()
         
        if self.task == "eetrack":
            kps[-7:] = self.config.eetrack_right_arm_kps
            kds[-7:] = self.config.eetrack_right_arm_kds
            target_tau[-7:] = 0.0

        elif self.task == "sit":
            print(kds)
            kds = np.array(self.config.sit_kds).astype(np.float32).copy()

        for mot_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].dq = 0.0
            self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(kps[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(kds[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].tau = float(target_tau[mot_idx])
        
        # observation dumping
        # self.dump_observations_and_joint_pos_target(target_dof_pos)

         
        # send the command
        self.send_cmd(self.low_cmd)

    def dump_observations_and_joint_pos_target(self, target_dof_pos):
        # log timestamp
        timestamp_low_freq = clock.get_time().nanoseconds / 1e9
        self.timestamp_low_freq = np.append(self.timestamp_low_freq, timestamp_low_freq)
        
        self.locomotion_observations = np.vstack((self.locomotion_observations, self.obs))
        self.locomotion_actions = np.vstack((self.locomotion_actions, self.locomotion_action))
        self.locomotion_vel_traj = np.vstack((self.locomotion_vel_traj, self.locomotion_vel_command))

        
            
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
                print("[Navigation] Press Button {B} to switch mode into **Sit**.")
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
