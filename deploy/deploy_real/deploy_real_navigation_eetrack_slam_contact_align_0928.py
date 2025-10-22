import os
import torch as th
import numpy as np
import math_utils
from pathlib import Path
from typing import Union, Literal
from legged_gym import LEGGED_GYM_ROOT_DIR
from scipy.spatial.transform import Rotation as R

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo
from rclpy.duration import Duration

from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from pyroki_ros.action import TrajOptSingleEE

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
from std_msgs.msg import Bool, Empty

import utils_robot as ur
import utils_locomotion as ul
import utils_stage as us
import utils_eetrack_tag_contact_align as ue
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray
import subprocess

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
        rot_type: str = 'axa',
        return_stamp=False,):
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
        if return_stamp:
            return (xyz, axa, t.header.stamp)
        else:
            return (xyz, axa)
    elif rot_type == 'quat':
        if return_stamp:
            return (xyz, quat_wxyz, t.header.stamp)
        else:
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
    

def interpolate_array_by_float_index(array, index):
    # Get the lower and upper bounds
    left_idx = int(np.floor(index))
    right_idx = int(np.ceil(index))
    
    # Get the values at the left and right indices
    left_value = array[left_idx]
    right_value = array[right_idx]
    
    # Calculate the fractional distance between the left and right indices
    fraction = index - left_idx
    
    # Linear interpolation formula
    interpolated_value = left_value + (right_value - left_value) * fraction
    
    return interpolated_value


def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos


def interpolate_quaternion(quat1, quat2, n_segments):
    quat1 = th.from_numpy(quat1[None, None, ...])
    quat2 = quat2[None, ...]
    t = th.linspace(0, 1, n_segments + 1).view(1, -1, 1)
    interp_q = math_utils.slerp_vectorized(quat1, quat2, t)
    interp_q = interp_q[0]
    return interp_q


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

        self.tasks = np.zeros((0,1))
        
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
        self.root_states_w = np.zeros((0, 7))

        self.target_dof_poss = np.zeros((0,self.num_joints))

        self.target_poses_w = np.zeros((0, 7))
        self.target_poses_b = np.zeros((0, 7))
        self.ee_poses_w = np.zeros((0, 7))
        self.ee_poses_b = np.zeros((0, 7))

        self.zed_pose_w = np.zeros(7)
        self.zed_poses_w = np.zeros((0,7))

        self.trajopt_target_joint_pos_traj = np.zeros((0,29))
        self.trajopt_target_joint_pos = np.zeros(29)

        # counter
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer(cache_time=rp.duration.Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self._node)

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

        self.errors_avg_10 = np.zeros((0,2))
        self.errors_avg_20 = np.zeros((0,2))
        self.errors_avg_30 = np.zeros((0,2))
        self.errors_avg_40 = np.zeros((0,2))

        ########################## Sit ##########################
        if "sit_ver3" in self.config.sit_policy_path:
            self.sit_obsmap = us.SitObservation(config, self.tf_buffer)
        else:
            self.sit_obsmap = us.SitObservation_v2(config, self.tf_buffer)
            
        self.sit_robot = self.locomotion_robot
        self.sit_actmap = us.SitActionVer2(config, self.sit_robot)
        self.vhcommand = us.VelocityHeightCommand(config)
        self.sit_policy = th.jit.load(config.sit_policy_path) # FIXME sit_policy_path
        self.sit_policy.eval()

        self.sit_observations = np.zeros((1,92))

        self.zed_executed = False

        ########################## Navigation ##########################
        # NOTE (bk) is this where we set the position target for the navigation?
        self.navigation_pos_target = np.array([1.0, 0.0, 0.0])
        self.navigation_heading_target = 0.0 # radians
        self.pos_command_bs = np.zeros((0, 3))
        self.pos_command_b = np.zeros(3)
        self.pos_error_bs = np.zeros((1,1))
        self.heading_error_bs = np.zeros((1,1))
        self.pelvis_to_midsole_offset_after_locomotion = {
            "x": -0.0069,
            "y": -0.0194,
            "yaw": 0.0061,
        }

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


        self.bending_counter = 0
        self.zed_start_publisher = self._node.create_publisher(
            Empty,
            '/start_zed',
            10,
        )

        self.zed_stop_publisher = self._node.create_publisher(
            Empty,
            '/stop_zed',
            10,
        )

        # Publisher for /vision/eetrack
        self.eetrack_vision_trigger_publisher = self._node.create_publisher(
            Bool,
            'eetrack_vision/trigger_vision_pipeline',
            10,
        )
        # Subscribe to /eetrack_vision topic (MultiArrayLayout)
        self.start_eetrack_vision_callback = False
        self.eetrack_vision_subscriber = self._node.create_subscription(
            Float64MultiArray,
            'eetrack_vision/weldpoints',
            self.eetrack_vision_callback,
            10
        )
        self.welding_points_from_vision = None

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

        self.is_apriltag_detection_on = False
        self.target_vec_b_log = np.zeros((0, 4))
        self.target_vec_bs = np.zeros((0, 4))
        self.naviation_last_action = np.zeros(3, dtype=np.float32)
        self.locomotion_vel_commands = np.zeros((0, 3))

        self.qj = np.zeros(self.config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_actions, dtype=np.float32)

        self.zed_optical_frame = "zed2i_left_camera_optical_frame"

        ######################### eetrack #########################
        self.contact_align_target_point : Literal["start_point", "end_point"] = "start_point"
        self.contact_aligned_start_ee_pose = None
        self.contact_aligned_end_ee_pose = None

        self.eetrack_command = None
        self.is_eetrack_first_iter = True

        self.ee_z_down = None
        self.ee_z_up = None

        self.weld_dx = 0.0
        self.weld_dy = 0.0
        self.weld_dz = 0.0
        self.weld_dpitch = 0.0
        self.weld_dyaw = 0.0

        self.trajopt_i = 0
        self.trajopt_joint_traj = None
        self.mot_from_trajopt = None
        self._goal_handle = None
        self.execute_trajopt_joint_traj = False
        self._action_client = ActionClient(
            self._node,
            TrajOptSingleEE,
            'trajopt_single_ee',
        )
        self._node.get_logger().info("Waiting for trajopt action server...")
        self._action_client.wait_for_server()

        self.act_joint = config.ik_joint
        self.ikctrl = IKCtrl('../../resources/robots/g1_description/g1_29dof_rev_1_0_zed2i_with_welder_v3.urdf',
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
        self.prev_task = self.task
        self.task_counter = 0

        self.zero_phase = False

        self.sitting = False
        self._mode_change = True
        self._terminate = False

        # calls run_wrapper every self.config.control_dt seconds
        self._timer = self._node.create_timer(self.config.control_dt, self.run_wrapper)
        self.welding_object_pose_w = None
        self._tf_timer = self._node.create_timer(0.01, self.publish_welding_object_pose_callback)
        self.stop_time = None
        try:
            rp.spin(self._node)
        except KeyboardInterrupt:
            self.log_metrics_and_trajectories()
            print("Log saved.")
        finally:
            subprocess.run(["pkill", "-f", "align"])
            self._node.destroy_timer(self._timer)
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self._node.destroy_node()
            rp.shutdown()
            print("Exit")


    def trigger_vision_pipeline(self):
        msg = Bool()
        msg.data = True
        self.eetrack_vision_trigger_publisher.publish(msg)
        self.start_eetrack_vision_callback = True


    def process_msg_from_eetrack_vision(self, msg):
        N = None
        if msg.layout and msg.layout.dim and len(msg.layout.dim) >= 2:
            # Expect row-major: [rows, columns] with columns==3
            rows = msg.layout.dim[0].size
            cols = msg.layout.dim[1].size
            if cols == 3:
                N = rows

        data = np.asarray(msg.data, dtype=np.float64)
        if N is None:
            if data.size % 3 != 0:
                print(f"Received {data.size} values (not divisible by 3). Dropping.")
                return
            N = data.size // 3

        pts_zed = data.reshape(N, 3) # originally the points are in zed frame
        return pts_zed

    def get_zed_pose_wrt_world(self):
        # Get the camera to world frame
        while True:
            try:
                t, q = body_pose(self.tf_buffer, self.zed_optical_frame, "mid_sole_link", rot_type="quat")
                break
            except Exception as ex:
                print(ex)
                print("No zed tf wrt world exists")
        return t,q

    def apply_transform_to_points(self, points, t, q):
        # t=translation, q=quternion
        q = np.roll(q, -1) # wxyz -> xyzw
        R_wc = R.from_quat(q).as_matrix()  # SciPy expects [x, y, z, w]
        pts_world = (R_wc @ points.T).T + t[None, :]
        return pts_world

    def eetrack_vision_callback(self, msg: 'Float64MultiArray'):
        if not self.start_eetrack_vision_callback:
            return
        # -------- parse incoming [N,3] points from Float64MultiArray ----------
        data = np.asarray(msg.data, dtype=np.float64)
        if data.size == 0:
            return

        # points format: (2,3), where each point indicates the start and
        # end points on the welding line
        pts_zed = self.process_msg_from_eetrack_vision(msg)
        t, q = self.get_zed_pose_wrt_world()
        self.zed_pose_w = np.concatenate([t, q])
        pts_world = self.apply_transform_to_points(pts_zed, t, q)

        self.welding_points_from_vision = pts_world.copy()
        # Move start position to right, end position to left.
        self.welding_points_from_vision[0] = 0.8*pts_world[0] + 0.2*pts_world[-1]
        # self.welding_points_from_vision[-1] = 0.2*pts_world[0] + 0.8*pts_world[-1]
        self._node.get_logger().info(f"Welding points recieved: {self.welding_points_from_vision}")

        # Disable after recieve
        self.start_eetrack_vision_callback = False


    def compute_welding_object_pose_from_welding_line(
        self,
        welding_start_pos_w,
        welding_end_pos_w,
        dx_mid_to_object = 0.0025,  # dx of welding mid point to object pose
        dz_mid_to_object = 0.005,   # dz of welding mid point to object pose
    ):
        # Assume the point is in world frame.
        z_axis = np.array([0,0,1])
        # Assume the start point is in left (+y) and the end point is in right (-y)
        y_axis = welding_start_pos_w - welding_end_pos_w
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        # 
        welding_mid_pos_w = (welding_start_pos_w + welding_end_pos_w) / 2
        welding_object_pos_w = welding_mid_pos_w + dx_mid_to_object*x_axis - dz_mid_to_object*z_axis

        mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        sciR = R.from_matrix(mat)
        # xyzw -> wxyz
        welding_object_quat_w = np.roll(sciR.as_quat(), 1)

        return np.concatenate([welding_object_pos_w, welding_object_quat_w])

    
    def publish_welding_object_pose_callback(self):
        if self.welding_object_pose_w is None:
            return
        t = TransformStamped()

        # Format header
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = 'mid_sole_link'
        t.child_frame_id = 'welding_object'

        # Populate translation
        t.transform.translation.x = self.welding_object_pose_w[0]
        t.transform.translation.y = self.welding_object_pose_w[1]
        t.transform.translation.z = self.welding_object_pose_w[2]

        t.transform.rotation.w = self.welding_object_pose_w[3]
        t.transform.rotation.x = self.welding_object_pose_w[4]
        t.transform.rotation.y = self.welding_object_pose_w[5]
        t.transform.rotation.z = self.welding_object_pose_w[6]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    
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
        t.header.frame_id = "mid_sole_link" #'world'
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
    
    def publish_tf(self, header_frame, child_frame, pos, quat):
        t = TransformStamped()

        # Format header
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = header_frame #'world'
        t.child_frame_id = child_frame

        # Populate translation
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])

        # Set world_from_pelvis quaternion based on IMU state
        qw, qx, qy, qz = [float(x) for x in quat[3:7]]
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


    def send_ee_goal_to_trajopt(self, target_pos_b, target_quat_b):
        if self._goal_handle is not None and not self._goal_done():
            return
        self._node.get_logger().info('Start sending end effector trajectories...')

        stamp = self._node.get_clock().now().to_msg()
        frame_id = "pelvis"

        goal_msg = TrajOptSingleEE.Goal()
        goal_msg.ee_goal.header.stamp = stamp
        goal_msg.ee_goal.header.frame_id = frame_id

        goal_msg.ee_goal.pose.position.x = target_pos_b[0]
        goal_msg.ee_goal.pose.position.y = target_pos_b[1]
        goal_msg.ee_goal.pose.position.z = target_pos_b[2]

        goal_msg.ee_goal.pose.orientation.w = target_quat_b[0]
        goal_msg.ee_goal.pose.orientation.x = target_quat_b[1]
        goal_msg.ee_goal.pose.orientation.y = target_quat_b[2]
        goal_msg.ee_goal.pose.orientation.z = target_quat_b[3]

        self._node.get_logger().info("Sending goal end effector trajectories...")
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        self._goal_handle = future.result()

        if not self._goal_handle.accepted:
            self._node.get_logger().info('Goal rejected :(')
            return

        self._node.get_logger().info('Goal accepted :)')

        self._get_result_future = self._goal_handle.get_result_async()

        self._get_result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        result = future.result().result

        joint_traj = result.joint_traj
        success = result.success
        error_message = result.error_message

        if len(joint_traj.points) == 0:
            return

        self.trajopt_joint_traj = np.array([point.positions for point in joint_traj.points])
        self.mot_from_trajopt = index_map(self.config.motor_joint, joint_traj.joint_names)

        self._node.get_logger().info(f'TrajOpt result recieved!!')


    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        percent = feedback_msg.percent_complete
        self._node.get_logger().info(f'In progress: {percent:.1f}')


    def _goal_done(self):
        """Return True if goal is done (succeeded, canceled, or aborted)."""
        status = self._goal_handle.status
        return status in (
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        )
    

    def solve_ik(self, target_pose_b):
        # Get current joint positions
        qj = np.zeros(self.num_joints, dtype=np.float32)
        for i_mot in range(len(self.config.motor_joint)):
            i_pin = self.pin_from_mot[i_mot]
            qj[i_pin] = self.low_state.motor_state[i_mot].q

        gravity_vec = 9.81*self.sit_obsmap._projected_gravity_from_lowstate(self.low_state)
        res_q, arm_nle = self.ikctrl(qj,
                                    target_pose_b,
                                    rel=False,
                                    gravity_vec=gravity_vec,
                                    )
        res_q = 2*res_q

        target_dof_pos = self.sit_target_dof_pos.copy()
        target_tau = np.zeros(self.num_joints, dtype=np.float32)
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

        return target_dof_pos, target_tau

    def get_locomotion_command(self):
        self.stop_locomotion = False
        self.navigation_counter = 0
        # NOTE (bk): what's this if statement? needs to be re-written
        if np.any(
            np.abs(np.array([self.remote_controller.ly, self.remote_controller.lx, self.remote_controller.rx])) > 0.
        ) or self.minimum_locomotion_iter != 0:

            if self.minimum_locomotion_iter == 0:
                self.minimum_locomotion_iter = 100
            self.minimum_locomotion_iter -= 1

            v_x = np.clip(self.remote_controller.ly, -0.25, 0.1)
            v_y = np.clip(self.remote_controller.lx * -1, -0.16, 0.16)
            v_z = np.clip(self.remote_controller.rx * -1, -0.4, 0.4)

            self.locomotion_vel_command[0] = v_x
            self.locomotion_vel_command[1] = v_y
            self.locomotion_vel_command[2] = v_z

            phase = (self.locomotion_counter * 0.02) % 1.0 / 1.0
            self.locomotion_counter += 1

        else:
            self.locomotion_vel_command = np.zeros(3)
            phase = self.locomotion_counter = 0
        
        return phase
    
    def get_navigation_command(self, xyz, quat_wxyz):
        phase = (self.counter * 0.02) % 1.0 / 1.0
        
        ###################### Compute state ######################
        # Pelvis heading direction
        forward_w = quat_apply(quat_wxyz.astype(np.float32), np.array([1., 0., 0.]).astype(np.float32))
        pelvis_heading_w = np.arctan2(forward_w[1], forward_w[0])

        heading_error = wrap_to_pi(np.array([self.pelvis_heading_target]).astype(np.float32) - np.array([pelvis_heading_w]).astype(np.float32))

        target_vec = self.pelvis_pos_target - xyz
        target_vec[2] = 0.0
        self.pos_command_b = quat_rotate_inverse(yaw_quat(quat_wxyz).astype(np.float32), target_vec.astype(np.float32))

        self.pos_error_bs = np.vstack((self.pos_error_bs, np.linalg.norm(self.pos_command_b[:2]).reshape((1,1))))
        self.heading_error_bs = np.vstack((self.heading_error_bs, heading_error.reshape(1,1)))

        if self.pos_error_bs.shape[0] > 50:
            for window_size in [10, 20, 30, 40]:
                translational_error = np.mean(self.pos_error_bs[-window_size:, :].reshape((window_size,)))
                rotational_error = np.mean(np.abs(self.heading_error_bs[-window_size:, :]).reshape((window_size,)))
                errors = np.array([translational_error, rotational_error])

                setattr(self, f"errors_avg_{window_size}", np.vstack((
                    getattr(self, f"errors_avg_{window_size}"),
                    errors
                )))
        # print("Pelvis height :", xyz[-1])
        print("Pelvis <> Target distance in local frame : ", self.pos_command_b)
        ###################### Hand design navigation ######################
        if True:
            if self.pos_error_bs.shape[0] > self.NUM_AVG:
                # print("pos command mean : ", np.mean(self.pos_error_bs[-self.NUM_AVG:, :].reshape((self.NUM_AVG,))))
                # print("heading error mean : ", np.mean(np.abs(self.heading_error_bs[-self.NUM_AVG:, :]).reshape((self.NUM_AVG,))))
                if np.mean(self.pos_error_bs[-self.NUM_AVG:, :].reshape((self.NUM_AVG,))) < self.ERROR_THRESHOLD \
                    and np.mean(np.abs(self.heading_error_bs[-self.NUM_AVG:, :]).reshape((self.NUM_AVG,))) < 0.1:
                    if self.stop_locomotion is not True :
                        self.stop_time = (self.counter-400) * self.config.control_dt
                    self.stop_locomotion = True

            if self.counter % 10 == 0:
                # self.locomotion_vel_command[:2] = np.clip(np.sign(pos_command_b[:2]) * MAX_LIN_VEL * np.sqrt(np.abs(pos_command_b[:2] / SLOW_BOUND)), -MAX_LIN_VEL, MAX_LIN_VEL)
                # X >= 0
                # X >= 0
                if self.pos_command_b[0] >= 0:
                    self.locomotion_vel_command[0] = np.clip(0.1 * np.sqrt(np.abs(self.pos_command_b[0] / 0.6)), 0., 0.1)
                # X < 0
                if self.pos_command_b[0] < 0:
                    self.locomotion_vel_command[0] = np.clip(-0.3 * np.sqrt(np.abs(self.pos_command_b[0] / 0.2)), -0.3, 0.)
                # Y >= 0
                if self.pos_command_b[1] >= 0:
                    self.locomotion_vel_command[1] = np.clip(0.1 * np.sqrt(np.abs(self.pos_command_b[1] / 0.3)), 0., 0.1)
                # Y < 0
                if self.pos_command_b[1] < 0:
                    self.locomotion_vel_command[1] = np.clip(-0.12 * np.sqrt(np.abs(self.pos_command_b[1] / 0.3)), -0.12, 0.)
                
                self.locomotion_vel_command[2] = np.clip(np.sign(heading_error) * 0.3 * np.sqrt(np.abs(heading_error / 0.2)), -0.3, 0.3)

                # vel_norm = np.linalg.norm(self.locomotion_vel_command)
                # if vel_norm < 0.2:
                #     self.locomotion_vel_command = 0.2 * self.locomotion_vel_command / vel_norm



            if self.stop_locomotion:
                self.locomotion_vel_command = np.array([0., 0., 0.])
                phase = 0.0

        if self.navigation_counter < 100:
            # vel cmd smoothing
            alpha = self.navigation_counter / 100
            self.locomotion_vel_command = np.zeros(3) * (1-alpha) + self.locomotion_vel_command * alpha
        
            self.navigation_counter += 1

        return phase

    def run_sit_policy(self, xyz):
        if self.remote_controller.button[KeyMap.down] == 1:
            self.sitting = True
        if self.remote_controller.button[KeyMap.up] == 1:
            self.sitting = False

        height_command = self.vhcommand(current_pelvis_height_w = xyz[2] + 0.00, sitting=self.sitting)

        # For stage 1 & 2.
        if self.config.use_interpolation:
            interpolation_length = 200 # 4s
            if (self.sit_counter > 100 and self.sit_counter < 100 + interpolation_length) \
                 and \
                ("sit_ver3" not in self.config.sit_policy_path):        
                alpha = (self.sit_counter - 100) / interpolation_length
                # hip_pitch_offset = np.array([-4.8070e-01 + 0.1, -3.1852e-01 + 0.1])*(1-alpha) + np.array([-4.8070e-01, -3.1852e-01])*alpha
                hip_pitch_offset = np.array([-4.8070e-01 + 0.2, -3.1852e-01 + 0.2])*(1-alpha) + np.array([-4.8070e-01, -3.1852e-01])*alpha
                ankle_pitch_offset = np.array([-2.3876e-01 + 0.2, -4.7379e-01 + 0.2])*(1-alpha) + np.array([-2.3876e-01, -4.7379e-01])*alpha
                if self.config.is_downhill:
                    # on downhill, we have to bend ankle pitch more on the transition phase
                    self.obs = self.sit_obsmap(self.low_state, height_command, xyz, hip_pitch_offset, ankle_pitch_offset)
                else:
                    self.obs = self.sit_obsmap(self.low_state, height_command, xyz, hip_pitch_offset)
            else:
                self.obs = self.sit_obsmap(self.low_state, height_command, xyz)
        else:
            self.obs = self.sit_obsmap(self.low_state, height_command, xyz)
            if self.config.is_downhill:
                interpolation_length = 200 # 4s
                if (self.sit_counter > 100 and self.sit_counter < 100 + interpolation_length) \
                    and \
                    ("sit_ver3" not in self.config.sit_policy_path):        
                    alpha = (self.sit_counter - 100) / interpolation_length
                    ankle_pitch_offset = np.array([-2.3876e-01 + 0.2, -4.7379e-01 + 0.2])*(1-alpha) + np.array([-2.3876e-01, -4.7379e-01])*alpha
                    # on downhill, we have to bend ankle pitch more on the transition phase
                    self.obs = self.sit_obsmap(self.low_state, height_command, xyz, ankle_pitch_joint_offset=ankle_pitch_offset)

        self.sit_obs = self.obs.copy()

        obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone().float()
        self.sit_action = self.sit_policy(obs_tensor).detach().numpy().squeeze()

        # target_dof_pos : motor joint ordered
        self.sit_target_dof_pos = self.sit_actmap(self.sit_action)

        if self.sit_counter < 100:
            alpha = self.sit_counter / 100
            self.run_locomotion_policy(phase=0.0)

            self.sit_target_dof_pos = self.locomotion_target_dof_pos
            arm_pos = (
                np.zeros_like(self.locomotion_actmap.lab_arm_offset) * (1-alpha)
                + np.array(self.locomotion_actmap.lab_arm_offset) * alpha
                )
            self.sit_target_dof_pos[self.locomotion_actmap.mot_from_lab_upper_joints] = arm_pos
        self.sit_counter += 1
        # elif self.sit_counter < 200:

        #     self.run_locomotion_policy(phase=0.0)
        #     self.sit_target_dof_pos = self.locomotion_target_dof_pos
        #     arm_pos = np.array(self.locomotion_actmap.lab_arm_offset)
        #     self.sit_target_dof_pos[self.locomotion_actmap.mot_from_lab_upper_joints] = arm_pos
        #     self.sit_counter += 1
        
    def run_locomotion_policy(self, phase):
        self.obs = self.locomotion_obsmap(self.low_state, self.locomotion_vel_command, phase, last_action=self.locomotion_last_action)
        obs_tensor = th.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone().float()
        self.locomotion_action = self.locomotion_policy(obs_tensor).detach().numpy().squeeze()
        self.locomotion_last_action = self.locomotion_action

        self.locomotion_target_dof_pos = self.locomotion_actmap(self.locomotion_action)
    
    def get_pelvis_from_world(self):
        if self.task in ["locomotion", "navigation"]:
            try:
                world_from_pelvis = body_pose(
                    self.tf_buffer,
                    'pelvis',
                    'world',
                    rot_type='quat',
                    # stamp=rp.time.Time()
                )
                print(world_from_pelvis)
            except:
                try:
                    p_a, q_a = body_pose(
                        self.tf_buffer,
                        'camera_init',
                        'world',
                        rot_type='quat',
                        # stamp=rp.time.Time()
                    )
                except Exception as e:
                    print(e)
                try:
                    p_b, q_b = body_pose(
                        self.tf_buffer,
                        'body',
                        'camera_init',
                        rot_type='quat',
                        # stamp=rp.time.Time()
                    )
                except Exception as e:
                    print(e)
                try:
                    p_c, q_c = body_pose(
                        self.tf_buffer,
                        'body_z_from_mid_sole_link',
                        'body',
                        rot_type='quat',
                        # stamp=rp.time.Time()
                    )
                except Exception as e:
                    print(e)
                try:
                    p_d, q_d = body_pose(
                        self.tf_buffer,
                        'pelvis',
                        'body_z_from_mid_sole_link',
                        rot_type='quat',
                        # stamp=rp.time.Time()
                    )
                except Exception as e:
                    print(e)
                
                p_a_b, q_a_b = combine_frame_transforms(
                    p_a, q_a,
                    p_b, q_b
                )
                p_a_c, q_a_c = combine_frame_transforms(
                    p_a_b, q_a_b,
                    p_c, q_c
                )
                p_a_d, q_a_d = combine_frame_transforms(
                    p_a_c, q_a_c,
                    p_d, q_d
                )
                world_from_pelvis = (p_a_d, q_a_d)
                print(world_from_pelvis)

        else:
            world_from_pelvis = body_pose(
                self.tf_buffer,
                'pelvis',
                'mid_sole_link',
                rot_type='quat',
                # stamp=rp.time.Time()
            )

        return world_from_pelvis
    


    ############################# MAIN LOOP #############################
    def run_policy(self):
        # If the button A is pressed, then finish the policy.
        if self.remote_controller.button[KeyMap.A] == 1:
            self._mode_change = True
            self.mode = Mode.finish
            return

        self.counter += 1

        world_from_pelvis = self.get_pelvis_from_world()

        xyz, quat_wxyz = world_from_pelvis
        self.root_state_w = np.zeros(7)
        self.root_state_w[0:3] = xyz
        self.root_state_w[3:7] = quat_wxyz

        # Add termination condition.
        if self.terminate_by_pelvis_condition(xyz, quat_wxyz):
            raise ValueError("Terminated by pelvis condition.")
        

        phase = (self.counter * 0.02) % 1.0 / 1.0

        ############################################ SWITCH MODE FROM LOCOMOTION TO NAVIVATION ############################################
        if self.remote_controller.button[KeyMap.Y] == 1 and (self.task not in  ["eetrack", "sit"] or not self.sitting):
            print("============== locomotion mode activated ==============")
            self.task = "locomotion" 
        if self.remote_controller.button[KeyMap.X] == 1 and (self.task not in  ["eetrack", "sit"] or not self.sitting):
            print("============== navigation mode activated ==============")
            self.task = "navigation"
            self.zed_stop_publisher.publish(Empty())

        if self.remote_controller.button[KeyMap.B] == 1 and self.task != "eetrack":
            print("============== Sitting mode activated ==============")
            self.task = "sit"
            subprocess.run(["pkill", "fastlio"])
            subprocess.run(["pkill", "livox"])
            
            self.zed_start_publisher.publish(Empty())

        if self.remote_controller.button[KeyMap.select] == 1 :
            if self.task == "sit" or self.task == "vision":
                print("============== Trigger vision pipeline ==============")
                # self.trigger_vision_pipeline()
                self.task = "vision"
                self.vision_start_counter = int(self.counter)
            elif self.task in ["navigation", "locomotion"]:
                print("============== Trigger apriltag detection ==============")
                if not self.is_apriltag_detection_on:
                    subprocess.Popen([
                        "python3",
                        "/root/localization/align_publisher_zed.py"
                        ],
                    stdout=subprocess.DEVNULL,   # discard standard output
                    stderr=subprocess.DEVNULL,   # discard error output
                    stdin=subprocess.DEVNULL     # detach from terminal input
                    )
                    self.is_apriltag_detection_on = True
                


        # Change to trajopt task only one the welding points are received.
        if self.remote_controller.button[KeyMap.R1] == 1 and self.welding_points_from_vision is not None and self.task == "vision":
            print("============== TrajOpt mode activated ==============")
            self.task = "trajopt"

            self.trajopt_i = 0
            self.trajopt_joint_traj = None
            self.execute_trajopt_joint_traj = False

            welding_start_pos_w = self.welding_points_from_vision[0]
            welding_end_pos_w = self.welding_points_from_vision[-1]
            eetrack_start_pos_w, eetrack_start_quat_w, _, _ \
                = ue.eetrack.get_eetrack_pos_quat(
                welding_start_pos_w,
                welding_end_pos_w,
                offset_len=0.06,
                approach_deg=45.0,
            )
            eetrack_start_pos_b, eetrack_start_quat_b = subtract_frame_transforms(
                world_from_pelvis[0],
                world_from_pelvis[1],
                eetrack_start_pos_w,
                eetrack_start_quat_w,
            )
            self.send_ee_goal_to_trajopt(eetrack_start_pos_b, eetrack_start_quat_b)
            
        if self.remote_controller.button[KeyMap.F1] == 1:
            if self.task == "trajopt":
                # Start going to welding start position from vision
                print("============== To Start mode activated ==============")
                self.task = "to_start"

                self.eetrack_command = ue.eetrack(
                    th.from_numpy(self.root_state_w)[None],
                    self.tf_buffer,
                    clock,
                    eetrack_vel=0.01,
                    start_pos_w=self.welding_points_from_vision[0],
                    end_pos_w=self.welding_points_from_vision[-1],
                    to_start=True
                )

        if self.remote_controller.button[KeyMap.L1] == 1 and (self.task == "eetrack" or self.task == "to_start"):
            # Perform contact align mode. Going to x axis direction (torch direction) of end-effector
            # TODO: also do this in the end position of welding line.
            print("============== Contact Align mode activated ==============")
            if self.task == "eetrack":
                self.contact_align_target_point = "end_point"
            else:
                self.contact_align_target_point = "start_point"

            self.task = "contact_align"

            self.prev_pos_ee_w = np.zeros(3)
            self.prev_quat_ee_w = np.array([1,0,0,0])
            self.prev_ee_stamp = self._node.get_clock().now().to_msg()

            # Move end-effector to z-up axis of EE.
            # self.ee_z_down = True
            # Move end-effector to z-down axis of EE.
            # self.ee_z_up = False
            # Move end-effector to mid position of contacted up and down position.
            # self.ee_z_mid = False

            # Move end-effector to x-axis direction of EE.
            self.ee_x_up = True

            self.contact_align_start_pos_ee_w, self.contact_align_start_quat_ee_w = body_pose(
                self.tf_buffer,
                frame="end_effector",
                ref_frame="mid_sole_link",
                rot_type='quat'
            )
            # self.contact_align_dz = 0.0

        if self.remote_controller.button[KeyMap.F2] == 1 and (self.task == "contact_align" or self.task == "to_start"):
        # if self.remote_controller.button[KeyMap.F2] == 1 and self.task == "to_start":
            print("============== eetrack mode activated ==============")
            self.task = "eetrack"
            print("contact target mode: ", self.contact_align_target_point)
            # TODO We should have an assertion to prevent self.welding_points_from_vision being None
            offset_from_ee_to_welding_object_when_fully_contacted = 0.0075
            offset_from_ee_to_welding_object_on_z_axis = -0.002

            start_x_offset_b = 0.0085 #0.0075
            start_z_offset_b = 0.0
            start_z_offset_w = -0.008 # -0.008

            end_x_offset_b = 0.0085 #0.0075
            end_z_offset_b = 0.0
            end_z_offset_w = start_z_offset_w

            if self.contact_align_target_point == "start_point":
                start_pos_w = (
                    self.contact_aligned_start_ee_pose[0] + 
                    start_x_offset_b*
                    matrix_from_quat(self.contact_aligned_start_ee_pose[1])[:3,0] +
                    # Add z-directional offset
                    start_z_offset_b *
                    matrix_from_quat(self.contact_aligned_start_ee_pose[1])[:3,2] +
                    # Add z-directional offset on world
                    np.array([0., 0., start_z_offset_w])
                )
                # x_offset_on_vision_point = -0.005
                # self.welding_points_from_vision[-1][0] += x_offset_on_vision_point
                end_pos_w = self.welding_points_from_vision[-1]
                inverse_y = False
                eetrack_vel = 0.01

            elif self.contact_align_target_point == "end_point":
                start_pos_w = (
                    self.contact_aligned_end_ee_pose[0] + 
                    # Add x-directional offset
                    end_x_offset_b *
                    matrix_from_quat(self.contact_aligned_end_ee_pose[1])[:3,0] +
                    # Add z-directional offset
                    end_z_offset_b *
                    matrix_from_quat(self.contact_aligned_end_ee_pose[1])[:3,2] +
                    # Add z-directional offset on world
                    np.array([0., 0., end_z_offset_w])
                )
                
                end_pos_w = (
                    self.contact_aligned_start_ee_pose[0] + 
                    # Add x-directional offset
                    start_x_offset_b *
                    matrix_from_quat(self.contact_aligned_start_ee_pose[1])[:3,0] +
                    # Add z-directional offset
                    start_z_offset_b *
                    matrix_from_quat(self.contact_aligned_start_ee_pose[1])[:3,2] +
                    # Add z-directional offset on world
                    np.array([0., 0., start_z_offset_w])
                )
                inverse_y = True
                eetrack_vel = 0.005

            self.eetrack_command = ue.eetrack(
                th.from_numpy(self.root_state_w)[None],
                self.tf_buffer,
                clock,
                eetrack_vel=eetrack_vel,
                # start_pos_w=self.welding_points_from_vision[0],
                # use prev_pos_ee_w from the last contact-align as the welding start position.
                # start position of welding line on welding object.
                start_pos_w=start_pos_w,
                # end position of welding line on welding object.
                end_pos_w=end_pos_w,
                to_start=False,
                inverse_y=inverse_y
            )

        if self.is_apriltag_detection_on and self.task in ["locomotion", "navigation"]:
            try:
                nav_target_pos , nav_target_axa = body_pose(
                    self.tf_buffer,
                    'nav_target',
                    'world',
                    rot_type='axa',
                    stamp=rp.time.Time()
                )

                self.pelvis_pos_target = nav_target_pos
                self.pelvis_heading_target = nav_target_axa[-1]
                print("Navigation target is detected !")
            except:
                print("Waiting for navigation target ...")

        ############################################ LOCOMOTION ############################################
        if self.task == "locomotion":
            phase = self.get_locomotion_command()
        
        ############################################ NAVIGATION ############################################
        elif self.task == "navigation":
            phase = self.get_navigation_command(xyz, quat_wxyz)
        
        # phase = 0.0
        target_tau = np.zeros(29, dtype=np.float32)
        if self.task in ["locomotion", "navigation"]:
            self.run_locomotion_policy(phase)
            target_dof_pos = self.locomotion_target_dof_pos


        elif self.task == "sit":
            self.run_sit_policy(xyz)
            target_dof_pos = self.sit_target_dof_pos

        elif self.task == "vision":
            target_dof_pos = self.sit_target_dof_pos.copy()

            if (self.counter - self.vision_start_counter) == 250:
                self.trigger_vision_pipeline()

            if self.welding_points_from_vision is not None:
                # Publish welding object pose
                welding_start_pos_w = self.welding_points_from_vision[0]
                welding_end_pos_w = self.welding_points_from_vision[-1]
                self.welding_object_pose_w = self.compute_welding_object_pose_from_welding_line(
                    welding_start_pos_w,
                    welding_end_pos_w,
                )

        elif self.task == "trajopt":
            target_dof_pos = self.sit_target_dof_pos.copy()

            if self.remote_controller.button[KeyMap.start] == 1:
                self.execute_trajopt_joint_traj = True
                self._node.get_logger().info("Start executing trajopt joint trajectories!!")

            if self.trajopt_joint_traj is not None and self.execute_trajopt_joint_traj:
                # Get current joint positions
                qj = np.zeros(29, dtype=np.float32)
                for i_mot in range(len(self.config.motor_joint)):
                    qj[i_mot] = self.low_state.motor_state[i_mot].q
                # Target joint pos
                target_q = np.zeros(29)
                self.trajopt_target_joint_pos = interpolate_array_by_float_index(self.trajopt_joint_traj, self.trajopt_i/5).copy()
                target_q[self.mot_from_trajopt] = self.trajopt_target_joint_pos
                res_q = target_q[-7:] - qj[-7:]
                res_q = res_q.clip(-0.05, 0.05)
                self.trajopt_i += 1
                self.trajopt_i = min(self.trajopt_i, 5*(len(self.trajopt_joint_traj)-1))
                # Gravity compensation
                gravity_vec = 9.81*self.sit_obsmap._projected_gravity_from_lowstate(self.low_state)
                arm_nle = self.ikctrl.get_gravity_compensation(qj, gravity_vec=gravity_vec)
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
                if self.trajopt_i == (5*(len(self.trajopt_joint_traj)-1)):
                    self._node.get_logger().info("Reached end of TrajOpt trajectory.")


        elif self.task == "contact_align":
            pos_ee_w, quat_ee_w, ee_stamp = body_pose(
                self.tf_buffer,
                frame="end_effector",
                ref_frame="mid_sole_link",
                rot_type='quat',
                return_stamp=True
            )

            x_axis_w = matrix_from_quat(self.contact_align_start_quat_ee_w)[:3,0]
            if self.ee_x_up:
                # TODO: incease target_dx (smaller than 0.001)
                target_dx = 0.0004 # How much the target far from the current EE
                target_pos_w = pos_ee_w + target_dx*x_axis_w
                target_quat_w = self.contact_align_start_quat_ee_w
                
                #interpolate_quaternion(
                #    quat_ee_w,
                #    th.from_numpy(self.contact_align_start_quat_ee_w)[None],
                #    5,
                #)[1].cpu().numpy()

            self.target_pose_w = np.concatenate([target_pos_w, target_quat_w], axis=-1)
            target_pos_b, target_quat_b = subtract_frame_transforms(
                self.root_state_w[:3],
                self.root_state_w[3:],
                target_pos_w,
                target_quat_w,
            )
            self.target_pose_b = np.concatenate([target_pos_b, target_quat_b], axis=-1)

            self.publish_hand_target()

            target_dof_pos, target_tau = self.solve_ik(self.target_pose_b)

            # When F2 (or F3) pressed, it passed to the eetrack_command at "eetrack" task.
            self.prev_pos_ee_w = pos_ee_w.copy()
            self.prev_quat_ee_w = quat_ee_w.copy()
            self.prev_ee_stamp = ee_stamp
            if self.contact_align_target_point == "start_point":
                self.contact_aligned_start_ee_pose = (self.prev_pos_ee_w, self.prev_quat_ee_w)
            elif self.contact_align_target_point == "end_point":
                self.contact_aligned_end_ee_pose = (self.prev_pos_ee_w, self.prev_quat_ee_w)


        ################################# EETrack #################################
        elif self.task == "to_start" or self.task == "eetrack":

            if self.is_eetrack_first_iter:
                print("\n[EETrack] EETrack has began.")
                self.is_eetrack_first_iter = False
                self.eetrack_initial_counter = self.counter

            _ = self.eetrack_command.get_command(
                th.from_numpy(self.root_state_w)[None]
                )[0].detach().cpu().numpy()
            
            self.target_pose_w = np.copy(
                self.eetrack_command.next_command_s_left.squeeze().detach().cpu().numpy()
            )
            self.target_pose_b = np.concatenate([
                self.eetrack_command.lerp_command_b_left_pos.squeeze().detach().cpu().numpy(),
                self.eetrack_command.lerp_command_b_left_quat.squeeze().detach().cpu().numpy(),
            ])

            self.publish_hand_target()

            target_dof_pos, target_tau = self.solve_ik(self.target_pose_b)

            # # Get current joint positions
            # qj = np.zeros(29, dtype=np.float32)
            # for i_mot in range(len(self.config.motor_joint)):
            #     i_pin = self.pin_from_mot[i_mot]
            #     qj[i_pin] = self.low_state.motor_state[i_mot].q

            # gravity_vec = 9.81*self.sit_obsmap._projected_gravity_from_lowstate(self.low_state)
            # res_q, arm_nle = self.ikctrl(qj,
            #                             self.target_pose_b,
            #                             rel=False,
            #                             gravity_vec=gravity_vec,
            #                             )
            # res_q = 2*res_q

            # target_dof_pos = self.sit_target_dof_pos.copy()
            # for i_act in range(len(res_q)):
            #     i_mot = self.mot_from_act[i_act]
            #     i_pin = self.pin_from_mot[i_mot]
            #     target_q_i = (
            #             self.low_state.motor_state[i_mot].q + res_q[i_act]
            #     )
            #     target_q_i = np.clip(target_q_i,
            #                     self.lim_lo_pin[i_pin],
            #                     self.lim_hi_pin[i_pin])
            #     target_dof_pos[i_mot] = target_q_i
            #     target_tau[i_mot] = arm_nle[i_act]

        kps = np.array(self.config.kps).astype(np.float32).copy()
        kds = np.array(self.config.kds).astype(np.float32).copy()

        if self.task == "vision" or self.task == "trajopt" or self.task == "to_start" or self.task == "contact_align" or self.task == "eetrack":
            kps = np.array(self.config.kps).astype(np.float32).copy()
            kds = np.array(self.config.kds).astype(np.float32).copy()

            kps[-7:] = self.config.eetrack_right_arm_kps
            kds[-7:] = self.config.eetrack_right_arm_kds

            ################# change lower body Kp, Kd values due to overheating #################
            if "sit_ver3" not in self.config.sit_policy_path:
                kps[:15] = self.config.kps[:15]
                kds[:15] = self.config.kds[:15]
            # else:
            #     kps[:15] = self.config.eetrack_lower_body_kps
            #     kds[:15] = self.config.eetrack_lower_body_kds

        elif self.task == "sit":
            if "sit_ver3" in self.config.sit_policy_path :
                if self.sit_counter < 100:
                    kps = np.array(self.config.kps).astype(np.float32).copy()
                else:
                    kps = np.array(self.config.sit_kps).astype(np.float32).copy()
                kds = np.array(self.config.sit_kds).astype(np.float32).copy()
            else:
                kps = np.array(self.config.kps).astype(np.float32).copy()
                kds = np.array(self.config.sit_kds).astype(np.float32).copy()
                # if self.sit_counter > 100 and self.sit_counter < 200:
                #     alpha = self.sit_counter / 100
                #     high_kps = np.array(self.config.kps).astype(np.float32).copy()
                #     low_kps = np.array(self.config.sit_kps).astype(np.float32).copy()
                #     kps[4:6] = alpha * high_kps[4:6] + (1-alpha) * low_kps[4:6]
                #     kps[10:12] = alpha * high_kps[10:12] + (1-alpha) * low_kps[10:12]

            if True:
                if self.prev_joint_pos_target is not None:
                    target_dof_pos = self.config.sit_smoothing * target_dof_pos + \
                                    (1-self.config.sit_smoothing) * self.prev_joint_pos_target

                self.prev_joint_pos_target = target_dof_pos

        for mot_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].dq = 0.0
            self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(kps[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(kds[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].tau = float(target_tau[mot_idx])
        
        # observation dumping
        self.dump_observations_and_joint_pos_target(target_dof_pos)

         
        # send the command
        self.send_cmd(self.low_cmd)
        self.prev_task = self.task

    def dump_observations_and_joint_pos_target(self, target_dof_pos):
        # log timestamp
        timestamp_low_freq = clock.get_time().nanoseconds / 1e9
        self.timestamp_low_freq = np.append(self.timestamp_low_freq, timestamp_low_freq)

        self.tasks = np.vstack((self.tasks, self.task))
        
        # self.locomotion_observations = np.vstack((self.locomotion_observations, self.obs))
        # self.locomotion_actions = np.vstack((self.locomotion_actions, self.locomotion_action))
        self.locomotion_vel_traj = np.vstack((self.locomotion_vel_traj, self.locomotion_vel_command))
        
        self.root_states_w = np.vstack((self.root_states_w, self.root_state_w))

        self.target_dof_poss = np.vstack((self.target_dof_poss, target_dof_pos))

        if self.task == "sit":
            # pass
            # if self.sit_counter >= 100:
                self.sit_observations = np.vstack((self.sit_observations, self.sit_obs))
        elif self.task == "vision":
            self.zed_poses_w = np.vstack((self.zed_poses_w, self.zed_pose_w))
        elif self.task =="trajopt":
            if self.trajopt_joint_traj is None:
                self.trajopt_target_joint_pos_traj = np.vstack((self.trajopt_target_joint_pos_traj, np.zeros_like(self.trajopt_target_joint_pos)))
            else:
                self.trajopt_target_joint_pos_traj = np.vstack((self.trajopt_target_joint_pos_traj, self.trajopt_target_joint_pos))
        elif self.task == "to_start" or self.task == "contact_align" or self.task == "eetrack":
            self.target_poses_w = np.vstack((self.target_poses_w, self.target_pose_w))
            self.target_poses_b = np.vstack((self.target_poses_b, self.target_pose_b))
            ee_pos_w, ee_quat_w = body_pose(self.tf_buffer, "end_effector", "mid_sole_link", rot_type="quat")
            self.ee_poses_w = np.vstack((self.ee_poses_w, np.concatenate((ee_pos_w, ee_quat_w))))
            ee_pos_b, ee_quat_b = body_pose(self.tf_buffer, "end_effector", "pelvis", rot_type="quat")
            self.ee_poses_b = np.vstack((self.ee_poses_b, np.concatenate((ee_pos_b, ee_quat_b))))

        elif self.task == "navigation":
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
            "sit_observations": self.sit_observations,
            "tasks": self.tasks,
            "root_states_w" : self.root_states_w,
            "target_dof_pos": self.target_dof_poss,
            "zed_poses_w": self.zed_poses_w,
            "target_poses_w": self.target_poses_w,
            "target_poses_b": self.target_poses_b,
            "ee_poses_w": self.ee_poses_w,
            "ee_poses_b": self.ee_poses_b,
            "target_trajopt_joint_pos": self.trajopt_target_joint_pos_traj,
            "locomotion_vel_cmd": self.locomotion_vel_traj,
            "pos_command_bs": self.pos_command_bs,

            # nav error
            "errors_avg_10" : self.errors_avg_10,
            "errors_avg_20" : self.errors_avg_20,
            "errors_avg_30" : self.errors_avg_30,
            "errors_avg_40" : self.errors_avg_40,
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
