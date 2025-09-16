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
from config_test_hand_delta_using_qr_tag import Config
from common.crc import CRC
from enum import Enum
from ikctrl import IKCtrl
from std_msgs.msg import Bool

import utils_robot as ur
import utils_locomotion as ul
import utils_stage as us
import utils_eetrack_tag as ue
import rclpy
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray

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

# returns a tuple containing the position and orientation of frame 2 w.r.t. frame 1.
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

    
def get_frame_from_tf_buffer_wrt_ref_frame(
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
        t = tf_buffer.lookup_transform(
            ref_frame,  # to
            frame,  # from
            stamp)
    except TransformException as ex:
        return None, None
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
        # counter
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)

        self.tf_broadcaster = TransformBroadcaster(self._node)

        self.act_joint = config.ik_joint
        #TODO needs to use the same urdf as the state publisher
        self.ikctrl = IKCtrl('../../resources/robots/g1_description/',
                             self.act_joint,
                             frame='end_effector') # end_effector = end of torch
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit
        self.trajopt_data = np.load("trajopt_result_2.npz")
        self.trajopt_i = 0
        
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
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        else:
            raise ValueError("Invalid msg_type")


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


    def apply_transform_to_points(self, points, t, q):
        # t=translation, q=quternion
        q = np.roll(q, -1) # wxyz -> xyzw
        R_wc = R.from_quat(q).as_matrix()  # SciPy expects [x, y, z, w]
        pts_world = (R_wc @ points.T).T + t[None, :]
        return pts_world
    
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
            print("In zero torque state")
            self._mode_change = True
            self.mode = Mode.damping
        else:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)


    def move_to_default_pos(self):
        for motor_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[motor_idx].q = self.config.locomotion_motor_joint_offsets[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].dq = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = float(self.config.kps[motor_idx])
            self.low_cmd.motor_cmd[motor_idx].kd = float(self.config.kds[motor_idx])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0
        self.send_cmd(self.low_cmd)



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
    
    def get_hand_pose_from_qr_tag(self):
        # tag 10 from optical frame
        tag_name = "tag_10"
        tag_pos_from_opt_frame, tag_quat_from_opt_frame = body_pose(
            self.tf_buffer,
            tag_name + "_from_opt_frame",
            self.zed_optical_frame,
            rot_type="quat"
        )
        # end effector from tag 10
        ee_pos_from_fk_tag, ee_quat_from_tag = body_pose(
            self.tf_buffer,
            "end_effector",
            "tag_10",
            rot_type="quat"
        )

        ee_pos_from_opt, ee_quat_from_opt = combine_frame_transforms(
            tag_pos_from_opt_frame, tag_quat_from_opt_frame,
            ee_pos_from_fk_tag, ee_quat_from_tag,
        )

        return (ee_pos_from_opt, ee_quat_from_opt)
        
    def get_delta_xyz_from_world_frame(self, ee_pos_from_opt):
        delta_pos_from_opt = self.pts_zed_cam_frame[0] - ee_pos_from_opt

        # Could not transform tag_10_from_opt_frame to 
        # zed2i_left_camera_optical_frame: "tag_10_from_opt_frame"
        # passed to lookupTransform argument source_frame does not exist. 

        opt_pos_w, opt_quat_w = body_pose(
            self.tf_buffer,
            self.zed_optical_frame,
            "world",
            rot_type="quat"
        )
        delta_pos_from_world = quat_apply(opt_quat_w, delta_pos_from_opt)

        self.tag_detected = True

        return delta_pos_from_world

    def get_next_target_hand_pose_wrt_cam(self):
        qr_tag_frame_name = 'tag0'
        optical_frame_name = 'camera_color_optical_frame'

        target_qr_tag_pos, target_qr_tag_quat = get_frame_from_tf_buffer_wrt_ref_frame(
                                                self.tf_buffer,
                                                frame = qr_tag_frame_name,
                                                ref_frame = optical_frame_name,
                                                rot_type = "quat"
                                                )
        return target_qr_tag_pos, target_qr_tag_quat


    def apply_transform_to_pose(pose, t, q):
        pass


    def differential_ik(self, target_hand_pose_wrt_pelvis):
        # Get current joint positions
        qj = np.zeros(29, dtype=np.float32)

        ## get current arm config
        for i_mot in range(len(self.config.motor_joint)):
            i_pin = self.pin_from_mot[i_mot]
            qj[i_pin] = self.low_state.motor_state[i_mot].q

        ## target_pose_b = end-effector pose in world frame
        gravity_vec = 9.81*np.array([0,0,-1])
        ## computes the residual
        res_q, arm_nle = self.ikctrl(qj,
                                    target_hand_pose_wrt_pelvis,
                                    rel=False,
                                    gravity_vec=gravity_vec,
                                    )
        # target_dof_pos = self.sit_target_dof_pos.copy() 
        # target_dof_pos = self.config.locomotion_motor_joint_offsets

        self._init_dof_pos = np.zeros(self.num_joints)
        for mot_idx in range(self.num_joints):
            self._init_dof_pos[mot_idx] = self.low_state.motor_state[mot_idx].q

        target_dof_pos = self._init_dof_pos

        # setting the target tau and dof pos just for the arm
        target_tau = np.zeros(29, dtype=np.float32)
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

    # inside move_hand_via_diff_ik_to, after you compute target_hand_pose_wrt_pelvis:
    def broadcast_hand_pose(self, target_hand_xyz, target_hand_quat, child_name, parent_name='pelvis'):
        t = TransformStamped()
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = parent_name              # parent frame
        t.child_frame_id = child_name          # your chosen child frame name
        target_hand_xyz = target_hand_xyz.squeeze()
        target_hand_quat = target_hand_quat.squeeze()
        t.transform.translation.x = float(target_hand_xyz[0])
        t.transform.translation.y = float(target_hand_xyz[1])
        t.transform.translation.z = float(target_hand_xyz[2])
        t.transform.rotation.x = float(target_hand_quat[0])
        t.transform.rotation.y = float(target_hand_quat[1])
        t.transform.rotation.z = float(target_hand_quat[2])
        t.transform.rotation.w = float(target_hand_quat[3])
        self.tf_broadcaster.sendTransform(t)


    def move_hand_via_diff_ik_to(self, target_hand_xyz, hand_pose_frame):
        # perform ik control
        # can i make it wait until the goal is achieved?
        # assert target hand pose must be in world frame
        # convert to world frame

        # put the target hand pose wrt pelvis
        pelvis_xyz, pelvis_quat = get_frame_from_tf_buffer_wrt_ref_frame(self.tf_buffer, \
                                                                         frame=hand_pose_frame,\
                                                                         ref_frame="pelvis",\
                                                                         rot_type="quat")
        pelvis_frame_detected = isinstance(pelvis_xyz, np.ndarray)
        if not pelvis_frame_detected:
            return 0
        
        target_hand_xyz_wrt_pelvis = self.apply_transform_to_points(target_hand_xyz, pelvis_xyz, pelvis_quat)
        target_hand_quat_wrt_pelvis = np.array([0.0, 0.0, 0.0, 1.0]) #NOTE this assumes that we have the lock mechanism on the waist, and waist is up-right
        target_hand_pose_wrt_pelvis = np.concatenate([target_hand_xyz_wrt_pelvis.squeeze(), target_hand_quat_wrt_pelvis])
        self.broadcast_hand_pose(target_hand_xyz_wrt_pelvis, target_hand_quat_wrt_pelvis, child_name="target_ee_pose")

        pelvis_xyz, pelvis_quat = get_frame_from_tf_buffer_wrt_ref_frame(self.tf_buffer, \
                                                                         frame=,\
                                                                         ref_frame="pelvis",\
                                                                         rot_type="quat")

        print("Computing the differential IK")
        target_dof_pos, target_tau = self.differential_ik(target_hand_pose_wrt_pelvis)
        kps = np.array(self.config.kps).astype(np.float32).copy()
        kds = np.array(self.config.kds).astype(np.float32).copy()

        # The way it works is that it iteratively solves for differential IK
        # I would like it so that it maintains the current joint poses for all other joints
        # kps[-7:] = self.config.eetrack_right_arm_kps
        # kds[-7:] = self.config.eetrack_right_arm_kds
        # target_tau[-7:] = 0.0

        print("Sending command")
        for mot_idx in range(self.num_joints):
            self.low_cmd.motor_cmd[mot_idx].q = float(target_dof_pos[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].dq = 0.0
            self.low_cmd.motor_cmd[mot_idx].kp = self.config.kpkd_smoothing * float(kps[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].kd = self.config.kpkd_smoothing * float(kds[mot_idx])
            self.low_cmd.motor_cmd[mot_idx].tau = float(target_tau[mot_idx])

        self.send_cmd(self.low_cmd)


    def get_delta_xyz_from_world_frame(self, ee_pos_from_opt):
        delta_pos_from_opt = self.pts_zed_cam_frame[0] - ee_pos_from_opt
        opt_pos_w, opt_quat_w = body_pose(
            self.tf_buffer,
            self.zed_optical_frame,
            "world",
            rot_type="quat"
        )
        delta_pos_from_world = quat_apply(opt_quat_w, delta_pos_from_opt)

        self.tag_detected = True

        return delta_pos_from_world


    def get_delta_xyz_between(target_xyz, curr_xyz):
        delta_xyz_wrt_cam = target_xyz - curr_xyz
        return delta_xyz_wrt_cam

    def run_policy(self):
        '''
        This function moves the hand to the qr tag
        If there is an offset between the target pose and the actual pose,
        it detects the qr-tag on the hand, computes the delta to the target pose,
        from the current pose, and adjusts for the delta.
        '''

        we_should_terminate = self.remote_controller.button[KeyMap.A] == 1
        if we_should_terminate:
            self._mode_change = True
            self.mode = Mode.finish
            return

        target_hand_xyz, target_hand_quat = self.get_next_target_hand_pose_wrt_cam()
        print("Detecting the tag")
        tag_detected = isinstance(target_hand_xyz, np.ndarray)
        if not tag_detected:
            return
        print("Tag detected")

        # Hand movement to the target
        print("Moving the ee to the target pose")
        camera_frame_name = 'camera_color_optical_frame'
        is_success = self.move_hand_via_diff_ik_to(target_hand_xyz, hand_pose_frame=camera_frame_name)
        if not is_success:
            return
        print("Finished")


        import pdb;pdb.set_trace()

        # hand QR tag detection
        curr_ee_pose_wrt_cam = self.detect_hand_qr_tag()

        # delta extraction
        delta_xyz = self.get_delta_xyz_between(target_hand_pose_wrt_cam, curr_ee_pose_wrt_cam)

        # hand movement
        new_target_hand_pose = target_hand_pose_wrt_cam + delta_xyz
        self.move_hand_to(new_target_hand_pose)


    def run_wrapper(self):
        if self.mode == Mode.finish:
            if self._mode_change:
                print("Finish.")
                self._mode_change = False
        else:
            self.run_policy()

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
