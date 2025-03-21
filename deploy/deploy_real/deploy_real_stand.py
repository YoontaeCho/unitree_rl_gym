import math_utils
import random as rd
from act_to_dof import ActToDof
import utils_metric as um
import utils_stage as us
from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union, List
import numpy as np
import time
import torch
import torch as th
from pathlib import Path

import rclpy as rp
from unitree_hg.msg import LowCmd as LowCmdHG, LowState as LowStateHG
from unitree_go.msg import LowCmd as LowCmdGo, LowState as LowStateGo

from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import PoseStamped

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster, TransformStamped
from common.command_helper_ros import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
from common.crc import CRC
from enum import Enum
import pinocchio as pin
from ikctrl import IKCtrl, xyzw2wxyz
from yourdfpy import URDF

import math_utils
import random as rd
from act_to_dof import ActToDof


class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy = 4
    null = 5


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


from common.xml_helper import extract_link_data


def compute_com(tf_buffer, com_data, body_frames: List[str]):
    """compute com of body frames"""
    mass_list = []
    com_list = []

    # iterate for frames
    for frame in body_frames:
        try:
            frame_data = com_data[frame]
        except KeyError:
            continue

        try:
            link_pos, link_wxyz = body_pose(tf_buffer,
                                            frame, rot_type='quat')
        except TransformException:
            continue

        com_pos_b, com_wxyz = frame_data['pos'], frame_data['quat']

        # compute com from world coordinates
        # NOTE 'math_utils' package will be brought from isaaclab
        com_pos = link_pos + quat_rotate(link_wxyz, com_pos_b)
        com_list.append(com_pos)

        # get math
        mass = frame_data['mass']
        mass_list.append(mass)

    com = sum([m * pos for m, pos in zip(mass_list, com_list)]) / sum(mass_list)
    return com


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


def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos


class eetrack:
    def __init__(self, root_state_w, tf_buffer):
        self.tf_buffer = tf_buffer
        # self.eetrack_midpt = root_state_w.clone()
        # self.eetrack_midpt[..., 1] += 0.3
        self.eetrack_midpt = (
            root_state_w[..., :3] +
            quat_rotate(root_state_w[0, 3:7].detach().cpu().numpy(),
                        np.array([0.3, 0.0, 0.0]))[None]
        )
        self.eetrack_end = None
        self.eetrack_subgoal = None
        self.number_of_subgoals = 60
        self.eetrack_line_length = 0.3
        self.device = "cpu"
        self.waypoints = self.create_eetrack(root_state_w)
        self.eetrack_subgoal = self.create_subgoal(
                root_state_w,
                self.waypoints)
        self.sg_idx = 0
        # first subgoal sampling time = 1.0s
        # self.init_time = rp.time.Time()#.nanoseconds / 1e9 + 1.0
        self.init_time = clock.get_time()

    def create_eetrack(self, root_state_w):
        self.eetrack_start = self.eetrack_midpt.clone()
        self.eetrack_end = self.eetrack_midpt.clone()
        is_hor = rd.choice([True, False])
        eetrack_offset = rd.uniform(-0.5, 0.5)
        # For testing
        is_box = True
        is_hor = True

        eetrack_offset = 0.0
        if is_box:
            waypoints = []

            dx = (self.eetrack_line_length) / 2.
            dy = (self.eetrack_line_length) / 2.

            deltas = [
                    [0, +dy, +dx + 0.1],
                    [0, +dy, -dx + 0.1],
                    [0, -dy, -dx + 0.1],
                    [0, -dy, +dx + 0.1],
                    [0, +dy, +dx + 0.1]
            ]

            for delta in deltas:
                waypoint = self.eetrack_midpt.clone()
                waypoint += math_utils.quat_rotate(
                    root_state_w[..., 3:7].float(),
                    th.as_tensor(delta, dtype=th.float32)[None]
                )
                waypoints.append( waypoint )
            return waypoints

        elif is_hor:
            dx = (self.eetrack_line_length) / 2.
            dz = eetrack_offset
            delta_body0 = [0, +dx, dz]
            delta_body1 = [0, -dx, dz]

            self.eetrack_start += math_utils.quat_rotate(
                root_state_w[..., 3:7].float(),
                th.as_tensor(delta_body0, dtype=th.float32)[None]
            )
            self.eetrack_end += math_utils.quat_rotate(
                root_state_w[..., 3:7].float(),
                th.as_tensor(delta_body1, dtype=th.float32)[None]
            )
            # self.eetrack_start[..., 2] += eetrack_offset
            # self.eetrack_end[..., 2] += eetrack_offset
            # self.eetrack_start[..., 0] -= (self.eetrack_line_length) / 2.
            # self.eetrack_end[..., 0] += (self.eetrack_line_length) / 2.
        else:
            # self.eetrack_start[..., 0] += eetrack_offset
            # self.eetrack_end[..., 0] += eetrack_offset
            # self.eetrack_start[..., 2] += (self.eetrack_line_length) / 2.
            # self.eetrack_end[..., 2] -= (self.eetrack_line_length) / 2.
            dx = eetrack_offset
            dz = (self.eetrack_line_length) / 2.
            delta_body0 = [0, dx, +dz]
            delta_body1 = [0, dx, -dz]
            self.eetrack_start += math_utils.quat_rotate(
                root_state_w[..., 3:7],
                th.as_tensor(delta_body0)[None]
            )
            self.eetrack_end += math_utils.quat_rotate(
                root_state_w[..., 3:7],
                th.as_tensor(delta_body1)[None]
            )

        return self.eetrack_start, self.eetrack_end

    def create_direction(self):
        angle_from_eetrack_line = torch.rand(1, device=self.device) * np.pi
        angle_from_xyplane_in_global_frame = torch.rand(
            1, device=self.device) * np.pi - np.pi / 2
        # For testing
        angle_from_eetrack_line = torch.rand(1, device=self.device) * np.pi / 2
        angle_from_xyplane_in_global_frame = torch.rand(
            1, device=self.device) * 0
        roll = torch.zeros(1, device=self.device)
        pitch = angle_from_xyplane_in_global_frame
        yaw = angle_from_eetrack_line
        euler = torch.stack([roll, pitch, yaw], dim=1)
        quat = math_utils.quat_from_euler_xyz(
            euler[:, 0], euler[:, 1], euler[:, 2])
        return quat

    def create_subgoal(self, root_state_w, waypoints):
        qs = []
        for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
            eetrack_subgoals = interpolate_position(
                p0, p1, self.number_of_subgoals)
            eetrack_subgoals = [
                (
                    l.clone().to(self.device, dtype=torch.float32)
                    if isinstance(l, torch.Tensor)
                    else torch.tensor(l, device=self.device, dtype=torch.float32)
                )
                for l in eetrack_subgoals
            ]
            eetrack_subgoals = torch.stack(eetrack_subgoals, axis=1)

            eetrack_ori = self.create_direction().unsqueeze(
                1).repeat(1, self.number_of_subgoals + 1, 1)
            if True:
                eetrack_ori[..., :] = root_state_w[..., None, 3:7]
            # welidng_subgoals -> Nenv x Npoints x (3 + 4)
            q = torch.cat([eetrack_subgoals, eetrack_ori], dim=2)
            qs.append(q)
        return torch.cat(qs, dim=1)

    def update_command(self):
        # print(rp.time.Time().nanoseconds)
        time = (clock.get_time() - self.init_time).nanoseconds / 1e9
        if (time >= 1.0):
            self.sg_idx = int((time - 1) / 0.1 + 1)
        print(time, self.sg_idx)
        # self.sg_idx.clamp_(0, self.number_of_subgoals + 1)
        self.sg_idx = min(
                self.sg_idx,
                self.eetrack_subgoal.shape[-2] - 1)
        self.next_command_s_left = self.eetrack_subgoal[...,
                                                        self.sg_idx, :]

    def get_command(self, root_state_w):
        self.update_command()

        pos_hand_b_left, quat_hand_b_left = body_pose(
            self.tf_buffer,
            "left_rubber_hand",
            rot_type='quat'
        )

        lerp_command_w_left = self.next_command_s_left

        (lerp_command_b_left_pos,
         lerp_command_b_left_quat) = math_utils.subtract_frame_transforms(
            root_state_w[..., 0:3],
            root_state_w[..., 3:7],
            lerp_command_w_left[:, 0:3],
            lerp_command_w_left[:, 3:7],
        )

        # lerp_command_b_left = lerp_command_w_left

        pos_delta_b_left, rot_delta_b_left = math_utils.compute_pose_error(
            torch.from_numpy(pos_hand_b_left)[None],
            torch.from_numpy(quat_hand_b_left)[None],
            lerp_command_b_left_pos,
            lerp_command_b_left_quat,
        )
        axa_delta_b_left = math_utils.wrap_to_pi(rot_delta_b_left)

        hand_command = torch.cat((pos_delta_b_left, axa_delta_b_left), dim=-1)
        return hand_command


import os

class Controller(um.MetricUtils):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config
        self.remote_controller = RemoteController()
        # Load policy
        print(config.policy_path)
        self.policy = torch.jit.load(config.policy_path)
        self.policy.eval()

        # Metric test
        self.prev_joint_pos_target = None
        self.smoothing = self.config.smoothing
        self.prev_q = None
        self.prev_dq = None
        self.prev_ddq = None
        self.prev_tau = None
        self.prev_prev_dq = None
        self._pos_diff = []
        self._pos_jitter = []
        self._torque_diff = []
        self.exp_name = os.path.basename(self.config.policy_path)


        # == build index map ==
        self.mot_from_lab = index_map(self.config.motor_joint,
                                             self.config.lab_joint)
        self.lab_from_mot = index_map(self.config.lab_joint,
                                      self.config.motor_joint)
        self.config.default_angles = np.asarray(self.config.lab_joint_offsets)[
            self.lab_from_mot
        ]

        # Data buffers
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.tf_broadcaster = TransformBroadcaster(self._node)

        ### Mapping helpers.
        self.obsmap = us.Stage1Observation(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config, self.tf_buffer)
        self.eetrack = None
        self.ikctrl = IKCtrl(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config.arm_joint,
            frame='left_rubber_hand')
        self.actmap = us.SimpleAction(config, self.ikctrl)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type

            self.low_cmd = LowCmdHG()
            self.low_state = LowStateHG()

            self.lowcmd_publisher_ = self._node.create_publisher(LowCmdHG,
                                                                 'lowcmd', 10)
            self.lowstate_subscriber = self._node.create_subscription(
                LowStateHG, 'lowstate', self.LowStateHgHandler, 10)
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

        elif config.msg_type == "go":
            raise ValueError(f"{config.msg_type} is not implemented yet.")

        else:
            raise ValueError("Invalid msg_type")

        self.goalpath_publisher = self._node.create_publisher(
                PathMsg, 'goalpath', 10)
        self.truepath_publisher = self._node.create_publisher(
                PathMsg, 'truepath', 10)
        self.goalpath = PathMsg()
        self.truepath = PathMsg()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        self.mode = Mode.policy

        self._mode_change = True
        self._timer = self._node.create_timer(
            self.config.control_dt, self.run_wrapper)
        self._terminate = False
        try:
            rp.spin(self._node)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
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

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.mode_machine = self.mode_machine_
        cmd.crc = CRC().Crc(cmd)
        size = len(cmd.motor_cmd)
        self.lowcmd_publisher_.publish(cmd)

    def wait_for_low_state(self):
        while self.low_state.crc == 0:
            print(self.low_state)
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        if self.remote_controller.button[KeyMap.start] == 1:
            self._mode_change = True
            self.mode = Mode.default_pos
        else:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)

    def prepare_default_pos(self):
        # move time 2s
        total_time = 2
        self.counter = 0
        self._num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        self._kps = [float(kp) for kp in kps]
        self._kds = [float(kd) for kd in kds]
        self._default_pos = np.concatenate(
            (self.config.default_angles, self.config.arm_waist_target), axis=0)
        self._dof_size = len(dof_idx)
        self._dof_idx = dof_idx

        self._init_dof_pos = np.zeros(29)
        for i in range(29):
            self._init_dof_pos[i] = self.low_state.motor_state[i].q

    def move_to_default_pos(self):
        # move to default pos
        if self.counter < self._num_step:
            alpha = self.counter / self._num_step
            # FIXME(hh) only use upper-body
            for j in self.mot_from_upper_body:
                motor_idx = j
                target_pos = self.config.default_angles[j]

                self.low_cmd.motor_cmd[motor_idx].q = (
                    self._init_dof_pos[j] * (1 - alpha) + target_pos * alpha)
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0

            for i in self.mot_from_lower_body:
                self.low_cmd.motor_cmd[i].q = float(
                    0.0
                )
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.0
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.send_cmd(self.low_cmd)
            self.counter += 1
        else:
            self._mode_change = True
            self.mode = Mode.damping

    def default_pos_state(self):
        # FIXME(hh) only use upper-body joints
        if self.remote_controller.button[KeyMap.A] != 1:
            for i in self.mot_from_upper_body:
                self.low_cmd.motor_cmd[i].q = float(
                    0.0
                )
                self.low_cmd.motor_cmd[i].kp = 40.0
                self.low_cmd.motor_cmd[i].kd = 5.0

            for i in self.mot_from_lower_body:
                self.low_cmd.motor_cmd[i].q = float(
                    0.0
                )
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.0
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy

    def get_motor_state(self, low_state):
        """
        get motor states of upper body joints
        """
        q_mot = []
        dq_mot = []
        ddq_mot = []
        tau_mot = []
        for i_mot in self.mot_from_lab:
            q_mot.append(low_state.motor_state[i_mot].q)
            dq_mot.append(low_state.motor_state[i_mot].dq)
            ddq_mot.append(low_state.motor_state[i_mot].ddq)
            tau_mot.append(low_state.motor_state[i_mot].tau_est)
        
        return np.asarray(q_mot), np.asarray(dq_mot), np.asarray(ddq_mot), np.asarray(tau_mot)
        # return np.asarray(q_mot), None, np.asarray(ddq_mot), np.asarray(tau_mot)


    def publish_target(self):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'target'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = float(self.target_pose[0])
        t.transform.translation.y = float(self.target_pose[1])
        t.transform.translation.z = float(self.target_pose[2])

        # Set world_from_pelvis quaternion based on IMU state
        # TODO(ycho): consider applying 90-deg offset?
        qw, qx, qy, qz = [float(x) for x in self.target_pose[3:7]]
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def terminate_by_pelvis_condition(self, root_pose, limit_euler_angle=[0.9, 1.0]):
        xyz, quat_wxyz = root_pose[:3], root_pose[3:]
        euler = math_utils.wrap_to_pi(
            th.stack(math_utils.euler_xyz_from_quat(torch.as_tensor(quat_wxyz)), dim=-1)
        )
        out_of_limit = th.logical_or(
            th.abs(euler[..., 0]) > limit_euler_angle[0],
            th.abs(euler[..., 1]) > limit_euler_angle[1],
        )
        print(out_of_limit)

    def run_policy(self):
        logpath = Path('/tmp/metric_test/')
        logpath.mkdir(parents=True, exist_ok=True)

        if self.remote_controller.button[KeyMap.select] == 1:
            self._mode_change = True
            self.mode = Mode.null
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
        self.eetrack = eetrack(torch.from_numpy(root_state_w)[None],
                                   self.tf_buffer)
        
        self.goalpath.header.frame_id = 'world'
        self.goalpath.header.stamp = clock.get_time().to_msg()
        wpts = self.eetrack.waypoints
        for p in wpts:
            p = p.detach().cpu().numpy().squeeze(axis=0)
            p = [float(x) for x in p]
            msg = PoseStamped()
            msg.header.frame_id = 'world'
            msg.header.stamp = clock.get_time().to_msg()
            msg.pose.position.x = p[0]
            msg.pose.position.y = p[1]
            msg.pose.position.z = p[2]
            self.goalpath.poses.append(msg)
        self.truepath.header.frame_id = 'world'
        self.truepath.header.stamp = clock.get_time().to_msg()

        _hands_command_ = self.eetrack.get_command(
            torch.from_numpy(root_state_w)[None])[0].detach().cpu().numpy()

        self.target_pose = np.copy(
            self.eetrack.next_command_s_left.squeeze().detach().cpu().numpy())
        self.publish_target()

        # For standing.
        self.obs = self.obsmap(self.low_state, _hands_command_)
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone().float()
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()

        # target_dof_pos : motor joint ordered
        target_dof_pos = self.actmap(self.action, self.obs)
        
        
        # FIXME(hh) If you want smoothing
        if self.smoothing:
            if self.prev_joint_pos_target is not None:
                if self.counter < 100:
                    smoothing = 0.2
                    target_dof_pos = smoothing * target_dof_pos + \
                                    (1-smoothing) * self.prev_joint_pos_target
                else:
                    target_dof_pos = self.smoothing * target_dof_pos + \
                                    (1-self.smoothing) * self.prev_joint_pos_target
            self.prev_joint_pos_target = target_dof_pos

        # Calculate metrics
        curr_q, curr_dq, curr_ddq, curr_tau = self.get_motor_state(self.low_state)
        self.calculate_metrics(curr_q, curr_dq, curr_ddq, curr_tau, logpath)

        # FIXME(hh) 2nd smoothing
        # Build low cmd
        for i in range(len(self.config.motor_joint)):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = self.config.kpkd_smoothing * float(self.config.kps[i])
            self.low_cmd.motor_cmd[i].kd = self.config.kpkd_smoothing * float(self.config.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0
        
        # send the command
        self.send_cmd(self.low_cmd)

    def run_wrapper(self):
        # print("hello", self.mode,
        # self.mode == Mode.zero_torque)
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
                print("Waiting for the Button A signal...")
                self._mode_change = False
            self.default_pos_state()
        elif self.mode == Mode.policy:
            if self._mode_change:
                print("Run policy.")
                self._mode_change = False
                self.counter = 0
            self.run_policy()
        elif self.mode == Mode.null:
            self._terminate = True

        # time.sleep(self.config.control_dt)


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
