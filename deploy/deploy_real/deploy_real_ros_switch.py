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
from low_state_to_obs import Observation
from eetrack import eetrack

class Mode(Enum):
    wait = 0
    zero_torque = 1
    default_pos = 2
    damping = 3
    policy_walk = 4
    policy_eetrack = 5
    null = 6


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


def interpolate_position(pos1, pos2, n_segments):
    increments = (pos2 - pos1) / n_segments
    interp_pos = [pos1 + increments * p for p in range(n_segments)]
    interp_pos.append(pos2)
    return interp_pos



class Controller:
    def __init__(self,
            config_walk: Config,
            config_eetrack: Config
        ) -> None:
        self.config_walk = config_walk
        self.config_eetrack = config_eetrack

        self.remote_controller = RemoteController()

        # Initialize the policy networks
        self.policy_walk = torch.jit.load(config_walk.policy_path)
        self.policy_eetrack = torch.jit.load(config_eetrack.policy_path)        

        # -- build handles for EETRACK --
        config=self.config_eetrack
        self.ikctrl = IKCtrl(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config.arm_joint,
            frame='left_rubber_hand')
        self.actmap = ActToDof(config, self.ikctrl)
        self.lim_lo_pin = self.ikctrl.robot.model.lowerPositionLimit
        self.lim_hi_pin = self.ikctrl.robot.model.upperPositionLimit

        # == build index maps ==
        arm_joint = config.arm_joint
        self.mot_from_pin = index_map(
            config.motor_joint,
            self.ikctrl.joint_names)
        self.pin_from_mot = index_map(
            self.ikctrl.joint_names,
            config.motor_joint
        )
        self.mot_from_arm = index_map(
            config.motor_joint,
            config.arm_joint
        )
        self.mot_from_nonarm = index_map(
            config.motor_joint,
            config.non_arm_joint
        )
        self.lab_from_mot = index_map(config.lab_joint,
                                      config.motor_joint)
        config.default_angles = np.asarray(config.lab_joint_offsets)[
            self.lab_from_mot
        ]


        # Data buffers
        # -- SHARED --
        self.counter = 0
        # -- for WALK --
        self.cmd = np.array([0.0, 0, 0])
        self.obs_walk = np.zeros(config_walk.num_obs, dtype=np.float32)
        self.action_walk = np.zeros(config_walk.num_actions, dtype=np.float32)
        self.qj = np.zeros(config_walk.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config_walk.num_actions, dtype=np.float32)
        # -- for EETRACK ONLY --
        self.action_eetrack = np.zeros(config_eetrack.num_actions, dtype=np.float32)


        # Create ROS handles & helpers
        rp.init()
        self._node = rp.create_node("low_level_cmd_sender")

        global clock
        clock = GlobalClock(self._node)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.tf_broadcaster = TransformBroadcaster(self._node)
        self.obsmap = Observation(
            '../../resources/robots/g1_description/g1_29dof_rev_1_0.urdf',
            config_eetrack, self.tf_buffer)

        # FIXME(ycho): give `root_state_w`
        self.eetrack = None

        if True:
            q_mot = np.array(config_eetrack.default_angles)
            q_pin = np.zeros_like(self.ikctrl.cfg.q)
            q_pin[self.pin_from_mot] = q_mot
            default_pose = self.ikctrl.fk(q_pin)
            xyz = default_pose.translation
            quat_wxyz = xyzw2wxyz(
                pin.Quaternion(
                    default_pose.rotation).coeffs())
            self.default_pose_b = np.concatenate([xyz, quat_wxyz])
            self.target_pose = None

        if config_eetrack.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = LowCmdHG()
            self.low_state = LowStateHG()

            self.lowcmd_publisher_ = self._node.create_publisher(LowCmdHG,
                                                                 'lowcmd', 10)
            self.lowstate_subscriber = self._node.create_subscription(
                LowStateHG, 'lowstate', self.LowStateHgHandler, 10)
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

        elif config_eetrack.msg_type == "go":
            raise ValueError(f"{config_eetrack.msg_type} is not implemented yet.")

        else:
            raise ValueError("Invalid msg_type")

        # NOTE(ycho): only for debugging purposes
        self.goalpath_publisher = self._node.create_publisher(
                PathMsg, 'goalpath', 10)
        self.truepath_publisher = self._node.create_publisher(
                PathMsg, 'truepath', 10)
        self.goalpath = PathMsg()
        self.truepath = PathMsg()

        # wait for the subscriber to receive data
        # self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=config_eetrack.weak_motor)
        
        self.mode = Mode.wait
        self._mode_change = True

        self._timer = self._node.create_timer(
            config_eetrack.control_dt, self.run_wrapper)
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
            time.sleep(self.config_eetrack.control_dt)
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
        config=self.config_walk
        total_time = 2
        self.counter = 0
        self._num_step = int(total_time / config.control_dt)

        dof_idx = config.leg_joint2motor_idx + config.arm_waist_joint2motor_idx
        kps = config.kps + config.arm_waist_kps
        kds = config.kds + config.arm_waist_kds
        self._kps = [float(kp) for kp in kps]
        self._kds = [float(kd) for kd in kds]
        self._default_pos = np.concatenate(
            (config.default_angles, config.arm_waist_target), axis=0)
        self._dof_size = len(dof_idx)
        self._dof_idx = dof_idx

        # record the current pos
        # self._init_dof_pos = np.zeros(self._dof_size,
        #                               dtype=np.float32)
        # for i in range(self._dof_size):
        #     self._init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        self._init_dof_pos = np.zeros(29)
        for i in range(29):
            self._init_dof_pos[i] = self.low_state.motor_state[i].q

    def move_to_default_pos(self):
        # move to default pos
        if self.counter < self._num_step:
            alpha = self.counter / self._num_step
            # for j in range(self._dof_size):
            for j in range(29):
                # motor_idx = self._dof_idx[j]
                # target_pos = self._default_pos[j]
                motor_idx = j
                target_pos = self.config_eetrack.default_angles[j]

                self.low_cmd.motor_cmd[motor_idx].q = (
                    self._init_dof_pos[j] * (1 - alpha) + target_pos * alpha)
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self._kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self._kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            self.counter += 1
        else:
            self._mode_change = True
            self.mode = Mode.damping

    def default_pos_state(self):
        if self.remote_controller.button[KeyMap.A] != 1:
            for i in range(29):
                self.low_cmd.motor_cmd[i].q = float(
                    0.0
                )
                self.low_cmd.motor_cmd[i].kp = 40.0
                self.low_cmd.motor_cmd[i].kd = 5.0
            self.send_cmd(self.low_cmd)
        else:
            self._mode_change = True
            self.mode = Mode.policy_walk

    def publish_hand_target(self):
        t = TransformStamped()

        # Format header
        t.header.stamp = self._node.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'target'

        # Populate translation
        t.transform.translation.x = float(self.target_pose[0])
        t.transform.translation.y = float(self.target_pose[1])
        t.transform.translation.z = float(self.target_pose[2])

        # Set world_from_pelvis quaternion based on IMU state
        qw, qx, qy, qz = [float(x) for x in self.target_pose[3:7]]
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def run_walk_policy(self):
        """
        requires:
        * self.low_state
        * self.config_walk

        side-effects:
        * self.qj
        * self.dqj
        * self.cmd
        * self.obs
        * self.action_walk
        * self.low_cmd
        """
        
        config = self.config_walk

        if self.remote_controller.button[KeyMap.select] == 1:
            self._mode_change = True
            self.mode = Mode.null
            return
        
        if self.remote_controller.button[KeyMap.B] == 1:
            self._mode_change = True
            self.mode = Mode.policy_eetrack
            return
        
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - config.default_angles) * config.dof_pos_scale
        dqj_obs = dqj_obs * config.dof_vel_scale
        ang_vel = ang_vel * config.ang_vel_scale
        period = 0.8
        count = self.counter * config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = config.num_actions
        self.obs_walk[0:3] = ang_vel
        self.obs_walk[3:6] = gravity_orientation
        self.obs_walk[6:9] = self.cmd * config.cmd_scale * config.max_cmd
        self.obs_walk[9 : 9 + num_actions] = qj_obs
        self.obs_walk[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs_walk[9 + num_actions * 2 : 9 + num_actions * 3] = self.action_walk
        self.obs_walk[9 + num_actions * 3] = sin_phase
        self.obs_walk[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs_walk).unsqueeze(0)
        self.action_walk = self.policy_walk(obs_tensor.float()).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = config.default_angles + self.action_walk * config.action_scale

        # Build low cmd
        for i in range(len(config.leg_joint2motor_idx)):
            motor_idx = config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[motor_idx].dq = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = float(config.kps[i])
            self.low_cmd.motor_cmd[motor_idx].kd = float(config.kds[i])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        for i in range(len(config.arm_waist_joint2motor_idx)):
            motor_idx = config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = float(config.arm_waist_target[i])
            self.low_cmd.motor_cmd[motor_idx].dq = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = float(config.arm_waist_kps[i])
            self.low_cmd.motor_cmd[motor_idx].kd = float(config.arm_waist_kds[i])
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        # send the command
        self.send_cmd(self.low_cmd)

    def run_eetrack_policy(self):
        if self.remote_controller.button[KeyMap.select] == 1:
            self._mode_change = True
            self.mode = Mode.null
            return
        self.counter += 1
        config = self.config_eetrack

        # Initialize hand target from current location.
        if self.target_pose is None:
            xyz, quat = body_pose(
                self.tf_buffer,
                'left_rubber_hand',
                'world',
                rot_type='quat'
            )
            self.target_pose = np.concatenate([xyz, quat])

        # Query root state.
        # NOTE(ycho): requires running `fake_world_tf_pub.py`.
        if True:
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

        # Initialize EETrack object
        if self.eetrack is None:
            self.eetrack = eetrack(torch.from_numpy(root_state_w)[None],
                                   self.tf_buffer,
                                   clock
                                   )

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
                # msg.pose.quaternion.w = p[3]
                # msg.pose.quaternion.x = p[4]
                # msg.pose.quaternion.y = p[5]
                # msg.pose.quaternion.z = p[6]
                self.goalpath.poses.append(msg)
            self.truepath.header.frame_id = 'world'
            self.truepath.header.stamp = clock.get_time().to_msg()
                        

        # NOTE(ycho) Get hands_command from eetrack 
        if True:
            hands_command = self.eetrack.get_command(
                torch.from_numpy(root_state_w)[None])[0].detach().cpu().numpy()

            # == clip hands_command...? to prevent rapid switches ==
            # likely not necessary for the policy, but potentially
            # necessary for the IK controller.
            if False:
                hands_command[..., 0:3] = np.clip(hands_command[..., 0:3],
                    -0.1, 0.1) # 10cm
                hands_command[..., 3:6] = np.clip(hands_command[..., 3:6],
                    -np.deg2rad(5),
                    np.deg2rad(5)) # 5deg

            self.target_pose = np.copy(
                self.eetrack.next_command_s_left.squeeze().detach().cpu().numpy()
            )
            self.publish_hand_target()

        obs = self.obsmap(self.low_state,
                        self.action_eetrack,
                        hands_command)
        logpath = Path('/tmp/eet28/')
        logpath.mkdir(parents=True, exist_ok=True)
        # np.save(F'{logpath}/obs{self.counter:03d}.npy',
        #         obs)

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        obs_tensor = obs_tensor.detach().clone()

        if self.action_eetrack is None:
            self.action_eetrack = self.policy_eetrack(obs_tensor.float()).detach().numpy().squeeze()
        else:
            action = self.policy_eetrack(obs_tensor.float()).detach().numpy().squeeze()
            self.action_eetrack = (0.7 * self.action_eetrack + 0.3 * action)

        # np.save(F'{logpath}/act{self.counter:03d}.npy',
        #          self.action)

        target_dof_pos, target_dof_eff = self.actmap(
            obs,
            self.action_eetrack,
            # NOTE(ycho): We don't use root_state_w[3:7] since
            # hands_command is already in body frame.
            # root_state_w[3:7]
        )

        # np.save(F'{logpath}/dof{self.counter:03d}.npy',
        #         target_dof_pos)

        q_mot = np.asarray(
            [self.low_state.motor_state[i_mot].q for i_mot in range(29)]
        )


        if self.counter <= 100:
            target_dof_pos = (
                0.8 * q_mot +
                0.2 * target_dof_pos
            )
        # if self.counter <= 100:
        #     target_dof_pos = (
        #         0.7 * q_mot +
        #         0.3 * target_dof_pos
        #     )
        # if self.counter <= 150:
        #     target_dof_pos = (
        #         0.6 * q_mot +
        #         0.4 * target_dof_pos
        #     )
        # if self.counter <= 200:
        #     target_dof_pos = (
        #         0.5 * q_mot +
        #         0.5 * target_dof_pos
        #     )
        # if self.counter <= 250:
        #     target_dof_pos = (
        #         0.4 * q_mot +
        #         0.6 * target_dof_pos
        #     )
        # if self.counter <= 300:
        #     target_dof_pos = (
        #         0.3 * q_mot +
        #         0.7 * target_dof_pos
        #     )
        else:
            target_dof_pos = (
                0.3 * q_mot +
                0.7 * target_dof_pos
            )

        # target_dof_pos = (
        #     0.6 * q_mot +
        #     0.4 * target_dof_pos
        # )

        # NOTE(ycho): Optionally,
        # try to reduce control targets on waist joints
        # target_dof_pos[..., [2,5,8]] = 0
        
        # np.save(F'{logpath}/dof{self.counter:03d}.npy',
        #         target_dof_pos)

        # Build low cmd
        for i in range(len(config.motor_joint)):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            # FIXME(ycho) ad-hoc 0.8x reduction
            self.low_cmd.motor_cmd[i].kp = 0.8 * float(config.kps[i])
            self.low_cmd.motor_cmd[i].kd = 1.0 * float(config.kds[i])
            self.low_cmd.motor_cmd[i].tau = 0.0 * float(target_dof_eff[i])
            # self.low_cmd.motor_cmd[i].q = 0. * float(target_dof_pos[i])
            # self.low_cmd.motor_cmd[i].dq = 0.0
            # self.low_cmd.motor_cmd[i].kp = 0. * float(config.kps[i])
            # self.low_cmd.motor_cmd[i].kd = 0.0 * float(config.kds[i])
            # self.low_cmd.motor_cmd[i].tau = 0.0 * float(target_dof_eff[i])

        # reduce KP for non-arm joints
        for i in self.mot_from_nonarm:
            # FIXME(ycho) ad-hoc 0.8x reduction
            self.low_cmd.motor_cmd[i].kp = 0.8 * float(config.kps[i])
            self.low_cmd.motor_cmd[i].kd = 1.0 * float(config.kds[i])
            # self.low_cmd.motor_cmd[i].kp = 0. * float(config.kps[i])
            # self.low_cmd.motor_cmd[i].kd = 0.0 * float(config.kds[i])

        # send the command
        self.send_cmd(self.low_cmd)

        # NOTE(ycho): ONLY for debugging purposes
        if False:
            msg = PoseStamped()
            msg.header.frame_id='world'
            msg.header.stamp=clock.get_time().to_msg()
            cur_xyz, cur_quat = body_pose(
                self.tf_buffer,
                'left_rubber_hand',
                'world',
                rot_type='quat')
            msg.pose.position.x = float(cur_xyz[0])
            msg.pose.position.y = float(cur_xyz[1])
            msg.pose.position.z = float(cur_xyz[2])
            msg.pose.orientation.w = float(cur_quat[0])
            msg.pose.orientation.x = float(cur_quat[1])
            msg.pose.orientation.y = float(cur_quat[2])
            msg.pose.orientation.z = float(cur_quat[3])
            self.truepath.poses.append(msg)
            self.goalpath_publisher.publish(self.goalpath)
            self.truepath_publisher.publish(self.truepath)

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
                print("Waiting for the Button A signal...")
                self._mode_change = False
            self.default_pos_state()
        elif self.mode == Mode.policy_walk:
            if self._mode_change:
                print("Run policy.")
                self._mode_change = False
                self.counter = 0
            self.run_walk_policy()
        elif self.mode == Mode.policy_eetrack:
            if self._mode_change:
                print("Run policy.")
                self._mode_change = False
                self.counter = 0
            self.run_eetrack_policy()
        elif self.mode == Mode.null:
            self._terminate = True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_walk",
        type=str,
        help="config file name in the configs folder",
        default="g1.yaml")
    parser.add_argument(
        "config_eetrack",
        type=str,
        help="config file name in the configs folder",
        default="g1_eetrack.yaml")
    args = parser.parse_args()

    # Load config
    config_walk_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config_walk}"
    config_walk = Config(config_walk_path)
    config_eetrack_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config_eetrack}"
    config_eetrack = Config(config_eetrack_path)

    controller = Controller(
        config_walk,
        config_eetrack)