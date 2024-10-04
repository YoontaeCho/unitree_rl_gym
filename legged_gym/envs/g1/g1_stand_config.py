from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class G1StandCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        # pos = [0.0, 0.0, 0.8] # x,y,z [m]
        pos = [0.0, 0.0, 0.75] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 'left_hip_pitch_joint': -0.1,
            # 'left_hip_roll_joint': 0,
            # 'left_hip_yaw_joint': 0.,
            # 'left_knee_joint': 0.3,
            # 'left_ankle_pitch_joint': -0.2,
            # 'left_ankle_roll_joint': 0,
            # 'right_hip_pitch_joint': -0.1,
            # 'right_hip_roll_joint': 0,
            # 'right_hip_yaw_joint': 0.,
            # 'right_knee_joint': 0.3,
            # 'right_ankle_pitch_joint': -0.2,
            # Zero pad
            'left_hip_pitch_joint': 0,
            # 'left_hip_roll_joint': np.pi/8,
            'left_hip_roll_joint': 0.,
            'left_hip_yaw_joint': 0.,
            'left_knee_joint': 0,
            'left_ankle_pitch_joint': 0,
            'left_ankle_roll_joint': 0,
            'right_hip_pitch_joint': 0,
            # 'right_hip_roll_joint': -np.pi/8,
            'right_hip_roll_joint': 0.,
            'right_hip_yaw_joint': 0.,
            'right_knee_joint': 0,
            'right_ankle_pitch_joint': 0,
            # Until here
            'right_ankle_roll_joint': 0,
            'torso_joint': 0.,
            'left_shoulder_pitch_joint': 0,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0.,
            'left_elbow_pitch_joint': 0.,
            'left_elbow_roll_joint': 0.,
            'left_five_joint': 0.,
            'left_six_joint': 0.,
            'left_three_joint': 0.,
            'left_four_joint': 0.,
            'left_zero_joint': 0.,
            'left_one_joint': 0.,
            'left_two_joint': 0.,
            'right_shoulder_pitch_joint': 0,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0.,
            'right_elbow_pitch_joint': 0.,
            'right_elbow_roll_joint': 0.,
            'right_five_joint': 0.,
            'right_six_joint': 0.,
            'right_three_joint': 0.,
            'right_four_joint': 0.,
            'right_zero_joint': 0.,
            'right_one_joint': 0.,
            'right_two_joint': 0.
        }
        # Change the dof lower and upper limits
        limits = {
           'torso_joint' : [-np.pi/3, np.pi/3],
        #    'torso_joint' : [-np.pi/4, np.pi/4],

        }
    
    class env(LeggedRobotCfg.env):
        # 3+3+3+12+3+12+12
        # 3+3+3+25+3+25+25
        num_envs = 4096
        num_observations = 126 + 16
        # num_observations = 126 + 12 + 5
        num_actions = 37
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 

        env_spacing = 5.  # not used with heightfields/trimeshes 
        # env_spacing = 10.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        # episode_length_s = 20 # episode length in seconds
        episode_length_s = 10 # episode length in seconds
        # episode_length_s = 8 # episode length in seconds
        test = False

        class termination:
            termination_force = 150
            termination_force_hand = 1
            rpy_thresh = [0.5, 1.5]

      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {
            'left_hip_pitch': 50.,
            'left_hip_roll': 50.,
            'left_hip_yaw': 50.,
            'left_knee': 50.,
            'left_ankle_pitch': 50.,
            'left_ankle_roll': 50.,
            'right_hip_pitch': 50.,
            'right_hip_roll': 50.,
            'right_hip_yaw': 50.,
            'right_knee': 50.,
            'right_ankle_pitch': 50.,
            'right_ankle_roll': 50.,
            'torso': 50.,
            'left_shoulder_pitch': 50.,
            'left_shoulder_roll': 50.,
            'left_shoulder_yaw': 50.,
            'left_elbow_pitch': 50.,
            'left_elbow_roll': 50.,
            'left_five': 50.,
            'left_six': 50.,
            'left_three': 50.,
            'left_four': 50.,
            'left_zero': 50.,
            'left_one': 50.,
            'left_two': 50.,
            'right_shoulder_pitch': 50.,
            'right_shoulder_roll': 50.,
            'right_shoulder_yaw': 50.,
            'right_elbow_pitch': 50.,
            'right_elbow_roll': 50.,
            'right_five': 50.,
            'right_six': 50.,
            'right_three': 50.,
            'right_four': 50.,
            'right_zero': 50.,
            'right_one': 50.,
            'right_two': 50.
        }  # [N*m/rad]
        damping = {  
            'left_hip_pitch': 1.,
            'left_hip_roll': 1.,
            'left_hip_yaw': 1.,
            'left_knee': 1.,
            'left_ankle_pitch': 1.,
            'left_ankle_roll': 1.,
            'right_hip_pitch': 1.,
            'right_hip_roll': 1.,
            'right_hip_yaw': 1.,
            'right_knee': 1.,
            'right_ankle_pitch': 1.,
            'right_ankle_roll': 1.,
            'torso': 1.,
            'left_shoulder_pitch': 1.,
            'left_shoulder_roll': 1.,
            'left_shoulder_yaw': 1.,
            'left_elbow_pitch': 1.,
            'left_elbow_roll': 1.,
            'left_five': 1.,
            'left_six': 1.,
            'left_three': 1.,
            'left_four': 1.,
            'left_zero': 1.,
            'left_one': 1.,
            'left_two': 1.,
            'right_shoulder_pitch': 1.,
            'right_shoulder_roll': 1.,
            'right_shoulder_yaw': 1.,
            'right_elbow_pitch': 1.,
            'right_elbow_roll': 1.,
            'right_five': 1.,
            'right_six': 1.,
            'right_three': 1.,
            'right_four': 1.,
            'right_zero': 1.,
            'right_one': 1.,
            'right_two': 1.
        }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/urdf/g1_full.urdf'
        # name = "g1"
        name = "g1_full"
        ankle_joint_name = "ankle"
        torso_name = "torso_link"
        foot_name = "ankle_roll"
        elbow_name = "elbow_roll"
        elbow_hand_offset = 0.20 # Distance between robot elbow <-> hand
        fingertip_links = ["two", "four", "six"]
        left_fingers = ["left_two", "left_four", "left_six"]
        right_fingers = ["right_two", "right_four", "right_six"]
        hip = ["hip_pitch"]
        # penalize_contacts_on = ["hip", "knee"]
        penalize_contacts_on = []
        # terminate_after_contacts_on = ["torso"]
        # terminate_after_contacts_on = ["hip_yaw"]
        terminate_after_contacts_on = ["hip_yaw"]
        # terminate_after_contacts_on = ["hip_yaw", "two", "four", "six"]
        # self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
    
    class object(LeggedRobotCfg.object):
        # density = 200
        density = 50
        initial_hand_obj_dist = 3.
        # density = 50
        box_size = [0.2, 0.2, 0.2]
        succeed_threshold = 0.6
        holding_time_threshold = 150 # Number of steps to "succeed"

        use_curriculum = False
        schedule_steps = [5, 10]
        mass_range = [1, 2]
    
    class footpoint:
        # used when computing the support polygon
        num_footpoints = 4
        offeset = [[0.13, -0.025, -0.032],
                   [0.13, 0.025, -0.032],
                   [-0.055, -0.025, -0.032],
                   [-0.055, 0.025, -0.032]]
        # epsilon = 0.01
        epsilon = 0.005



    class domain_rand:
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 4
        max_push_force_xy = 10000
        # max_push_force_xy = 5000
        max_push_vel_rpy = 1.
        randomize_init_orn = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.728
        class zmp:
            max_dist = 1.5
            sigma = 6.
            sigma_dist = 4.
            sigma_margin = 10.
        class com:
            max_dist = 1.5
            sigma = 4.
        class scales( LeggedRobotCfg.rewards.scales ):
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = -1.0
            # base_height = -10.0
            # termination = -10
            # survive = 0.5

            termination = -20
            survive = 2.0
            # survive = 1.0
            # Reward set 1: Standing(balancing) reward
            stand_foot_orient_v3 = 0.
            stand_foot_height = 0.
            stand_foot_vel_v2 = 0.
            stand_zmp_supp_dist = 0.
            stand_zmp_dist = -0.5
            stand_zmp_margin = 1.0
            # stand_planar_contact = -0.5
            # stand_ankle_torque = -0.0002
            stand_ankle_torque_v2 = 0.
            stand_ankle_limit = 0.
            # Reward set 2: task-relevant rewards
            # dist_left = 0.5
            # # dist_right = 0.5
            # dist_left_v2 = 2.0
            # dist_right_v2 = 2.0
            # left_contact = 0.5
            # right_contact = 0.5
            # pickup_v2 = 20.
            # # obj_zvel = 10.
            # # obj_xyvel = -1.
            # completion = 1500.
            # lifted = 8.
            # pelvis_height =0.
            # base_obj_dist = 0.

            # com_avgfoot_dist = -1.0
            # zmp_avgfoot_dist = 0.
            # zmp_avgfoot_dist_v2 = -0.2
            # com_avgfoot_dist_v2 = 0.
            # base_orient = -2.0
            # foot_orient = -0.1
            # foot_orient_v2 = -0.1
            # foot_vel = -0.05
            # single_foot_contact = -0.3

            # Reward set 3: Regulation reward
            reg_dof_vel = -0.00001
            reg_action_rate = -0.002
            # action_double_rate = -0.0005
            reg_dof_acc = -1e-8
            reg_torques = -0.001
            reg_dof_pos_limits = -0.05


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        # tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0
    
    class noise(LeggedRobotCfg.noise):
        add_noise = False

class G1StandCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'g1'

        # max_iterations = 10000 # number of policy updates
        max_iterations = 3000 # number of policy updates
        # max_iterations = 15000 # number of policy updates
        # save_interval = 250
        save_interval = 200


  
