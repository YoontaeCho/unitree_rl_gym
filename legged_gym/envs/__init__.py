from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_full_config import G1GraspCfg, G1RGraspCfgPPO
from legged_gym.envs.g1.g1_stand_config import G1StandCfg, G1StandCfgPPO
from .base.legged_robot import LeggedRobot
from legged_gym.envs.g1.g1_stand import G1

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO(), 'go2')
task_registry.register( "h1", LeggedRobot, H1RoughCfg(), H1RoughCfgPPO(), 'h1')
task_registry.register( "h1_2", LeggedRobot, H1_2RoughCfg(), H1_2RoughCfgPPO(), 'h1_2')
task_registry.register( "g1", LeggedRobot, G1RoughCfg(), G1RoughCfgPPO(), 'g1')
task_registry.register( "g1_grasp", LeggedRobot, G1GraspCfg(), G1RGraspCfgPPO(), 'g1')
task_registry.register( "g1_stand", G1, G1StandCfg(), G1StandCfgPPO(), 'g1')
