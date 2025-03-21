def index_map(k_to, k_from):
    """
    Returns an index mapping from k_from to k_to.

    Given k_to=a, k_from=b,
    returns an index map "a_from_b" such that
    array_a[a_from_b] = array_b

    Missing values are set to -1.

    k_from 에 있는 조인트들이 k_to 에 어디에 있는지.
    """
    index_dict = {k: i for i, k in enumerate(k_to)}  # O(len(k_from))
    return [index_dict.get(k, -1) for k in k_from]  # O(len(k_to))

from legged_gym import LEGGED_GYM_ROOT_DIR

config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/g1_stand.yaml"
from config import Config
config = Config(config_path)

mot_from_lab = index_map(config.motor_joint,config.lab_joint)


lab_from_mot = index_map(config.lab_joint,config.motor_joint)


lab_from_jpa = index_map(
            config.lab_joint,
            config.jpa_joint
            )


rjpa_from_mot = index_map(
            config.jpa_joint,
            config.motor_joint
            )

jpa_from_lab = index_map(
            config.motor_joint,
            config.jpa_joint
            )   

mot_from_jpa = index_map(   
            config.motor_joint,
            config.jpa_joint
            )   

# k_from 에 있는 조인트들이 k_to 에 어디에 있는지.
target = mot_from_jpa

print(len(target))
for i in target:
    print(f"{i} -> {config.lab_joint[i]}")