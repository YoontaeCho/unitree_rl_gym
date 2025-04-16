#!/usr/bin/env bash
python3 localization/state_pub.py &
ros2 launch localization/state_pub.launch.py &
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml rviz:=False &
python3 localization/pelvis_pub.py &
# python3 fake_world_tf_pub.py &
ros2 run rviz2 rviz2
