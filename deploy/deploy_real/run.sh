#!/usr/bin/env bash
python3 state_pub.py &
ros2 launch state_pub.launch.py &
python3 fake_world_tf_pub.py &
ros2 run rviz2 rviz2
