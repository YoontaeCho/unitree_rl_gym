#!/usr/bin/env bash

trap "trap - SIGINT && kill -- -$$" SIGINT SIGTERM

ros2 run tf2_ros static_transform_publisher 0.5 0.5 0 1.0 0 0 world fake_world &
python3 state_pub.py &
ros2 launch state_pub_zed.launch.py &
python3 fake_world_tf_v3_pub.py &
ros2 run rviz2 rviz2 &

wait
