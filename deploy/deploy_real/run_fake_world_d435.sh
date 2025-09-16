#!/usr/bin/env bash

trap "trap - SIGINT && kill -- -$$" SIGINT SIGTERM

ros2 run tf2_ros static_transform_publisher 0.5 0.5 0 1.0 0 0 world fake_world &
python3 state_pub.py &
ros2 launch state_pub_d435.launch.py &
python3 fake_world_tf_d435_pub.py &
python3 mid_sole_tf_pub.py &
ros2 run rviz2 rviz2 &

wait
