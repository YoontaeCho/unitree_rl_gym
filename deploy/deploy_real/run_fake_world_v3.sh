#!/usr/bin/env bash

trap "trap - SIGINT && kill -- -$$" SIGINT SIGTERM

# ros2 run tf2_ros static_transform_publisher 0.5 0.5 0 1.0 0 0 mid_sole_link world &
python3 state_pub.py &
ros2 launch state_pub.launch.py &
python3 fake_world_tf_v3_pub.py &
python3 mid_sole_tf_pub.py &
ros2 run rviz2 rviz2 &

wait
