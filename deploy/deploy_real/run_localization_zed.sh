#!/usr/bin/env bash
TIME=`date +"%s"`
trap "trap - SIGINT && kill -- -$$" SIGINT SIGTERM

python3 state_pub.py &
ros2 launch state_pub_zed.launch.py &
python3 localization/pelvis_pub_v2_zed.py &
python3 mid_sole_tf_pub.py &
python3 fixed_z_pub.py &
# python3 fake_world_tf_v2_pub.py &
ros2 run rviz2 rviz2 &
# ros2 bag record /odom /tf /tf_static -o /tmp/bag/$TIME

wait