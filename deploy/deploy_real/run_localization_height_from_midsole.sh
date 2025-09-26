#!/usr/bin/env bash
source /root/localization/tag_ws/install/local_setup.sh

TIME=`date +"%s"`
trap "trap - SIGINT && kill -- -$$" SIGINT SIGTERM

python3 state_pub.py &
ros2 launch state_pub.launch.py &
ros2 launch livox_ros_driver2 msg_MID360_launch.py &
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml rviz:=False &
python3 localization/pelvis_pub_v2_height_from_midsole.py &
python3 mid_sole_tf_pub_slam.py &

# Turn on apriltag detection ros node
ros2 launch apriltag_ros tag_zed.launch.py \
    config_path:=/root/localization/tag_ws/src/apriltag_ros/apriltag_ros/cfg/tags_36h11_filtered_align.yaml \
    image_topic:=left_raw/image_raw_color &

ros2 run rviz2 rviz2 &
# ros2 bag record /odom /tf /tf_static -o /tmp/bag/$TIME

wait