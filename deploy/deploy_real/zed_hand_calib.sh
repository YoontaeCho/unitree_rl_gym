source /root/localization/tag_ws/install/local_setup.sh
trap "trap - SIGINT && kill -- -$$" SIGINT SIGTERM
python3 tag_zed_to_zed2i.py &
ros2 launch apriltag_ros tag_zed.launch.py \
    config_path:=/root/localization/tag_ws/src/apriltag_ros/apriltag_ros/cfg/tags_36h11_filtered_self_arm.yaml \
    image_topic:=left_raw/image_raw_color
# python3 align_publisher_zed.py &
wait