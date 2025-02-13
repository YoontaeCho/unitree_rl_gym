#!/usr/bin/env python3
import os
import cv2
import csv
import numpy as np

# rosbag2_py is available in ROS2 Foxy and later.
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message

# Import message types
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage

# cv_bridge is used to convert ROS image messages to OpenCV images.
from cv_bridge import CvBridge

def find_most_recent_tf_for_relation(tf_msgs, target_timestamp, parent_frame, child_frame):
    """
    Given a sorted list of tf_msgs (each is a tuple of (timestamp, TFMessage)) and a target timestamp,
    return the most recent TF message (i.e. the one with the largest timestamp that is <= target_timestamp)
    that contains a transform with the specified parent and child frames.
    
    Returns:
        A tuple (tf_timestamp, transform) if found, or None if no matching transform is found.
    """
    # Iterate in reverse (most recent first)
    for timestamp, tf_msg in reversed(tf_msgs):
        if timestamp <= target_timestamp:
            # Check each transform in this TF message for the desired relation.
            for transform in tf_msg.transforms:
                if (transform.header.frame_id == parent_frame and 
                    transform.child_frame_id == child_frame):
                    return timestamp, transform
    return None

def tf_to_matrix(transform):
    """
    Convert a geometry_msgs Transform (inside a TransformStamped)
    to a 4x4 homogeneous transformation matrix.
    """
    T = np.eye(4, dtype=np.float32)
    # Set translation.
    T[0, 3] = transform.transform.translation.x
    T[1, 3] = transform.transform.translation.y
    T[2, 3] = transform.transform.translation.z
    # Convert quaternion to rotation matrix.
    qx = transform.transform.rotation.x
    qy = transform.transform.rotation.y
    qz = transform.transform.rotation.z
    qw = transform.transform.rotation.w
    R = quaternion_to_rotation_matrix([qx, qy, qz, qw])
    T[:3, :3] = R
    return T

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    R = np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)
    return R

def rotationMatrixToQuaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw].
    """
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
    return np.array([qx, qy, qz, qw])

def main():
    # Path to your bag file
    # bag_file = '/root/unitree/unitree_rl_gym/rosbag2_2025_02_10-12_28_39_0/rosbag2_2025_02_10-12_28_39_0.db3'
    bag_file = '/root/unitree/unitree_rl_gym/rosbag2_2025_02_12-07_40_15/rosbag2_2025_02_12-07_40_15_0.db3'
    
    # Set up the rosbag2_py reader.
    storage_options = StorageOptions(uri=bag_file, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr',
                                         output_serialization_format='cdr')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Prepare output directories and files.
    image_dir = "images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    tf_output_filename = "tf_messages.txt"
    zed_csv_filename = "frame_zed_mapping.csv"
    lidar_csv_filename = "frame_lidar_mapping.csv"
    
    # Open a file to dump TF messages in human-readable form.
    tf_file = open(tf_output_filename, "w")
    
    # Lists to store messages with their bag timestamps.
    tf_msgs = []      # Each element: (timestamp, TFMessage)
    image_msgs = []   # Each element: (timestamp, image_filename)
    
    bridge = CvBridge()
    
    print("Processing bag file offline and saving TF messages and images...")
    # Iterate over each message in the bag.
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == '/tf':
            # Deserialize the TF message.
            msg = deserialize_message(data, TFMessage)
            tf_msgs.append((timestamp, msg))
            # Write out each transform in a human‑readable format.
            for transform in msg.transforms:
                output = (
                    f"Time: {transform.header.stamp.sec}.{transform.header.stamp.nanosec}\n"
                    f"Frame ID: {transform.header.frame_id}\n"
                    f"Child Frame ID: {transform.child_frame_id}\n"
                    f"Translation: x={transform.transform.translation.x}, "
                    f"y={transform.transform.translation.y}, "
                    f"z={transform.transform.translation.z}\n"
                    f"Rotation: x={transform.transform.rotation.x}, "
                    f"y={transform.transform.rotation.y}, "
                    f"z={transform.transform.rotation.z}, "
                    f"w={transform.transform.rotation.w}\n"
                    f"{'-'*50}\n"
                )
                tf_file.write(output)
        elif topic == '/zed/zed_node/rgb/image_rect_color':
            # Deserialize the Image message.
            msg = deserialize_message(data, Image)
            # Convert the ROS image to an OpenCV image.
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                print(f"Error converting image: {e}")
                continue
            # Save the image file using a zero-padded counter.
            filename = os.path.join(image_dir, f"frame_{len(image_msgs):04d}.png")
            cv2.imwrite(filename, cv_image)
            image_msgs.append((timestamp, filename))
            print(f"Saved image {filename}")
    
    tf_file.close()
    print("Finished processing bag file.")
    
    # Ensure the lists are sorted by timestamp (they should be already, but sorting for safety).
    tf_msgs.sort(key=lambda x: x[0])
    image_msgs.sort(key=lambda x: x[0])
    
    # Build a CSV mapping file.
    # For each image frame, find the most recent TF message (with timestamp <= image timestamp)
    # that contains the transform from odom to zed_camera_link.
    with open(zed_csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write CSV header.
        csv_writer.writerow([
            "frame_index", "image_filename", "image_timestamp", "tf_timestamp",
            "rel_translation_x", "rel_translation_y", "rel_translation_z",
            "rel_rotation_qx", "rel_rotation_qy", "rel_rotation_qz", "rel_rotation_qw"
        ])
        T_first = None
        for idx, (img_timestamp, img_filename) in enumerate(image_msgs):
            tf_result = find_most_recent_tf_for_relation(tf_msgs, img_timestamp, "odom", "zed_camera_link")
            if tf_result is not None:
                tf_timestamp, transform = tf_result
                T = tf_to_matrix(transform)
                if T_first is None:
                    T_first = T  # Save the first valid transform as reference.
                # Compute relative transform: T_rel = inv(T_first) * T.
                T_rel = np.linalg.inv(T_first) @ T
                trans = T_rel[:3, 3]
                R_rel = T_rel[:3, :3]
                quat = rotationMatrixToQuaternion(R_rel)
                csv_writer.writerow([
                    idx,
                    img_filename,
                    img_timestamp,
                    tf_timestamp,
                    trans[0],
                    trans[1],
                    trans[2],
                    quat[0],
                    quat[1],
                    quat[2],
                    quat[3]
                ])
            else:
                csv_writer.writerow([
                    idx, img_filename, img_timestamp, "None",
                    "None", "None", "None", "None", "None", "None", "None"
                ])
    
    with open(lidar_csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write CSV header.
        csv_writer.writerow([
            "frame_index", "image_filename", "image_timestamp", "tf_timestamp",
            "rel_translation_x", "rel_translation_y", "rel_translation_z",
            "rel_rotation_qx", "rel_rotation_qy", "rel_rotation_qz", "rel_rotation_qw"
        ])
        T_first = None
        for idx, (img_timestamp, img_filename) in enumerate(image_msgs):
            # Look for the most recent TF message with parent "camera_init" and child "body"
            tf_result = find_most_recent_tf_for_relation(tf_msgs, img_timestamp, "camera_init", "body")
            if tf_result is not None:
                tf_timestamp, transform = tf_result
                T = tf_to_matrix(transform)
                if T_first is None:
                    T_first = T  # Save the first valid transform as reference.
                # Compute the relative transform: T_rel = inv(T_first) * T.
                T_rel = np.linalg.inv(T_first) @ T
                trans = T_rel[:3, 3]
                R_rel = T_rel[:3, :3]
                quat = rotationMatrixToQuaternion(R_rel)
                csv_writer.writerow([
                    idx,
                    img_filename,
                    img_timestamp,
                    tf_timestamp,
                    trans[0],
                    trans[1],
                    trans[2],
                    quat[0],
                    quat[1],
                    quat[2],
                    quat[3]
                ])
            else:
                csv_writer.writerow([
                    idx, img_filename, img_timestamp, "None",
                    "None", "None", "None", "None", "None", "None", "None"
                ])


if __name__ == '__main__':
    main()
