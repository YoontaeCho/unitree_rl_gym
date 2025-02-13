#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import argparse

# Import the Duckietown AprilTag detector from lib-dt-apriltags.
from dt_apriltags import Detector

# -----------------------------------------------------
# Camera Intrinsics (from camera_info.csv):
# fx = 260.4433, fy = 260.4433, cx = 313.7494, cy = 169.0348, assume zero distortion.
# -----------------------------------------------------
camera_matrix = np.array([[260.4433, 0.0,      313.7494],
                          [0.0,      260.4433, 169.0348],
                          [0.0,      0.0,      1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# -----------------------------------------------------
# AprilTag physical size in meters (113.3 mm)
# -----------------------------------------------------
tag_size = 0.1133

# -----------------------------------------------------
# Define the AprilTag world poses.
# The tags are arranged in two rows:
#   Top row (row 0): tags [10, 8, 6, 4, 2, 0] from left to right.
#   Bottom row (row 1): tags [11, 9, 7, 5, 3, 1].
# We define tag 0 (top row, rightmost column) to be at (0,0) with identity rotation.
# Horizontal spacing is 0.6 m and vertical offset (between tags on the same sheet) is 0.1433 m.
# -----------------------------------------------------
def create_tag_pose(x, y, z=0.0):
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T

top_tags = [10, 8, 6, 4, 2, 0]
bottom_tags = [11, 9, 7, 5, 3, 1]
april_tag_poses = {}
for col in range(6):
    x = -(col - 5) * 0.6  # Column 5 -> x=0; decreasing to the left.
    # Top row (y = 0)
    tag_id_top = top_tags[col]
    april_tag_poses[tag_id_top] = create_tag_pose(x, 0.0)
    # Bottom row (y = -0.1433)
    tag_id_bottom = bottom_tags[col]
    april_tag_poses[tag_id_bottom] = create_tag_pose(x, 0.1433)

# -----------------------------------------------------
# Helper: Get the 3D object points for the tag (in its own coordinate system).
# Order: top-left, top-right, bottom-right, bottom-left.
# -----------------------------------------------------
def get_tag_object_points(tag_size):
    half_size = tag_size / 2.0
    object_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)
    return object_points

# -----------------------------------------------------
# Helper: Convert a rotation matrix to Euler angles (roll, pitch, yaw) in radians.
# -----------------------------------------------------
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0
    return roll, pitch, yaw

# -----------------------------------------------------
# Helper: Flip the z axis of a 4x4 homogeneous transformation.
# This redefines the tag coordinate system so that the camera pose’s z becomes positive.
# -----------------------------------------------------
def flip_z_axis(T):
    # Create a flip matrix for z axis.
    R_flip = np.diag([-1, 1, -1])
    T_flipped = np.eye(4, dtype=np.float32)
    T_flipped[:3, :3] = R_flip
    # print(
    #     T_flipped,
    #     T,
    #     T @ T_flipped,
    #     T_flipped @ T,


    # )
    return T_flipped @ T
    # return T

# -----------------------------------------------------
# Main function.
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute and print camera pose for a given frame.")
    parser.add_argument("frame_number", type=int, help="Frame number (e.g., 0 for frame_0000.png)")
    args = parser.parse_args()

    # Construct the image filename.
    image_file = os.path.join("images", f"frame_{args.frame_number:04d}.png")
    if not os.path.exists(image_file):
        print(f"Error: {image_file} not found.")
        sys.exit(1)
    
    # Load image in grayscale.
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load {image_file}.")
        sys.exit(1)

    print(f"Processing {image_file} ...")
    
    # Initialize the Duckietown AprilTag detector.
    detector = Detector(families="tag16h5", nthreads=4, quad_decimate=1.0,
                        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
    
    # Detect tags.
    detections = detector.detect(image)
    if len(detections) == 0:
        print("No AprilTags detected in this frame.")
        sys.exit(0)

    # Get 3D object points for the tag.
    object_points = get_tag_object_points(tag_size)

    # Process each detection in the image that is in our known set (tags 0..11).
    for detection in detections:
        tag_id = detection.tag_id
        if tag_id not in april_tag_poses:
            continue

        print(f"\nTag ID: {tag_id}")
        print("Detected image corners:")
        for i, corner in enumerate(detection.corners):
            print(f"  Corner {i}: {corner}")

        # Convert corners to NumPy array.
        image_points = np.array(detection.corners, dtype=np.float32)

        # Use solvePnP to compute the pose of the tag relative to the camera.
        ret, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                       camera_matrix, dist_coeffs)
        if not ret:
            print("solvePnP failed for this tag.")
            continue

        # Build the transformation from tag to camera frame: T_tag_cam.
        T_tag_cam = np.eye(4, dtype=np.float32)
        R_tag_cam, _ = cv2.Rodrigues(rvec)
        T_tag_cam[:3, :3] = R_tag_cam
        T_tag_cam[:3, 3] = tvec.flatten()

        # Invert to get the camera pose in the tag frame.
        T_cam_tag = np.linalg.inv(T_tag_cam)
        # Apply a flip on the z axis so that the camera pose shows a positive z.
        T_cam_tag_flipped = flip_z_axis(T_cam_tag)

        # Decompose T_cam_tag_flipped.
        trans_tag = T_cam_tag_flipped[:3, 3]
        R_cam_tag = T_cam_tag_flipped[:3, :3]
        roll_tag, pitch_tag, yaw_tag = rotationMatrixToEulerAngles(R_cam_tag)

        print("\nCamera Pose in Detected Tag Frame (after z flip):")
        print(f"Translation (x, y, z): {trans_tag[0]:.3f}, {trans_tag[1]:.3f}, {trans_tag[2]:.3f}")
        print(f"Rotation (roll, pitch, yaw in rad): {roll_tag:.3f}, {pitch_tag:.3f}, {yaw_tag:.3f}")

        # Compute the camera pose in the world frame (defined by tag 0).
        # Get the known world pose for the detected tag.
        T_tag_world = april_tag_poses[tag_id]
        # Use the flipped camera pose: T_cam_world = T_tag_world * T_cam_tag_flipped.
        T_cam_world = np.dot(T_tag_world, T_cam_tag_flipped)
        trans_world = T_cam_world[:3, 3]
        R_cam_world = T_cam_world[:3, :3]
        roll_world, pitch_world, yaw_world = rotationMatrixToEulerAngles(R_cam_world)

        print("\nCamera Pose in World Frame (Tag 0 frame):")
        print(f"Translation (x, y, z): {trans_world[0]:.3f}, {trans_world[1]:.3f}, {trans_world[2]:.3f}")
        print(f"Rotation (roll, pitch, yaw in rad): {roll_world:.3f}, {pitch_world:.3f}, {yaw_world:.3f}")
    
if __name__ == '__main__':
    main()
