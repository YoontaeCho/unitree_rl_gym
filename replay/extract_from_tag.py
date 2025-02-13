#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import glob
import csv
import argparse
from dt_apriltags import Detector

# ----- Camera Intrinsics (from your camera_info.csv) -----
# fx = 260.4433, fy = 260.4433, cx = 313.7494, cy = 169.0348, assume zero distortion.
camera_matrix = np.array([[260.4433, 0.0,      313.7494],
                           [0.0,      260.4433, 169.0348],
                           [0.0,      0.0,      1.0]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# ----- AprilTag physical size (in meters) -----
# 113.3 mm = 0.1133 m
tag_size = 0.1133

# ----- Define AprilTag world poses -----
# Using tags from tag16_05_00000 to tag16_05_00011.
# Arrangement (based on the last two digits):
#   Top row:    10   8   6   4   2   0
#   Bottom row: 11   9   7   5   3   1
# We assume the tags are aligned horizontally with 0.6 m offset between columns.
# And the two tags on the same sheet have a vertical offset of 0.1433 m.
# We define the world coordinate system such that tag 0 (top row, rightmost column) is at (0,0,0)
# with identity rotation.
def create_tag_pose(x, y, z=0.0):
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T

# Define tag IDs using the last two digits.
top_tags = [10, 8, 6, 4, 2, 0]
bottom_tags = [11, 9, 7, 5, 3, 1]
april_tag_poses = {}
for col in range(6):
    # Column index: 0 (leftmost) to 5 (rightmost); here x = (col - 5)*0.6 so that tag 0 gets x = 0.
    x = (col - 5) * 0.6
    # Top row: y = 0.
    tag_id = top_tags[col]
    april_tag_poses[tag_id] = create_tag_pose(x, 0.0)
    # Bottom row: y = 0.1433.
    tag_id = bottom_tags[col]
    april_tag_poses[tag_id] = create_tag_pose(x, 0.1433)

# ----- AprilTag Detector Initialization -----
# Make sure dt-apriltags is installed: pip install dt-apriltags
detector = Detector(families="tag16h5", nthreads=4, quad_decimate=1.0,
                    quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

# ----- Helper: Get the 3D object points for the tag (in its coordinate frame) -----
def get_tag_object_points(tag_size):
    half_size = tag_size / 2.0
    # Coordinates in the tag frame (centered on the tag, in the XY plane, Z=0)
    object_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)
    return object_points

# ----- Helper: Convert a rotation matrix to quaternion (qx, qy, qz, qw) -----
def rotationMatrixToQuaternion(R):
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

# ----- Helper: Compute camera pose in tag frame from solvePnP outputs,
# then flip the z axis (and modify x,y accordingly) to maintain a right-handed frame.
#
# Given solvePnP yields (rvec, tvec) for:
#   X_tag = R_tag_cam * X_cam + tvec,
# the standard inversion is:
#   R_cam_tag = R_tag_camᵀ,   t_cam_tag = -R_tag_camᵀ * tvec.
#


def compute_camera_pose_from_solvePnP(rvec, tvec):
    R_tag_cam, _ = cv2.Rodrigues(rvec)
    # Standard inversion:
    R_cam_tag = R_tag_cam.T
    t_cam_tag = -R_cam_tag @ tvec
    # Define flip matrix F(why? April tag's z frmae is pointing into the ground)
    F = np.diag([1, 1, -1])
    R_cam_tag_new = F @ R_cam_tag @ F
    t_cam_tag_new = F @ t_cam_tag
    T_cam_tag = np.eye(4, dtype=np.float32)
    T_cam_tag[:3, :3] = R_cam_tag_new
    T_cam_tag[:3, 3] = t_cam_tag_new.flatten()
    return T_cam_tag

# ----- Main processing -----
def main():
    parser = argparse.ArgumentParser(description="Compute and save relative camera poses for each frame.")
    parser.add_argument("--output", default="frame_pose_mapping.csv",
                        help="Output CSV filename (default: frame_pose_mapping.csv)")
    args = parser.parse_args()

    # Get list of image files.
    image_files = sorted(glob.glob("images/frame_*.png"))
    if not image_files:
        print("No image files found in 'images' folder.")
        sys.exit(0)

    # Prepare list to store results.
    # Each result: (image_file, tx, ty, tz, qx, qy, qz, qw, valid)
    results = []

    # Use the first valid frame's world pose as the odom frame.
    odom_pose = None

    # Process each image.
    for image_file in image_files:
        print("Processing", image_file)
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("  Failed to load", image_file)
            results.append((image_file, 0, 0, 0, 0, 0, 0, 0, 0))
            continue

        detections = detector.detect(image)
        valid_detections = [d for d in detections if d.tag_id in april_tag_poses]
        if valid_detections:
            best_detection = max(valid_detections, key=lambda d: d.decision_margin)
            if best_detection.decision_margin < 30:
                print("  Best detection decision margin below threshold:", best_detection.decision_margin)
                results.append((image_file, 0, 0, 0, 0, 0, 0, 0, 0))
                continue
            print("  Using tag", best_detection.tag_id, "with decision margin", best_detection.decision_margin)
            image_points = np.array(best_detection.corners, dtype=np.float32)
            object_points = get_tag_object_points(tag_size)
            ret, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                           camera_matrix, dist_coeffs)
            if not ret:
                print("  solvePnP failed for", image_file)
                results.append((image_file, 0, 0, 0, 0, 0, 0, 0, 0))
                continue

            # Compute camera pose in tag frame.
            T_cam_tag = compute_camera_pose_from_solvePnP(rvec, tvec)
            tag_id = best_detection.tag_id
            T_tag_world = april_tag_poses[tag_id]
            T_cam_world = T_tag_world @ T_cam_tag

            if odom_pose is None:
                odom_pose = T_cam_world

            # Compute relative transformation: T_rel = inv(odom_pose) * T_cam_world.
            T_rel = np.linalg.inv(odom_pose) @ T_cam_world
            trans = T_rel[:3, 3]
            R_rel = T_rel[:3, :3]
            quat = rotationMatrixToQuaternion(R_rel)
            valid_flag = 1
            results.append((image_file, trans[0], trans[1], trans[2],
                            quat[0], quat[1], quat[2], quat[3], valid_flag))
        else:
            print("  No valid AprilTag detected in", image_file)
            results.append((image_file, 0, 0, 0, 0, 0, 0, 0, 0))

    # Write results to CSV.
    with open(args.output, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ["image_file", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "valid"]
        csv_writer.writerow(header)
        for row in results:
            csv_writer.writerow(row)
    
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
