#!/usr/bin/env python3
import csv
import numpy as np
import open3d as o3d
import glob
import argparse
import math

# ============================================================
# Define the 4x4 transformation matrices that map each CSV file's local frame into the global frame.
# Modify these matrices as needed.
# For now, we'll assume:
#   T_lidar: transformation for frame_lidar_mapping.csv (default: identity)
#   T_pose: transformation for frame_pose_mapping.csv (default: identity)
#   T_zed: transformation for frame_zed_mapping.csv (default: identity)
# ============================================================
# T_lidar = np.eye(4, dtype=np.float32)
theta = math.radians(35.)
T_lidar = np.array([[math.cos(theta), 0, math.sin(theta), 0],
                    [0,              -1, 0,              0],
                    [math.sin(theta), 0, -math.cos(theta), 0],
                    [0,               0, 0,               1]], dtype=np.float32)

T_pose = np.array([[0, -1, 0, 0],
                [-1, 0, 0, 0],
                [0,  0, -1, 0],
                [0,  0, 0, 1]], dtype=np.float32)
T_zed   = np.eye(4, dtype=np.float32)

# ============================================================
def load_trajectory_from_csv(csv_filename, translation_indices, valid_index=None, local2global=None):
    """
    Load a trajectory from a CSV file, converting each point from the local frame
    to the global frame using the provided transformation.
    
    Parameters:
      csv_filename: str, path to CSV file.
      translation_indices: tuple of three integers specifying the columns for x, y, z.
      valid_index: (optional) integer specifying the column index that holds the valid flag.
                   Only rows with a valid flag of 1 (or convertible to 1) are kept.
      local2global: (optional) a 4x4 numpy array representing the transformation from
                    the CSV's local frame to the global frame.
                    
    Returns:
      points: np.ndarray of shape (N,3) with the transformed translations.
      names: list of corresponding image/file names (assumed to be in column index 1).
    """
    points = []
    names = []
    with open(csv_filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if valid_index is not None:
                try:
                    flag = float(row[valid_index])
                    if flag < 1:
                        continue
                except ValueError:
                    continue
            try:
                x = float(row[translation_indices[0]])
                y = float(row[translation_indices[1]])
                z = float(row[translation_indices[2]])
            except ValueError:
                continue
            point = np.array([x, y, z, 1.0], dtype=np.float32)
            if local2global is not None:
                point = local2global @ point
            points.append(point[:3].tolist())
            names.append(row[1])
    return np.array(points), names

def create_lineset_from_points(points, color):
    """
    Given an Nx3 array of points, create an Open3D LineSet connecting consecutive points.
    """
    if len(points) < 2:
        return None
    lines = [[i, i+1] for i in range(len(points)-1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = [color for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_connection_lineset(points1, points2, color):
    """
    Given two arrays of points (Nx3) of the same length, create an Open3D LineSet that
    connects each corresponding pair (points1[i] to points2[i]) with a separate line.
    """
    n = min(len(points1), len(points2))
    combined = []
    connection_lines = []
    for i in range(n):
        combined.append(points1[i])
        combined.append(points2[i])
        connection_lines.append([2*i, 2*i+1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(combined))
    line_set.lines = o3d.utility.Vector2iVector(connection_lines)
    colors = [color for _ in connection_lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def main():
    parser = argparse.ArgumentParser(description="Visualize three pose trajectories in a global frame and connect corresponding points.")
    parser.add_argument("--lidar_csv", default="frame_lidar_mapping.csv",
                        help="CSV file for the LIDAR trajectory (default: frame_lidar_mapping.csv)")
    parser.add_argument("--pose_csv", default="frame_pose_mapping.csv",
                        help="CSV file for the VIO pose trajectory (default: frame_pose_mapping.csv)")
    parser.add_argument("--zed_csv", default="frame_zed_mapping.csv",
                        help="CSV file for the ZED trajectory (default: frame_zed_mapping.csv)")
    args = parser.parse_args()

    # Load trajectories:
    # For frame_lidar_mapping.csv, translation columns are indices 4,5,6.
    points_lidar, names_lidar = load_trajectory_from_csv(args.lidar_csv, (4,5,6), local2global=T_lidar)
    # For frame_pose_mapping.csv, translation columns are indices 1,2,3 and valid flag is in column 8.
    points_pose, names_pose = load_trajectory_from_csv(args.pose_csv, (1,2,3), valid_index=8, local2global=T_pose)
    # For frame_zed_mapping.csv, translation columns are assumed to be indices 4,5,6.
    points_zed, names_zed = load_trajectory_from_csv(args.zed_csv, (4,5,6), local2global=T_zed)
    
    print(f"Loaded {len(points_lidar)} points from {args.lidar_csv}")
    print(f"Loaded {len(points_pose)} valid points from {args.pose_csv}")
    print(f"Loaded {len(points_zed)} points from {args.zed_csv}")
    
    # Create line sets for each trajectory.
    ls_lidar = create_lineset_from_points(points_lidar, [1, 0, 0])    # red for LIDAR trajectory
    ls_pose  = create_lineset_from_points(points_pose, [0, 0, 1])      # blue for april tag trajectory
    ls_zed   = create_lineset_from_points(points_zed, [0, 1, 0])     # orange for ZED trajectory
    
    # Create connecting line segments between corresponding points:
    # (VIO pose <-> LIDAR) in green, and (VIO pose <-> ZED) in cyan.
    ls_connect_pose_lidar = create_connection_lineset(points_pose, points_lidar, [0, 1, 1])
    ls_connect_pose_zed   = create_connection_lineset(points_pose, points_zed, [1, 0, 1])
    ls_connect_zed_lidar   = create_connection_lineset(points_zed, points_lidar, [1, 1, 0])
    
    # Create a coordinate frame for reference.
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    geometries = [coord_frame]
    if ls_lidar is not None:
        geometries.append(ls_lidar)
    if ls_pose is not None:
        geometries.append(ls_pose)
    if ls_zed is not None:
        geometries.append(ls_zed)
    if ls_connect_pose_lidar is not None:
        geometries.append(ls_connect_pose_lidar)
    if ls_connect_pose_zed is not None:
        geometries.append(ls_connect_pose_zed)
    if ls_connect_zed_lidar is not None:
        geometries.append(ls_connect_zed_lidar)
    
    o3d.visualization.draw_geometries(geometries,
                                      window_name="Combined Trajectories",
                                      width=1000, height=800, left=50, top=50,
                                      point_show_normal=False)

if __name__ == '__main__':
    main()
