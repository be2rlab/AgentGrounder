from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


def remove_ceiling(scan_pc):
    scan_pc = scan_pc.copy()

    scene_planes = detect_planes(scan_pc[:, :3])
    scene_ceiling = extract_ceiling(scene_planes, (scan_pc[:, 2].min(), scan_pc[:, 2].max()))

    if scene_ceiling:
        for i, ceiling_part in enumerate(scene_ceiling):
            filtered_pcd, filtered_pcd_colors = remove_points_in_bbox(scan_pc[:, :3], scan_pc[:, 3:], scene_ceiling[i]['bbox'])
            # filtered_pcd, filtered_pcd_colors = remove_points_by_height(scan_pc[:, :3], scan_pc[:, 3:], scene_ceiling[i]['bbox']['z'][0])
            scan_pc = np.zeros((filtered_pcd.shape[0], 6))
            scan_pc[:, :3] = filtered_pcd
            scan_pc[:, 3:] = filtered_pcd_colors

    return scan_pc

def detect_planes(
    pcd: np.ndarray,
    distance_threshold: float = 0.05,
    min_points: int = 100,
    max_planes: int = 10,
):
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)

    planes = []
    remaining_pcd = o3d_pcd

    for i in range(max_planes):
        if len(remaining_pcd.points) < min_points:
            break

        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=10000
        )
        if len(inliers) < min_points:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        norm = np.linalg.norm(normal)
        normal = normal / norm
        distance = d / norm

        plane_info = {
            "normal": normal.tolist(),
            "distance": distance,
            "inlier_count": len(inliers),
            "points": np.asarray(remaining_pcd.points)[inliers],
        }
        planes.append(plane_info)

        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    return planes


def extract_ceiling(
    planes: List[Dict],
    z_range: Tuple[float, float],
    vertical_threshold: float = 0.2,
    min_points_ratio: float = 0.0001,
    lowest_point_threshold: float = 0.85,
) -> Optional[List[Dict]]:

    z_min, z_max = z_range
    height = z_max - z_min
    threshold_z = z_min + height * lowest_point_threshold
    best_plane = None
    max_inliers = 0

    for plane in planes:
        normal = np.asarray(plane["normal"])
        points = plane["points"]

        if abs(normal[2]) < vertical_threshold:
            continue

        plane_min_z = np.min(points[:, 2])
        if plane_min_z <= threshold_z:
            continue

        inliers = plane.get("inlier_count", len(points))

        if best_plane is not None and inliers < max_inliers * min_points_ratio:
            continue

        if inliers > max_inliers:
            max_inliers = inliers
            best_plane = plane

    if best_plane is None:
        return None

    normal = np.asarray(best_plane["normal"])
    points = best_plane["points"]
    mean_z_val = float(np.mean(points[:, 2]))
    inliers_count = best_plane.get("inlier_count", len(points))

    bbox = {
        "x": [float(points[:, 0].min()), float(points[:, 0].max())],
        "y": [float(points[:, 1].min()), float(points[:, 1].max())],
        "z": [float(points[:, 2].min()), float(points[:, 2].max())],
    }

    return [
        {
            "z": mean_z_val,
            "normal": normal.tolist(),
            "distance": best_plane["distance"],
            "inliers": inliers_count,
            "mean_z": mean_z_val,
            "bbox": bbox,
        }
    ]


def remove_points_in_bbox(pcd, colors, bbox):
    x_min, x_max = bbox["x"]
    y_min, y_max = bbox["y"]
    z_min_b, z_max_b = bbox["z"]

    mask = ~(
        (pcd[:, 0] >= x_min)
        & (pcd[:, 0] <= x_max)
        & (pcd[:, 1] >= y_min)
        & (pcd[:, 1] <= y_max)
        & (pcd[:, 2] >= z_min_b)
        & (pcd[:, 2] <= z_max_b)
    )

    return pcd[mask], colors[mask]


def remove_points_by_height(pcd, colors, ceiling_z, tolerance=-0.3):
    mask = pcd[:, 2] < (ceiling_z + tolerance)
    return pcd[mask], colors[mask]
