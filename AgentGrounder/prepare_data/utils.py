import argparse
import json
import os
import pickle
import sys
import torch
import numpy as np
from tqdm import tqdm


def load_point_cloud(scan_id, keep_background=False, scan_dir=None):
    """
    Load point cloud data for a given scan ID.

    Parameters:
        scan_id (str): The ID of the scan.
        keep_background (bool, optional): Whether to keep background objects. Default is False.
        scan_dir (str, optional): Directory containing the scan data.

    Returns:
        tuple: A tuple containing:
            - batch_labels (List[str]): List of object labels.
            - obj_ids (List[int]): List of object IDs.
            - inst_locs (List[np.ndarray]): List of object instance locations and sizes.
            - center (np.ndarray): The center of the point cloud.
            - batch_pcds (torch.Tensor): Batch of point clouds with sampled points.
    """
    pcds, colors, _, instance_labels = torch.load(
        os.path.join(scan_dir, "pcd_with_global_alignment", f"{scan_id}.pth")
    )
    obj_labels = json.load(
        open(os.path.join(scan_dir, "instance_id_to_name", f"{scan_id}.json"))
    )

    origin_pcds = []
    batch_pcds = []
    batch_labels = []
    inst_locs = []
    obj_ids = []

    for i, obj_label in enumerate(obj_labels):
        if not keep_background and obj_label in ["wall", "floor", "ceiling"]:
            continue
        mask = instance_labels == i
        assert np.sum(mask) > 0, f"scan: {scan_id}, obj {i}"

        obj_pcd = pcds[mask]
        obj_color = colors[mask]
        origin_pcds.append(np.concatenate([obj_pcd, obj_color], axis=1))

        # Object instance location (center and size)
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], axis=0))

        # Normalize point cloud
        obj_pcd -= obj_pcd.mean(0)
        max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, axis=1)))
        max_dist = 1 if max_dist < 1e-6 else max_dist
        obj_pcd /= max_dist
        obj_color = obj_color / 127.5 - 1

        # Sample points
        pcd_idxs = np.random.choice(
            len(obj_pcd), size=2048, replace=(len(obj_pcd) < 2048)
        )
        obj_pcd = obj_pcd[pcd_idxs]
        obj_color = obj_color[pcd_idxs]
        obj_height = obj_pcd[:, 2:3] - obj_pcd[:, 2:3].min()

        batch_pcds.append(np.concatenate([obj_pcd, obj_height, obj_color], axis=1))
        batch_labels.append(obj_label)
        obj_ids.append(i)

    batch_pcds = torch.from_numpy(np.stack(batch_pcds, axis=0))
    center = (pcds.max(0) + pcds.min(0)) / 2

    return batch_labels, obj_ids, inst_locs, center, batch_pcds
