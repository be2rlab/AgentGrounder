import os
import json
import pickle
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import sys

from data.scannet200_constants import CLASS_LABELS_200
from prepare_data.utils import load_point_cloud


def load_pred_ins(
    scan_id,
    normalize=True,
    use_scannet200=False,
    mask3d_pred="/remote-home/share/vg_datasets/mask3d_pred/",
):
    root_dir = os.path.join(mask3d_pred, "Mask3d/scannet")
    if use_scannet200:
        root_dir = os.path.join(mask3d_pred, "Mask3d/scannet200")
    data = np.load(os.path.join(root_dir, scan_id + ".npz"), allow_pickle=True)
    batch_labels = data["ins_labels"]

    batch_pcds = []
    inst_locs = []
    scene_pc = []

    for i, obj in enumerate(data["ins_pcds"]):
        if obj.shape[0] == 0:
            obj = np.zeros((1, 6))
        obj_pcd = obj[:, :3]
        scene_pc.append(obj_pcd)
        obj_color = obj[:, 3:6]
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))

        height_array = obj_pcd[:, 2:3] - obj_pcd[:, 2:3].min()

        # normalize
        if normalize:
            obj_pcd = obj_pcd - obj_pcd.mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd**2, 1)))
            if max_dist < 1e-6:  # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd = obj_pcd / max_dist
            obj_color = obj_color / 127.5 - 1

        # sample points
        pcd_idxs = np.random.choice(
            len(obj_pcd), size=2048, replace=(len(obj_pcd) < 2048)
        )
        obj_pcd = obj_pcd[pcd_idxs]
        obj_color = obj_color[pcd_idxs]
        obj_height = height_array[pcd_idxs]

        batch_pcds.append(
            np.concatenate(
                [
                    obj_pcd,
                    obj_height,
                    obj_color,
                ],
                1,
            )
        )

    batch_pcds = torch.from_numpy(np.stack(batch_pcds, 0))
    scene_pc = np.concatenate(scene_pc, 0)
    center = (scene_pc.max(0) + scene_pc.min(0)) / 2

    return batch_labels, inst_locs, center, batch_pcds


def convert_to_serializable(obj):
    """
    Convert non-serializable objects to serializable ones.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def read_file_to_list(file_path):
    """
    Read a file and convert its contents to a list.

    Parameters:
    - file_path: The path to the file.

    Returns:
    - A list containing each line of the file as an element.
    """
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
    return sorted(lines)


def scanrefer_gt(scan_ids, args):

    output_dir = os.path.join(args.output_dir, "gt")
    os.makedirs(output_dir, exist_ok=True)

    for scan_id in tqdm(scan_ids):
        if scan_id in ["scene0217_00"]:
            continue
        print(scan_id)
        batch_labels, obj_ids, inst_locs, center, batch_pcds = load_point_cloud(
            scan_id, scan_dir=args.scan_dir
        )

        res_list = []
        for obj_id, loc, label in zip(obj_ids, inst_locs, batch_labels):
            res_list.append(
                convert_to_serializable(
                    {
                        # 'scan_id': scan_id,
                        "bbox_id": obj_id,
                        "target": label,
                        "bbox_3d": list(loc),
                    }
                )
            )

        output_file_path = os.path.join(output_dir, f"{scan_id}.json")
        with open(output_file_path, "w") as output_file:
            json.dump(res_list, output_file, indent=4)

        print(f"Saved {output_file_path}")
    return


def convert_to_serializable(obj):
    """
    Convert non-serializable objects to serializable ones.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def scanrefer_pred(scan_ids, args):
    output_dir = os.path.join(args.output_dir, "pred")
    os.makedirs(output_dir, exist_ok=True)

    for scan_id in scan_ids:
        print(scan_id)

        pred_labels, inst_locs, center, batch_pcds = load_pred_ins(
            scan_id, use_scannet200=True, mask3d_pred=args.mask3d_pred
        )
        verified_boxes = []
        for i, obj_name in enumerate(pred_labels):
            if not obj_name in ["wall", "floor", "ceiling", "object", "objects"]:
                verified_boxes.append(
                    {"bbox_id": i, "target": obj_name, "bbox_3d": inst_locs[i]}
                )

        bboxes = convert_to_serializable(verified_boxes)

        output_file_path = os.path.join(output_dir, f"{scan_id}.json")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w") as output_file:
            json.dump(bboxes, output_file, indent=4)
        print(f"Saved {output_file_path}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Classifier for 3D Scenes")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="Name of the tokenizer model.",
    )
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="Name of the CLIP model.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="PCGrounder/data/scannet/scannetv2_val.txt",
        help="Path to the validation split file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="PCGrounder/data/scanrefer/object_lookup_table/",
        help="Directory to save output results.",
    )
    parser.add_argument(
        "--feat_file",
        type=str,
        default="PCGrounder/data/nr3d/feats_3d.pkl",
        help="Path to the 3D features pickle file.",
    )
    parser.add_argument(
        "--scan_dir", default="/remote-home/share/vg_datasets/referit3d/scan_data"
    )
    parser.add_argument(
        "--mask3d_pred", default="/remote-home/share/vg_datasets/mask3d_pred/"
    )

    args = parser.parse_args()
    scan_ids = read_file_to_list(args.val_file)
    print(len(scan_ids))

    # gt
    scanrefer_gt(scan_ids, args)

    # pred
    scanrefer_pred(scan_ids, args)
