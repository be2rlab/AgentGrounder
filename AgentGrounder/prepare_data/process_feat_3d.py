import argparse
import json
import os
import pickle
import sys
import torch
import numpy as np
from tqdm import tqdm

# Import custom utilities and models
from models.pcd_classifier import PcdClassifier
from prepare_data.utils import load_point_cloud


def read_file_to_list(file_path):
    """
    Read a file and convert its contents to a sorted list.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        List[str]: A sorted list of lines from the file.
    """
    with open(file_path, "r") as file:
        return sorted(file.read().splitlines())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="3D Point Cloud Feature Extraction")
    parser.add_argument(
        "--scene_list",
        type=str,
        default="PCGrounder/data/scannet/scannetv2_val.txt",
        help="Path to the scene list file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="PCGrounder/weights/pcd_clasifier/pnext_cls.pth",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="PCGrounder/data/nr3d/feats_3d.pkl",
        help="Output path for saving features.",
    )
    parser.add_argument(
        "--scan_dir",
        default='/remote-home/share/vg_datasets/referit3d/scan_data'
        # default="/mnt/ssd1/rongl/datasets/referit3d/scan_data/",
    )

    args = parser.parse_args()

    # Load the validation scene list
    scene_ids = read_file_to_list(args.scene_list)
    print(f"Number of scenes: {len(scene_ids)}")

    # Initialize model and load weights
    model = PcdClassifier().cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
    print(f"Loaded model weights from {args.checkpoint}")

    model.eval()
    torch.backends.cudnn.benchmark = False

    data = {}

    # Process each scan in the scene list
    for scan_id in tqdm(scene_ids):

        if scan_id in ["scene0217_00"]:
            continue
        batch_labels, obj_ids, inst_locs, center, batch_pcds = load_point_cloud(
            scan_id, keep_background=False, scan_dir=args.scan_dir
        )

        # Extract object embeddings
        obj_embeds = model(batch_pcds[..., :4].cuda())  # (B, D)
        obj_embeds = torch.nn.functional.normalize(obj_embeds, p=2, dim=-1)

        data[scan_id] = {
            "batch_labels": batch_labels,
            "obj_ids": obj_ids,
            "inst_locs": inst_locs,
            "center": center,
            "obj_embeds": obj_embeds.detach().cpu(),
        }

    # Save extracted features to file
    with open(args.output, "wb") as f:
        pickle.dump(data, f)
    print(f"Features saved to {args.output}")
