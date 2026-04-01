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
from transformers import AutoTokenizer, CLIPModel


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


class ObjectClassifier(nn.Module):
    """
    Object Classifier for processing 3D scenes and generating object classification results.
    """

    def __init__(self, args):
        super().__init__()
        self.use2d = True
        self.output_dir = os.path.join(args.output_dir, "pred")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, model_max_length=512, use_fast=True
        )
        self.clip = CLIPModel.from_pretrained(args.clip_model_name).cuda()
        print(f"Loaded CLIP model from {args.clip_model_name}")

        # Prepare class labels and embeddings
        self.class_name_list = [
            cls for cls in CLASS_LABELS_200 if cls not in ["wall", "floor", "ceiling"]
        ]
        self.label_lang_infos = self._prepare_class_embeddings()

        # Load precomputed features
        with open(args.feat_file, "rb") as f:
            self.feats = pickle.load(f)
        print("Loaded 3D features")

    def _prepare_class_embeddings(self):
        """
        Generate text embeddings for class names using CLIP.
        """
        class_name_tokens = self.tokenizer(
            [f"a {class_name} in a scene" for class_name in self.class_name_list],
            padding=True,
            return_tensors="pt",
        )
        for name in class_name_tokens.data:
            class_name_tokens.data[name] = class_name_tokens.data[name].cuda()

        label_embeddings = self.clip.get_text_features(**class_name_tokens)
        return label_embeddings / label_embeddings.norm(p=2, dim=-1, keepdim=True)

    def load_pc(self, scan_id: str):
        """
        Load precomputed features for a given scan ID.
        """
        scan_data = self.feats[scan_id]
        return (
            scan_data["obj_ids"],  # Object IDs
            scan_data["inst_locs"],  # Object locations
            scan_data["center"],  # Scene center
            scan_data["obj_embeds"],  # Object embeddings
            scan_data["batch_labels"],  # Ground truth labels
        )

    def process_scan_ids(self, scan_ids: list):
        """
        Process a list of scan IDs and save results to JSON files.
        """
        for scan_id in tqdm(scan_ids, desc="Processing Scan IDs"):
            print(scan_id)
            if scan_id in ["scene0217_00"]:
                continue
            obj_ids, inst_locs, center, obj_embeds, gt_labels = self.load_pc(scan_id)

            # Compute class predictions
            try:
                class_logits_3d = torch.matmul(
                    self.label_lang_infos, obj_embeds.t().cuda()
                )
            except:
                import pdb

                pdb.set_trace()
            pred_classes = [
                self.class_name_list[idx] for idx in class_logits_3d.argmax(dim=0)
            ]

            self.feats[scan_id]["pred_class_list"] = pred_classes
            valid_results = self._generate_results(scan_id)
            self._save_results_to_json(valid_results, scan_id)

    def _generate_results(self, scan_id: str):
        """
        Generate results for a given scan ID by validating predictions.
        """
        pred_classes = self.feats[scan_id]["pred_class_list"]
        results = []

        for i, pred_class in enumerate(pred_classes):
            object_id = self.feats[scan_id]["obj_ids"][i]
            bbox_3d = self.feats[scan_id]["inst_locs"][i]

            results.append(
                {
                    "bbox_id": object_id,
                    "target": pred_class,
                    "bbox_3d": convert_to_serializable(bbox_3d),
                }
            )

        return results

    def _save_results_to_json(self, results: list, scan_id: str):
        """
        Save results to a JSON file.
        """
        output_file_path = os.path.join(self.output_dir, f"{scan_id}.json")
        if os.path.exists(output_file_path):
            print(f"Skipping existing file: {output_file_path}")
            return

        with open(output_file_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {output_file_path}")


def nr3d_gt(scan_ids, args):

    output_dir = os.path.join(args.output_dir, "gt")
    os.makedirs(output_dir, exist_ok=True)

    with open(args.feat_file, "rb") as f:
        scans = pickle.load(f)

    for scan_id in scan_ids:
        if scan_id in ["scene0217_00"]:
            continue
        scan_data = scans[scan_id]
        obj_ids = scan_data["obj_ids"]  # e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        inst_locs = scan_data[
            "inst_locs"
        ]  # e.g. [[x, y, z, w, h, l], [x, y, z, w, h, l], ...]
        batch_labels = scan_data[
            "batch_labels"
        ]  # e.g. ['wall', 'wall', 'wall', 'wall', 'wall', 'wall', 'wall', 'wall', 'wall', 'door

        objects_list = []

        for obj_id, loc, label in zip(obj_ids, inst_locs, batch_labels):
            obj_dict = {"bbox_id": obj_id, "target": label, "bbox_3d": list(loc)}
            objects_list.append(convert_to_serializable(obj_dict))

        # save as json file
        output_file_path = os.path.join(output_dir, f"{scan_id}.json")
        with open(output_file_path, "w") as output_file:
            json.dump(objects_list, output_file, indent=4)

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
        default="PCGrounder/data/nr3d/object_lookup_table/",
        help="Directory to save output results.",
    )
    parser.add_argument(
        "--feat_file",
        type=str,
        default="PCGrounder/data/nr3d/feats_3d.pkl",
        help="Path to the 3D features pickle file.",
    )

    args = parser.parse_args()
    scan_ids = read_file_to_list(args.val_file)

    # gt
    nr3d_gt(scan_ids, args)

    # pred
    classifier = ObjectClassifier(args)
    classifier.process_scan_ids(scan_ids)
