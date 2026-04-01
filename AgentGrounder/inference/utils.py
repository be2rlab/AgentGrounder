import re
from difflib import get_close_matches
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import base64
import json
import os
import numpy as np
import open3d as o3d
import torch

from inference.vg_agent.vg_agent_tools import FinalAnswer

nltk.download("punkt_tab")
stemmer = PorterStemmer()


def fuzzy_match(names, object_names, threshold=0.8):
    matched_names = set()
    for name in names:
        matches = get_close_matches(name, object_names, n=1, cutoff=threshold)
        if matches:
            matched_names.add(matches[0])
    return matched_names


def stem_match(names, object_names):
    matched_names = set()
    if isinstance(names, str):
        names = [names]
    for name in names:
        name_stems = [stemmer.stem(word) for word in word_tokenize(name)]
        for obj_name in object_names:
            obj_name_stems = [stemmer.stem(word) for word in word_tokenize(obj_name)]
            if set(name_stems) & set(obj_name_stems):
                matched_names.add(obj_name)
    return matched_names


def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def save_to_file(file_path, content):
    """Save content to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file:
        file.write(content)


def encode_img(image_path):
    """Encode image to Base64 format."""
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read())
    return f"data:image;base64,{encoded_image.decode('utf-8')}"


def read_file_to_list(file_path):
    """
    Read a file and convert its contents to a list.
    """
    with open(file_path, "r") as file:
        lines = file.read().splitlines()
    return sorted(lines)


def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """
    box_a = np.array(box_a)
    box_b = np.array(box_b)

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union


def parse_response(response):
    """
    Parse model response to extract predicted ID and explanation
    """
    predicted_id = None
    explanation = None

    if isinstance(response, FinalAnswer):
        predicted_id = response.object_id
        explanation = response.explanation

    elif isinstance(response, str):
        for line in response.split("\n"):
            if line.startswith("Predicted ID:"):
                predicted_id = line.split(":")[1].strip()
            elif line.startswith("Explanation:"):
                explanation = line.split(":", 1)[1].strip()
        try:
            predicted_id = int(predicted_id)
        except:
            predicted_id = None
            
    return predicted_id, explanation


# Data Loading and Processing
def load_bboxes(room, bbox_dir, file_type="pred"):
    """Load bounding boxes (GT or predicted)."""
    bbox_file = os.path.join(bbox_dir, f"{room}.json")
    bboxes = load_json(bbox_file)
    return {int(bbox["bbox_id"]): bbox for bbox in bboxes}


def generate_objects_info(pred_bbox_list, anchors_targets=None):
    """Generate a formatted string of object information."""
    if anchors_targets:
        pred_bbox_list = list(pred_bbox_list).copy()

        object_ids = set(sorted([entry['bbox_id'] for entry in anchors_targets]))
        pred_bbox_list = [bbox for bbox in pred_bbox_list if bbox['bbox_id'] in object_ids]

    return "\n".join(
        [
            f"Object ID: {bbox['bbox_id']}, Type: {bbox['target']}, Dimensions: Width {bbox['bbox_3d'][3]:.2f}, Length {bbox['bbox_3d'][4]:.2f}, Height {bbox['bbox_3d'][5]:.2f}, Center Coordinates: X {bbox['bbox_3d'][0]:.2f}, Y {bbox['bbox_3d'][1]:.2f}, Z {bbox['bbox_3d'][2]:.2f}"
            for bbox in pred_bbox_list
            if bbox["target"] not in ["wall", "floor", "ceiling"]
        ]
    )

# Rendering
def load_scene_pcd(room, scan_dir='/remote-home/share/vg_datasets/referit3d/scan_data/pcd_with_global_alignment/'):
    """Load and process point cloud data."""
    pcds, colors, _, instance_labels = torch.load(
        os.path.join(scan_dir, '%s.pth' % room),
        weights_only=False,    
    )
    
    scan_pc = np.concatenate((pcds, colors/255), axis=1).astype("float32")
    center = np.mean(scan_pc[:, :3], axis=0)
    return scan_pc, center
