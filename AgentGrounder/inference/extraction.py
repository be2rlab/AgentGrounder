import torch
import point_cloud_utils as pcu
import numpy as np
import sys
import os

from inference.projection import render_point_cloud_with_pytorch3d_with_objects

from inference.utils import (
    stem_match,
    fuzzy_match,
)

from inference.embedding_database_class import EmbeddingDatabase
from utils.langchain_utils import create_prompt
from utils.config_loader import Prompt
from utils.visual import draw_and_save_sam3_results, extract_and_project

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import pybboxes as pbx
import json

def extract_anchor_targets_nlp(parsed_query, object_names, mask3d_bboxes):
    # Matching target and anchor
    try:
        target_name = parsed_query["Target"]
        anchor_name = parsed_query["Anchor"]
    except:
        target_name = ""
        anchor_name = ""
        
    print(f"Parsed target: {target_name}; anchor: {anchor_name}")

    matched_targets = fuzzy_match(target_name, object_names).union(
        stem_match(target_name, object_names)
    )
    matched_anchors = fuzzy_match(anchor_name, object_names).union(
        stem_match(anchor_name, object_names)
    )
    print(f"Matched target: {matched_targets}; anchor: {matched_anchors}")

    targets = [
        obj for obj in mask3d_bboxes.values() if obj["target"] in matched_targets
    ]
    anchors = [
        obj for obj in mask3d_bboxes.values() if obj["target"] in matched_anchors
    ]
    
    return anchors, targets


def extract_anchor_targets_embeds(embedding_db: EmbeddingDatabase, query: str, parsed_query: dict, object_names, mask3d_bboxes: dict):
    # Matching target and anchor
    try:
        target_name_list = parsed_query["Target"]
        anchor_name_list = parsed_query["Anchor"]
    except:
        target_name_list = [query]
        anchor_name_list = [query]

    # check if types are list, if not, convert to list
    if not isinstance(target_name_list, list):
        target_name_list = [target_name_list]
    if not isinstance(anchor_name_list, list):
        anchor_name_list = [anchor_name_list]
        
    print(f"Parsed target: {target_name_list}; anchor: {anchor_name_list}")

    matched_targets = []
    for target_name in target_name_list:
        matched_targets += embedding_db.get_top_k_similar_items(target_name, top_k=40, score_threshold=0.2)

    matched_anchors = []
    for anchor_name in anchor_name_list:
        matched_anchors += embedding_db.get_top_k_similar_items(anchor_name, top_k=40, score_threshold=0.2)

    target_label_list = [obj.get("target") for obj in matched_targets]
    anchor_label_list = [obj.get("target") for obj in matched_anchors]
    print(f"Matched target: {target_label_list}; anchor: {anchor_label_list}")


    extended_target_list = list()
    for obj in matched_targets:
        bbox_id = int(obj["bbox_id"])
        extended_target_list.append({
            "bbox_id": bbox_id,
            "target": obj["target"],
            "bbox_3d": mask3d_bboxes[bbox_id]["bbox_3d"],
            "description": obj["description"],
        })
    

    extended_anchor_list = list()
    for obj in matched_anchors:
        bbox_id = int(obj["bbox_id"])
        extended_anchor_list.append({
            "bbox_id": bbox_id,
            "target": obj["target"],
            "bbox_3d": mask3d_bboxes[bbox_id]["bbox_3d"],
            "description": obj["description"],
        })

    # Just for backup. But this should never happen
    if len(extended_target_list) == 0:
        extended_target_list = list(mask3d_bboxes.values())

    if len(extended_anchor_list) == 0:
        extended_anchor_list = extended_target_list

    return extended_anchor_list, extended_target_list


def fill_missing_targets(anchors, targets, mask3d_bboxes):
    if len(targets) == 0:
        targets = list(mask3d_bboxes.values())

    if len(anchors) == 0:
        anchors = targets.copy()
        
    return anchors, targets


def enclosing_bbox(bboxes):
    """
    bboxes: list/Nx6 array [[x,y,z, w,l,h], ...]
    Returns: [x1,y1,z1, x2,y2,z2] enclosing all (same format)
    """
    bboxes = np.array(bboxes)  # Shape: (N, 6)
    
    centers = bboxes[:, :3]     # Nx3
    sizes = bboxes[:, 3:] / 2   # Nx3 half-sizes
    
    # Min/max corners across all boxes
    min_corners = centers - sizes
    max_corners = centers + sizes
    
    global_mins = min_corners.min(axis=0)
    global_maxs = max_corners.max(axis=0)
    
    return global_mins, global_maxs


def crop_pointcloud_bbox(points, global_mins, global_maxs):
    """
    points: Nx3 array [X,Y,Z]
    bbox: [x,y,z, w,l,h] -> center + sizes
    Returns: filtered_points (Mx3), indices
    """
    x_min, y_min, z_min = global_mins
    x_max, y_max, z_max = global_maxs
    
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    
    return points[mask]


def sample_points(scan_pc, anchors, targets, mask3d_bboxes):
    points = scan_pc[:, :3]
    
    global_mins, global_maxs = enclosing_bbox([bbox['bbox_3d'] for bbox in anchors])
    
    cropped_points = crop_pointcloud_bbox(points, global_mins - 1, global_maxs + 1)
    
    num_samples = 25
    idx = pcu.downsample_point_cloud_poisson_disk(cropped_points, radius=0.0, target_num_samples=num_samples)

    sampled_points = cropped_points[idx]
    
    return sampled_points


def update_mask3d_bboxes(mask3d_bboxes, new_bboxes):
    mask3d_bboxes = mask3d_bboxes.copy()
    new_bboxes = new_bboxes.copy()
    
    max_id = max(key for key in mask3d_bboxes.keys())
    
    for i, bb in enumerate(new_bboxes):
        bb["bbox_id"] = max_id + i + 1
        
        mask3d_bboxes[int(bb['bbox_id'])] = bb
        
    return mask3d_bboxes, new_bboxes


def predict_missing_targets(query, parsed_query, anchors, targets, mask3d_bboxes, scan_pc, scene_center, save_dir, segmentor):
    if len(targets) == 0:
        if len(anchors) == 0:
            anchors = list(mask3d_bboxes.values())
        
        image_path, rasterizer = render_point_cloud_with_pytorch3d_with_objects(
            mask3d_bboxes.values(),
            [], # targets
            anchors, # anchors
            scene_center,
            scan_pc,
            save_dir=save_dir,
            image_size=1008,
            draw_id=False,
            draw_img=True,
            return_rasterizer=True
        )
        
        try:
            target_name_list = parsed_query["Target"]
        except:
            target_name_list = [query]

        bboxes_3d = segmentor.inference(image_path, target_name_list, scan_pc, rasterizer)

        new_mask3d_bboxes, pred_bboxes = update_mask3d_bboxes(mask3d_bboxes, bboxes_3d)
        
        mask3d_bboxes = new_mask3d_bboxes
        targets = pred_bboxes.copy()

    else:       
        if len(anchors) == 0:
            anchors = targets.copy()
            
    return anchors, targets, mask3d_bboxes