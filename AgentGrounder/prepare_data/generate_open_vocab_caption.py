from dataclasses import dataclass
import os
import json
import yaml
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import argparse
import sys

from inference.utils import load_bboxes
from utils.config_loader import load_configuration, Prompt
from utils.model_loader import create_llm, OllamaLLM
from utils.langchain_utils import create_prompt

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=Path('PCGrounder/configs/scanrefer.yaml'), help="Path to config")

    args = parser.parse_args()

    return args


def generate_captions_for_room(model: OllamaLLM, prompt_config: Prompt, room: str, 
                               predicted_bboxes_dir_path: Path, projections_dir_path: Path, output_dir_path: Path):
    pred_bboxes = load_bboxes(room, predicted_bboxes_dir_path)
    # create folder if not exist
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        
    output_file = os.path.join(output_dir_path, f"{room}.json")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping")
        return
    
    results = []
    for id in list(pred_bboxes.keys()):
        image_path = os.path.join(projections_dir_path, room, f"object_{id}.png")
        # break if image does not exist
        try:
            hint_label = pred_bboxes[id].get("target", "unknown")
            _,_,_, width, length, height = pred_bboxes[id]["bbox_3d"]
            
            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist. Using backup...")
                raise Exception
            prompt = create_prompt(prompt_config, use_image=True, image_path=image_path, input_variables=['width', 'length', 'height', 'hint_label'])
        
            chain = prompt | model
            answer = chain.invoke({"width": width, "length": length, "height": height, "hint_label": hint_label})
        except Exception:
            results.append({
                "bbox_id": id,
                "target": hint_label,
                "description": hint_label,
            })
            continue
        
        results.append({
            "bbox_id": id,
            "target": hint_label,
            "description": answer,
        })
        
        print(f"Room {room}: Object {id}: {answer}")

    # save results to json
    print(f"Saving data for room {room}")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    args = get_parser_args()
    config = load_configuration(yaml_path=args.config_path)
    
    model = create_llm(config.model)

    dir_paths = config.experiment.data
    
    predicted_bboxes_dir_path = dir_paths.pred_bbox_dir
    projections_dir_path = dir_paths.multi_view_projection_dir
    output_dir_path = dir_paths.open_vocab_captions_dir

    query_dir_path = dir_paths.language_annotation_dir # needed only for list of query
    scan_ids = list(os.listdir(query_dir_path))
    scan_ids = sorted([i.split(".")[0] for i in scan_ids])
    
    prompt_config = config.prompts.ov_caption_prompt
    
    for room in scan_ids:
        generate_captions_for_room(model, prompt_config, room, predicted_bboxes_dir_path, projections_dir_path, output_dir_path)

    