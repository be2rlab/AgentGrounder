import os
import sys
import json
from typing import Union, List, Optional
from pathlib import Path

import torch
import pybboxes as pbx
from PIL import Image
import numpy as np

from rex_omni import RexOmniWrapper, RexOmniVisualize
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from utils.visual import draw_and_save_sam3_results, extract_and_project
from utils.config_loader import Segmentor3DConfig

from inference.projection import get_3d_bbox_from_masks

class Segmentor3D:
    def __init__(
            self, 
            rex_folder_path: Path, sam3_bpe_path: Path, sam3_weights_path: Path,
            rex_device: torch.device = "cuda", sam3_device: torch.device = "cuda", save_dir: Optional[Path] = None
        ) -> None:
        self.rex = RexOmniWrapper(
            model_path=rex_folder_path,
            backend="transformers",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.8,
            top_k=5,
            repetition_penalty=1.05,
            device_map=rex_device
        )

        self.sam3 = build_sam3_image_model(
            bpe_path=sam3_bpe_path,
            device=sam3_device,
            eval_mode=True,
            checkpoint_path=sam3_weights_path,
            load_from_HF=False
        )
        
        self.sam3_processor = Sam3Processor(self.sam3, confidence_threshold=0.5, resolution=1008, device=sam3_device)

        self.save_dir = save_dir

    def inference_detector(self, image: Union[Image.Image, Path, str], targets: Union[str, List[str]]) -> List[dict]:
        if isinstance(image, (Path, str)):
            image = Image.open(image).convert("RGB")
    
        if not isinstance(targets, list):
            targets = [targets]
        
        detection = self.rex.inference(images=image, task="detection", categories=targets)[0]

        predictions = []
        for key, bboxes_list in detection["extracted_predictions"].items():
            for bb in bboxes_list:
                predictions.append(
                    {
                        "label": key,
                        "bbox": pbx.convert_bbox(
                            bb['coords'], 
                            from_type="voc", 
                            to_type="yolo", 
                            image_size=detection['image_size']
                        )
                    }
                )

        if self.save_dir is not None:
            vis = RexOmniVisualize(
                image=image,
                predictions=detection["extracted_predictions"],
                font_size=20,
                draw_width=5,
                show_labels=True,
            )

            vis.save(f"{self.save_dir}/detected.png")

        return predictions

    def inference_segmentor(self, image: Union[Image.Image, Path, str], bboxes: List[dict]) -> List[dict]:
        if isinstance(image, (Path, str)):
            image = Image.open(image).convert("RGB")

        inference_state = self.sam3_processor.set_image(image)

        predictions = []
        for i, bbox in enumerate(bboxes):
            self.sam3_processor.reset_all_prompts(inference_state)
            # output = self.sam3_processor.set_text_prompt(state=inference_state, prompt=bbox['label'])
            output = self.sam3_processor.add_geometric_prompt(state=inference_state, box=bbox['bbox'], label=True)

            pred = {
                "label": '[object]', # bbox['label'], '[probably target object]'
                "masks": output["masks"].cpu(),
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu()
            }

            predictions.append(pred)
            
            if self.save_dir is not None:
                draw_and_save_sam3_results(
                    image, pred['masks'], pred['boxes'], pred['scores'], pred['label'], 
                    f"{self.save_dir}/masked_{i}.png"
                )

        return predictions

    def inference(self, image: Union[Image.Image, Path, str], targets: Union[str, List[str]], scan_pc, rasterizer) -> List[dict]:
        if isinstance(image, (Path, str)):
            image = Image.open(image).convert("RGB")
    
        if not isinstance(targets, list):
            targets = [targets]

        bboxes_pred = self.inference_detector(image, targets)
        masks_pred = self.inference_segmentor(image, bboxes_pred)

        bboxes_3d = []
        for i, entry in enumerate(masks_pred):
            if len(entry['masks']) > 0:
                # best_mask = entry['masks'][torch.argmax(entry['scores'])]

                projected_bboxes = get_3d_bbox_from_masks(scan_pc, rasterizer, entry['masks'])
                
                if not projected_bboxes:
                    continue

                for bbox_3d in projected_bboxes:
                    bbox_3d = bbox_3d.tolist()

                    bboxes_3d.append({
                        "target": entry['label'],
                        "bbox_3d": bbox_3d
                    })
                
                # if self.save_dir is not None:
                #     # extract_and_project(scan_pc, bbox_3d, f"{self.save_dir}/object_3d_{i}.png")
                #     pil_img = Image.fromarray((masked_img * 255).astype(np.uint8))
                #     pil_img.save(f"{self.save_dir}/masked_img_{i}.png")
                #     # print(pil_img)

        if self.save_dir is not None:
            with open(f"{self.save_dir}/new_scene.json", "w", encoding="utf-8") as f:
                json.dump(bboxes_3d, f, indent=4, ensure_ascii=False)

        return bboxes_3d
    
    def set_save_dir(self, save_dir: Optional[Path] = None) -> None:
        self.save_dir = save_dir


def create_segmentor_3d(config: Segmentor3DConfig, save_dir: Optional[Path] = None) -> Segmentor3D:
    return Segmentor3D(
        rex_folder_path=config.rex_folder_path,
        sam3_bpe_path=config.sam3_bpe_path,
        sam3_weights_path=config.sam3_weights_path,
        save_dir=save_dir
    )