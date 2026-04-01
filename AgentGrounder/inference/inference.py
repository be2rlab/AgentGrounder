import sys
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import current_thread
from typing import Optional
from time import perf_counter
from tqdm import tqdm
import json
from pathlib import Path

# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.output_parsers import StrOutputParser

from inference.extraction import extract_anchor_targets_nlp, extract_anchor_targets_embeds, fill_missing_targets, predict_missing_targets
from inference.projection import render_point_cloud_with_pytorch3d_with_objects
from inference.utils import (
    parse_response,
    calc_iou,
    encode_img,
    read_file_to_list,
    save_to_file,
    stem_match,
    fuzzy_match,
    load_json,
    load_bboxes,
    generate_objects_info,
    load_scene_pcd,
)
from prepare_data.pcd_preparation import remove_ceiling
from inference.embedding_database_class import create_embedding_database, EmbeddingDatabase

from inference.segmentor_3d import Segmentor3D, create_segmentor_3d

from utils.config_loader import load_configuration
from utils.model_loader import create_llm
from utils.langchain_utils import create_prompt

import numpy as np

from inference.vg_agent.vg_agent import VGAgent
from inference.vg_agent.vg_agent_tools import FinalAnswer, RoomData

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=Path('/home/docker_user/PCGrounder/configs/scanrefer.yaml'), help="Path to config")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for query processing in each room")

    args = parser.parse_args()

    return args

def process_query(
    query,
    objects_info,
    model,
    prompt_config,
    vg_agent: VGAgent,
    use_image=False,
    image_path=None,
    model_name="qwen3-vl",
    log_file=None,
    scan_pc=None,
    center=None,
    render_save_dir=None,
) -> FinalAnswer:
    """Process query and return model's response."""
    assert objects_info is not None
    assert query is not None
    assert vg_agent is not None

    response: FinalAnswer = vg_agent.invoke(
        query,
        image_path,
        log_file,
        scan_pc=scan_pc,
        center=center,
        render_save_dir=render_save_dir,
    ) # type: ignore
    # print(f"Response from VGAgent: {response}")

    return response

def prepare_prediction(query, mask3d_bboxes, predicted_id, image_path, parsed_query, explanation):
    try:
        pred_bbox = mask3d_bboxes[predicted_id]
    except:
        pred_bbox = None

    return {
        "query": query,
        "predicted_id": predicted_id,
        "pred_bbox": pred_bbox["bbox_3d"] if pred_bbox else None,
        "image_path": image_path,
        "parsed_query": parsed_query,
        "explanation": explanation
    }


def make_prediction(query, parsed_query, room, pcd_dir, mask3d_bboxes,
                        model, prompt_config, use_image, model_name, log_file, save_dir, save_dir_aug, embedding_db, segmentor, vg_agent, scan_pc=None, center=None):
    
    print('-'*20)
    print(f"Making prediction for query: {query}")

    object_names = [obj["target"] for obj in mask3d_bboxes.values()]
    
    # Generate query-aligned image
    if scan_pc is None or center is None:
        scan_pc, center = load_scene_pcd(room, pcd_dir)
        scan_pc = remove_ceiling(scan_pc)

    # anchors, targets = extract_anchor_targets_nlp(parsed_query, object_names, mask3d_bboxes)
    # anchors, targets = extract_anchor_targets_embeds(embedding_db, query, parsed_query, object_names, mask3d_bboxes)
    
    # anchors, targets = fill_missing_targets(anchors, targets, mask3d_bboxes)
    # anchors, targets, mask3d_bboxes = predict_missing_targets(query, parsed_query, anchors, targets, mask3d_bboxes, scan_pc, center, save_dir_aug, segmentor)

    # print("Anchors:", [f"{entry['bbox_id']}: {entry['target']}" for entry in anchors], sep=' ')
    # print("Targets", [f"{entry['bbox_id']}: {entry['target']}" for entry in targets], sep=' ')

    objects_info = generate_objects_info(mask3d_bboxes.values())

    # image_path = render_point_cloud_with_pytorch3d_with_objects(
    #     mask3d_bboxes.values(),
    #     targets,
    #     anchors,
    #     center,
    #     scan_pc,
    #     save_dir=save_dir,
    #     image_size=680,
    #     draw_id=True,
    #     draw_img=True,
    # )
    image_path = None

    # Process query 
    response = process_query(
        query,
        objects_info,
        model,
        prompt_config,
        vg_agent,
        use_image,
        image_path, # type: ignore
        model_name,
        log_file,
        scan_pc=scan_pc,
        center=center,
        render_save_dir=save_dir,
    )

    print(f'RESPONSE: {response}')
    
    predicted_id, explanation = parse_response(response)

    result = prepare_prediction(query, mask3d_bboxes, predicted_id, image_path, parsed_query, explanation)
    
    return result


def load_scanrefer_data(room, language_annotation_dir, gt_bbox_dir, pred_bbox_dir, open_vocab_captions_dir):
    language_annotation_file = os.path.join(language_annotation_dir, f"{room}.json")
    
    data = load_json(language_annotation_file)
    queries = [it for it in data if it["scan_id"] == room]
    queries = sorted(queries, key=lambda x: int(x["target_id"]))
    
    gt_bboxes = load_bboxes(room, gt_bbox_dir, "gt")
    
    mask3d_bboxes = load_bboxes(room, pred_bbox_dir, "pred")
    
    return queries, gt_bboxes, mask3d_bboxes


def process_room(
    dataset,
    room,
    pcd_dir,
    split,
    output_dir,
    language_annotation_dir,
    gt_bbox_dir,
    pred_bbox_dir,
    models,
    model_base_urls,
    prompt_config,
    embedding_db: EmbeddingDatabase,
    segmentor,
    open_vocab_captions_dir,
    use_image=False,
    model_name=None,
    verbose=True,
    num_workers=1,
):
    """Process a single room with queries and predictions."""
    # Load annotations and bounding boxes
    
    output_file = os.path.join(output_dir, "pred", f"{room}.json")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    queries, gt_bboxes, mask3d_bboxes = load_scanrefer_data(room, language_annotation_dir, gt_bbox_dir, pred_bbox_dir, open_vocab_captions_dir)

    system_message = prompt_config.system_prompt

    scan_pc, center = load_scene_pcd(room, pcd_dir)
    scan_pc = remove_ceiling(scan_pc)

    def run_single_query(i, d):
        query = d["caption"]
        gt_id = int(d["target_id"])
        parsed_query = d['parsed_query']

        model_idx = i % len(models)
        selected_model = models[model_idx]
        selected_base_url = model_base_urls[model_idx]
        print(f"[MODEL_ROUTE] room={room} idx={i} model_idx={model_idx} base_url={selected_base_url}")
        
        save_dir = os.path.join(output_dir, "logs", f"projection_img/{room}/{i}")
        save_dir_aug = os.path.join(output_dir, "logs", f"projection_img_aug/{room}/{i}")

        log_file = os.path.join(output_dir, "logs", f"projection_img/{room}/{i}/log.md")
        print(f"Will log objects_info to {log_file}")

        vg_agent = VGAgent(model=selected_model, system_prompt=system_message, objects=mask3d_bboxes, vectorstore=embedding_db.vectorstore)

        # segmentor.set_save_dir(save_dir_aug)
        
        pred_result = make_prediction(query, parsed_query, room, pcd_dir, mask3d_bboxes, 
                    selected_model, prompt_config, use_image, model_name, log_file, save_dir, save_dir_aug, embedding_db, segmentor, vg_agent, scan_pc=scan_pc, center=center)
        
        print(f"GT id is {gt_id}; Pred id is {pred_result['predicted_id']}") # , {explanation}, {image_path}")

        pred_result["gt_id"] = gt_id
        pred_result["gt_bbox"] = gt_bboxes[gt_id]['bbox_3d']
        # pred_result["unique"] = d["unique"]

        # for debugging
        try:
            pred_id = pred_result["predicted_id"]
            gt_bbox = gt_bboxes[gt_id]['bbox_3d']
            pred_bbox = mask3d_bboxes[pred_id]['bbox_3d']
            pred_result["iou"] = calc_iou(gt_bbox, pred_bbox)   
            print(f"IOU: {pred_result['iou']}")
        except Exception as e:
            print(f"Error calculating IoU: {e}")
            pred_result["iou"] = 0.0


        # add dataset-specific metadata if available
        if dataset == "scanrefer":
            pred_result["unique"] = d.get("unique")
        elif dataset == "nr3d":
            pred_result["easy"] = d.get("easy")
            pred_result["view_dep"] = d.get("view_dep")

        return i, pred_result

    results = []

    effective_workers = max(1, min(int(num_workers), len(queries)))
    mode = "parallel" if effective_workers > 1 else "sequential"
    print(
        f"[QUERY_EXECUTION] room={room} mode={mode} configured_workers={num_workers} "
        f"effective_workers={effective_workers} total_queries={len(queries)}"
    )

    room_start = perf_counter()

    def _run_single_query_with_timing(i, d):
        thread_name = current_thread().name
        query_start = perf_counter()
        print(f"[QUERY_START] room={room} idx={i} thread={thread_name}")
        idx, pred = run_single_query(i, d)
        elapsed = perf_counter() - query_start
        print(f"[QUERY_END] room={room} idx={i} thread={thread_name} elapsed_sec={elapsed:.2f}")
        return idx, pred

    if effective_workers == 1:
        for i, d in enumerate(tqdm(queries)):
            _, pred_result = _run_single_query_with_timing(i, d)
            results.append(pred_result)
    else:
        indexed_results: list[Optional[dict]] = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_run_single_query_with_timing, i, d): i
                for i, d in enumerate(queries)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Process queries in {room}"):
                idx, pred_result = future.result()
                indexed_results[idx] = pred_result

        results = [item for item in indexed_results if item is not None]

    room_elapsed = perf_counter() - room_start
    print(f"[ROOM_DONE] room={room} mode={mode} elapsed_sec={room_elapsed:.2f}")

    save_to_file(output_file, json.dumps(results, indent=4))


def main(args):
    config = load_configuration(yaml_path=args.config_path)

    experiment = config.experiment
    prompt_config = config.prompts.inference_prompt
    data = experiment.data

    scan_ids = list(os.listdir(data.language_annotation_dir))
    scan_ids = sorted([i.split(".")[0] for i in scan_ids])
    print(f"Found {len(scan_ids)} scans in {data.language_annotation_dir}")

    # segmentor = create_segmentor_3d(config=config.segmentor3d)
    segmentor = None

    raw_base_urls = config.model.base_url
    model_base_urls = [url.strip() for url in raw_base_urls.split(",") if url.strip()]
    if not model_base_urls:
        raise ValueError("No valid model base URL found. Please set model.base_url in config or pass --model_base_urls.")

    models = [create_llm(config=config.model, base_url=base_url) for base_url in model_base_urls]
    print(f"[MODEL_POOL] size={len(models)} base_urls={model_base_urls}")

    for room in tqdm(scan_ids, desc="Process rooms"):
        embedding_db = create_embedding_database(
            config.embedding_model,
            vectorstore_dir=data.vectorstore_dir,
            collection_name=room
        )
            
        process_room(
            dataset=experiment.dataset,
            split=experiment.split,
            room=room,
            output_dir=experiment.output_dir,
            language_annotation_dir=data.language_annotation_dir,
            pcd_dir=data.pcd_dir,
            gt_bbox_dir=data.gt_bbox_dir,
            pred_bbox_dir=data.pred_bbox_dir,
            models=models,
            model_base_urls=model_base_urls,
            prompt_config=prompt_config,
            model_name=config.model.name,
            use_image=config.rendering.use_image,
            embedding_db=embedding_db,
            segmentor=segmentor,
            open_vocab_captions_dir=data.open_vocab_captions_dir,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    args = get_parser_args()
    
    main(args=args)
