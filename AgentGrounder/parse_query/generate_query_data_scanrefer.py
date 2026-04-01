import os
import json
import argparse
import jsonlines
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI
import torch
import numpy as np
import pandas as pd


def load_ref_data(anno_file, scan_id_file):
    with open(anno_file, "r") as f:
        data = json.load(f)

    print(len(data))

    split_scan_ids = set(x.strip() for x in open(scan_id_file, "r"))
    ref_data = []
    for item in data:
        if item["scene_id"] in split_scan_ids:
            ref_data.append(item)

    print(len(ref_data))
    return ref_data


def convert_numpy_types(data):
    """
    Recursively convert numpy types to native Python types.

    Args:
        data (dict or list): The data to convert.

    Returns:
        dict or list: The converted data.
    """
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(element) for element in data]
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    else:
        return data


def load_pc(scan_id, keep_background=False, scan_dir=""):
    """
    Returns:
        tuple: A tuple containing the following elements:
            - batch_labels (list): List of object labels.
            - obj_ids (list): List of object IDs.
            - inst_locs (list): List of instance locations and sizes.
            - center (numpy.ndarray): Center of the point cloud.
            - batch_pcds (torch.Tensor): Batch of point clouds with sampled points.
    """
    pcds, colors, _, instance_labels = torch.load(
        os.path.join(scan_dir, "pcd_with_global_alignment", "%s.pth" % scan_id)
    )
    obj_labels = json.load(
        open(os.path.join(scan_dir, "instance_id_to_name", "%s.json" % scan_id))
    )

    origin_pcds = []
    batch_pcds = []
    batch_labels = []
    inst_locs = []
    obj_ids = []
    res_list = []
    for i, obj_label in enumerate(obj_labels):
        if (not keep_background) and obj_label in ["wall", "floor", "ceiling"]:
            continue
        mask = instance_labels == i
        assert np.sum(mask) > 0, "scan: %s, obj %d" % (scan_id, i)
        obj_pcd = pcds[mask]
        obj_color = colors[mask]
        origin_pcds.append(np.concatenate([obj_pcd, obj_color], 1))

        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))

        height_array = obj_pcd[:, 2:3] - obj_pcd[:, 2:3].min()

        # normalize
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

        batch_pcds.append(np.concatenate([obj_pcd, obj_height, obj_color], 1))
        batch_labels.append(obj_label)
        obj_ids.append(i)

    batch_pcds = torch.from_numpy(np.stack(batch_pcds, 0))
    center = (pcds.max(0) + pcds.min(0)) / 2

    return batch_labels, obj_ids, inst_locs, center, batch_pcds


def process_reference_item(
    ref,
    program_prompt,
    client,
    args,
    batch_labels,
    obj_ids,
    inst_locs,
    center,
    batch_pcds,
    labels_pd,
    batch_class_ids,
):
    """
    Process each reference item and generate new data for it.

    :param ref: A single reference item
    :param program_prompt: The prompt to be used for the OpenAI model
    :param client: The OpenAI client
    :param args: Parsed command-line arguments
    :return: Processed data for the reference item
    """
    caption = ref["description"]
    print("-" * 20)
    print(caption)
    print(ref["scene_id"])

    scan_id = ref["scene_id"]

    index = obj_ids.index(int(ref["object_id"]))
    target_class_id = batch_class_ids[index]
    unique = (np.array(batch_class_ids) == target_class_id).sum() == 1

    input_prompt = f"Query: {caption}"

    messages = [
        {"role": "system", "content": program_prompt},
        {"role": "user", "content": [{"type": "text", "text": input_prompt}]},
    ]
    chat_response = client.chat.completions.create(
        model=args.model_name, messages=messages
    )
    answer = chat_response.choices[0].message.content

    try:
        answer = answer.replace("'", '"')
        answer = json.loads(answer)
        print(answer)
        target = answer["Target"]
        anchor = answer["Anchor"]
        print(f"Target: {target}, Anchor: {anchor}")
    except:
        print(answer)
        target = ""
        anchor = ""
        print("!!! Warning, Error in answer")

    return {
        "scan_id": ref["scene_id"],
        "target_id": ref["object_id"],
        "target_name": ref["object_name"],
        "caption": ref["description"],
        "parsed_query": answer,
        "unique": unique,
    }


def save_processed_data(new_data, save_dir, scan_id):
    """
    Save the processed data for a scan to a file.

    :param new_data: The new data to be saved
    :param save_dir: Directory where the file should be saved
    :param scan_id: The scan ID for which the data is being saved
    """
    os.makedirs(save_dir, exist_ok=True)

    new_data = convert_numpy_types(
        new_data
    )  # Convert numpy types to native Python types

    with open(f"{save_dir}/{scan_id}.json", "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved scan {scan_id} data to {save_dir}/{scan_id}.json")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process rooms for object detection.")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="your_openai_api_key",
        help="OpenAI API key.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/root/Qwen2-VL/PCGrounder/data/scanrefer/query",
        help="Directory to save data.",
    )
    parser.add_argument(
        "--openai_api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI API base URL.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen2-VL-72B-Instruct",
        help="Model name for OpenAI API.",
    )
    parser.add_argument(
        "--anno_file",
        type=str,
        default="/root/Qwen2-VL/PCGrounder/data/scanrefer/ScanRefer_filtered_val.json",
        help="Path to the annotation file.",
    )
    parser.add_argument(
        "--scan_id_file",
        type=str,
        default="/data/scannetv2_val.txt",
        help="Path to the scan ID file.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="PCGrounder/prompts/parsing_query.txt",
        help="Path to the prompt file.",
    )
    parser.add_argument(
        "--label_map_file",
        type=str,
        default="/remote-home/share/vg_datasets/referit3d/official_data/annotations/meta_data/scannetv2-labels.combined.tsv",
    )
    parser.add_argument(
        "--scan_data",
        type=str,
        default="/remote-home/share/vg_datasets/referit3d/scan_data",
    )

    args = parser.parse_args()

    # Load reference data
    ref_data = load_ref_data(args.anno_file, args.scan_id_file)

    # Load prompt
    with open(args.prompt_file, "r") as f:
        program_prompt = f.read()

    # Group data by scan_id
    grouped_data = defaultdict(list)
    for ref in ref_data:
        grouped_data[ref["scene_id"]].append(ref)

    # Sort scan_ids
    sorted_scan_ids = sorted(grouped_data.keys())

    # Process each scan_id and generate new data
    for scan_id in sorted_scan_ids:
        # if os.path.exists(f"{args.save_dir}/{scan_id}.json"):
        #     print(f"Skipping {scan_id}. Already exists.")
        #     continue

        entries = grouped_data[scan_id]
        new_data = []

        batch_labels, obj_ids, inst_locs, center, batch_pcds = load_pc(
            scan_id, scan_dir=args.scan_data
        )

        # Load label map
        labels_pd = pd.read_csv(args.label_map_file, sep="\t", header=0)

        batch_class_ids = []

        for i, obj_label in enumerate(batch_labels):
            label_ids = labels_pd[labels_pd["raw_category"] == obj_label]["nyu40id"]
            label_id = int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
            batch_class_ids.append(label_id)

        # Process each reference item
        for ref in tqdm(entries[:5]):
            new_data.append(
                process_reference_item(
                    ref,
                    program_prompt,
                    OpenAI(api_key=args.openai_api_key, base_url=args.openai_api_base),
                    args,
                    batch_labels,
                    obj_ids,
                    inst_locs,
                    center,
                    batch_pcds,
                    labels_pd,
                    batch_class_ids,
                )
            )

        # Save processed data
        save_processed_data(new_data, args.save_dir, scan_id)
