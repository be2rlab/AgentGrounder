<p align="center">  
  <h1 align="center">Agent-Grounder: Zero-Shot 3D Visual Pointcloud Grounding</h1>
</p>

# Table of Content
- [0. Environment Setup](#0-environment-setup)
- [1. Download Model Weights](#1-download-model-weights)
- [2. Download Datasets](#2-download-datasets)
  - [2.1. ScanRefer](#21-scanrefer)
  - [2.2. Nr3D](#22-nr3d)
  - [2.3. Vil3dref Preprocessed Data](#23-vil3dref-preprocessed-data)
- [3. Data Processing](#3-data-processing)
- [4. Inference](#4-inference)
  - [4.1. Deploying VLM Service](#41-deploying-vlm-service)
  - [4.2. Generating Anchors & Targets](#42-generating-anchors--targets)
  - [4.3. Predictions](#43-predictions)
  - [4.4. Evaluations](#44-evaluations)
- [5. License](#5-license)


# 0. Environment Setup

We recommend using our docker image for environment setup
```bash
make build
make up
make into
make stop
```

Install our package inside docker:
```bash
pip install -e .
```

# 1. Download Model Weights

```bash
cd AgentGrounder/weights
git clone https://huggingface.co/IDEA-Research/Rex-Omni         # Rex-Omni
git clone https://huggingface.co/IDEA-Research/Rex-Omni-AWQ     # Quantized Rex-Omni
git clone https://huggingface.co/facebook/sam3                  # SAM3
```


# 2. Download Datasets

## 2.1. ScanRefer 

Download ScanRefer dataset from [official repo](https://github.com/daveredrum/ScanRefer), and place it in the following directory:
```bash
data/ScanRefer/ScanRefer_filtered_val.json
```

## 2.2. Nr3D 

Download the Nr3D dataset from the [official repo](https://github.com/referit3d/referit3d), and place it in the following directory:

```
data/Nr3D/Nr3D.json
```

## 2.3. Vil3dref Preprocessed Data

Download the preprocessed Vil3dref data from [vil3dref](https://github.com/cshizhe/vil3dref).


The expected structure should look like this:
```
referit3d/
.
├── annotations
|   ├── meta_data
|   │   ├── cat2glove42b.json
|   │   ├── scannetv2-labels.combined.tsv
|   │   └── scannetv2_raw_categories.json
│   └── ...
├── ...
└── scan_data
    ├── ...
    ├── instance_id_to_name
    └── pcd_with_global_alignment
```

# 3. Data Processing

Download [mask3d pred](https://github.com/CurryYuan/ZSVG3D) first.

- ScanRefer 
```bash
python -m prepare_data.object_lookup_table_scanrefer
```

- Nr3D

```bash
python -m prepare_data.process_feat_3d

python -m prepare_data.object_lookup_table_nr3d
```

# 4. Inference

## 4.1. Deploying VLM Service

We use `ollama` to deploy the VLM. Please install `ollama` server on your server.

## 4.2. Generating Anchors & Targets

- ScanRefer
```bash
python -m parse_query.generate_query_data_scanrefer
```

- Nr3D
```bash
python -m parse_query.generate_query_data_nr3d
```

## 4.3. Predictions

```bash
python -m inference.inference --config_path <nr3d_or_scanrefer_config_path>
```

## 4.4. Evaluations

- ScanRefer 
```bash
python -m eval.eval_nr3d
```

- Nr3D
```bash
python -m eval.eval_scanrefer
```


# 5. License
This work is released under the CC BY 4.0 license.