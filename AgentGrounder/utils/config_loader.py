import yaml
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal

from utils.yaml_loader import Loader

class DataPaths(BaseModel):
    """All dataset-related directories."""

    language_annotation_dir: Path = Field(
        default=Path("PCGrounder/data/scanrefer/query/"),
        description="Parsed language annotation (with anchor and target) file path"
    )

    gt_bbox_dir: Path = Field(
        default=Path("PCGrounder/data/scanrefer/object_lookup_table/gt"),
        description="Ground truth bounding box directory"
    )

    pred_bbox_dir: Path = Field(
        default=Path("PCGrounder/data/scanrefer/object_lookup_table/pred"),
        description="Predicted bounding box directory"
    )

    pcd_dir: Path = Field(
        default=Path("/remote-home/share/vg_datasets/referit3d/scan_data/pcd_with_global_alignment/"),
        description="Point cloud directory with global alignment"
    )

    ply_dir: Path = Field(
        default=Path("/data/scannet/scans"),
        description="Directory containing scene PLY mesh files (e.g. scene0000_00_vh_clean_2.ply)"
    )

    val_file: Path = Field(
        default=Path("/root/Qwen2-VL/PCGrounder/data/scannet/scannetv2_val.txt"),
        description="Validation split file path"
    )

    open_vocab_captions_dir: Path = Field(
        default=Path("/preprocessed_data/scanrefer/open_vocab_captions"),
        description="Open vocab captions dir path"
    )

    vectorstore_dir: Path = Field(
        default=Path("/preprocessed_data/scanrefer/chromadb_data/mask3d_labels"),
        description="Chroma vectorstore persist directory"
    )

    multi_view_projection_dir: Path = Field(
        default=Path("/preprocessed_data/scanrefer/multi_view_imgs_data/multi-view-projection-imgs"),
        description="Multi-view rendered img dir path"
    )

class ExperimentConfig(BaseModel):
    """All file system paths."""

    dataset: Literal["scanrefer", "referit3d", "nr3d"] = Field(
        default="scanrefer",
        description="Dataset name"
    )

    split: Literal["train", "val", "test"] = Field(
        default="test",
        description="Dataset split"
    )
    
    device: str = Field(
        default="cuda:0",
        description="Device to use for computation"
    )

    output_dir: Path = Field(
        default=Path("/outputs/test"),
        description="Directory to store evaluation outputs"
    )

    data: DataPaths

class Rendering(BaseModel):
    """Image rendering settings."""

    use_image: bool = Field(
        default=True,
        description="Enable image rendering for visual grounding"
    )

class ModelConfig(BaseModel):
    """LLM/VLM model configuration."""

    api_key: str = Field(
        default="your_openai_api_key",
        description="OpenAI-compatible API Key"
    )

    base_url: str = Field(
        default="http://localhost:11434",
        description="OpenAI-compatible API Base URL"
    )

    name: str = Field(
        default="qwen3-vl",
        description="Model name (qwen3-vl, llama3.1, gpt-4o-mini, etc.)"
    )

    seed: int = Field(
        default=69,
        description="Seed for consistent output of model"
    )

    temperature: float = Field(
        default=0.0,
        description="Temperature of model"
    )

    num_ctx: int = Field(
        default=16384,
        description="Context window size for the model"
    )

class Prompt(BaseModel):
    """Prompt config"""

    system_prompt: str = Field(
        default="",
        description="System prompt"
    )

    user_prompt: str = Field(
        description="User prompt"
    )

class PromptsConfig(BaseModel):
    """All prompts for chain or graph"""

    inference_prompt: Prompt
    ov_caption_prompt: Prompt

class Segmentor3DConfig(BaseModel):
    '''Segmentor3D configuration.'''

    rex_folder_path: Path = Field(
        default=Path("/home/docker_user/PCGrounder/weights/Rex-Omni"),
        description="Directory path for Rex-Omni weights"
    )

    sam3_bpe_path: Path = Field(
        default=Path("/home/docker_user/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        description="BPE path for SAM3"
    )

    sam3_weights_path: Path = Field(
        default=Path("/home/docker_user/PCGrounder/weights/sam3/sam3.pt"),
        description="Weights path for SAM3"
    )

class Configuration(BaseModel):
    """Complete evaluation configuration."""

    model: ModelConfig
    embedding_model: ModelConfig
    rendering: Rendering
    experiment: ExperimentConfig
    prompts: PromptsConfig
    segmentor3d: Segmentor3DConfig


def load_configuration(yaml_path, **entries):
    with open(yaml_path) as f:
        data = yaml.load(f, Loader)
        
    data.update(entries)

    config = Configuration(**data)
    
    return config