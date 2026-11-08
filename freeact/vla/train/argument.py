"""Command-line arguments of stages 2-3, extending those of the Qwen3-VL finetuning code
(https://github.com/QwenLM/Qwen3-VL, Apache-2.0)."""

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen3-VL-4B-Instruct"


@dataclass
class DataArguments:
    # Robot datasets: "repo_id:weight:cam_calib,..." (cam_calib: camera extrinsics available, so
    # actions are expressed in the camera frame)
    dataset_use: str = ""
    lerobot_root: Optional[str] = None
    lerobot_resolution: int = 224
    lerobot_image_aug: bool = True
    lerobot_delta_time: float = 1.0
    # Action-free (generated) videos for stage 3: "repo_id:weight,..."
    video_dataset_use: str = ""
    # Share this dataset's q01/q99 action statistics across all action datasets (stage 3)
    action_norm_dataset: Optional[str] = None

    action_dim: int = 7
    lam_checkpoint: Optional[str] = None

    # Qwen-VL image budget (224x224 -> 50176)
    max_pixels: int = 50176
    min_pixels: int = 784


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = "adamw_torch"

    # LoRA (stage 2 creates the adapter, stage 3 continues training it)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0

    # "pretrain" (stage 2: latent tokens only) | "finetune" (stage 3: + continuous action head)
    vla_training_phase: str = field(default="pretrain", metadata={"choices": ["pretrain", "finetune"]})
    action_decoder_hidden_dim: int = 512
    action_decoder_lr: float = 3.5e-4
    action_decoder_weight_decay: float = 1e-3

    # Gradient projection between the source (action) and generated (video-only) streams
    grad_proj: bool = True
