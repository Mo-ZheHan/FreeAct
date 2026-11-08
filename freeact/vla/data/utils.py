"""Shared utilities for VLA data processing."""

from typing import Dict

import numpy as np
import torch
from PIL import Image

from freeact.data import get_action_key

# Local rank of this process (set by train.py); rank0_print prints on local rank 0 only
local_rank = None


def rank0_print(*args):
    """Print only on rank 0."""
    if local_rank is None or local_rank == 0:
        print(*args)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [C,H,W] float tensor in [0, 1] to PIL Image."""
    tensor = tensor.cpu()
    return Image.fromarray(
        (tensor.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).numpy()
    )


def build_vla_message(image: Image.Image, task: str) -> list:
    """Build the user turn of the VLA chat template."""
    prompt = (
        f"What action should the robot take to {task.lower()}?"
        if task
        else "What action should the robot take?"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_vla_training_message(image: Image.Image, task: str, action_str: str) -> list:
    """Build VLA training message with the latent action tokens as the assistant response."""
    return build_vla_message(image, task) + [
        {"role": "assistant", "content": [{"type": "text", "text": action_str}]}
    ]


def extract_action_stats(dataset) -> Dict[str, np.ndarray]:
    """Extract action q01/q99 stats from dataset metadata."""
    stats = dataset.meta.stats[get_action_key(dataset.meta.repo_id)]
    return {k: np.asarray(stats[k], dtype=np.float32) for k in ("q01", "q99")}


def normalize_action(action: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Normalize action to [-1, 1] using q01/q99 quantiles."""
    return np.clip(2.0 * (action - q01) / np.maximum(q99 - q01, 1e-8) - 1.0, -1.0, 1.0)


def denormalize_action(norm: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Denormalize action from [-1, 1] using q01/q99 quantiles."""
    return (norm + 1.0) / 2.0 * (q99 - q01) + q01
