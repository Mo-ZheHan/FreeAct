"""Closed-loop inference with a FreeAct stage-3 policy."""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

from freeact.data import libero_from_camera_frame

from .data.data_processor import IMAGE_TOKEN_INDEX
from .data.utils import build_vla_message, denormalize_action, tensor_to_pil
from .model_utils import load_model_and_processor, resize_embeddings_if_needed
from .train.action_decoder import VLAActionDecoder

__all__ = ["VLAPolicy", "load_vla_model"]


def center_crop_resize(img: torch.Tensor, size: int, scale: float = 0.9) -> torch.Tensor:
    """Deterministic counterpart of the training-time ``RandomResizedCrop(scale=(s, s), ratio=1)``."""
    _, h, w = img.shape
    crop_side = min(int(round(math.sqrt(scale * h * w))), h, w)
    top, left = (h - crop_side) // 2, (w - crop_side) // 2
    cropped = img[:, top : top + crop_side, left : left + crop_side]
    return v2.functional.resize(cropped, [size, size], antialias=True)


class VLAPolicy(nn.Module):
    """Qwen3-VL + MAP action head.

    The VLA samples ``num_codes`` latent action tokens; the hidden states of those tokens and of
    the image tokens are decoded into a normalized camera-frame action chunk, which is
    denormalized and (for LIBERO) rotated back into the robot base frame.
    """

    def __init__(
        self,
        model,
        processor,
        action_decoder: VLAActionDecoder,
        action_stats: Dict[str, np.ndarray],
        device: str = "cuda",
        center_crop: bool = True,
        crop_scale: float = 0.9,
        resolution: int = 224,
        camera_frame: bool = True,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.action_decoder = action_decoder
        self.action_stats = action_stats
        self.device = device
        self.center_crop = center_crop
        self.crop_scale = crop_scale
        self.resolution = resolution
        self.camera_frame = camera_frame

    def _preprocess(self, img: torch.Tensor):
        if self.center_crop:
            img = center_crop_resize(img, self.resolution, self.crop_scale)
        return tensor_to_pil(img)

    @torch.inference_mode()
    def select_action(self, obs: Dict[str, Any]) -> torch.Tensor:
        """``obs["observation.images.image"]``: (B, C, H, W) in [0, 1]; ``obs["task"]``: list[str]."""
        images = obs["observation.images.image"]
        tasks = obs.get("task", [""] * images.shape[0])

        actions = []
        for i, img in enumerate(images):
            task = tasks[i] if isinstance(tasks, (list, tuple)) else tasks
            inputs = self.processor.apply_chat_template(
                build_vla_message(self._preprocess(img), task),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            action = self._predict(inputs)
            if self.camera_frame:
                action = libero_from_camera_frame(action)
            actions.append(action)
        return torch.tensor(np.stack(actions), dtype=torch.float32, device=self.device)

    def _predict(self, inputs: Dict) -> np.ndarray:
        num_codes = self.action_decoder.num_codes
        # num_codes + 1 generation steps yield num_codes hidden states (step 0 is the prefill)
        out = self.model.generate(
            **inputs,
            max_new_tokens=num_codes + 1,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        prefill = out.hidden_states[0][-1]  # [1, input_len, H]
        visual_embed = prefill[:, inputs["input_ids"][0] == IMAGE_TOKEN_INDEX, :]
        latent_action = torch.cat([out.hidden_states[i][-1] for i in range(1, num_codes + 1)], dim=1)

        pred = self.action_decoder(latent_action.float(), visual_embed.float())[0].float().cpu().numpy()
        action = pred[0]
        return denormalize_action(action, self.action_stats["q01"], self.action_stats["q99"])


def _load_action_decoder(vla_path: Path, device: str) -> VLAActionDecoder:
    with open(vla_path / "action_decoder_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    decoder = VLAActionDecoder(**cfg)
    decoder.load_state_dict(torch.load(vla_path / "action_decoder.pt", map_location=device, weights_only=True))
    logging.info("Action head: window=%d, latent action tokens=%d", cfg["window_size"], cfg["num_codes"])
    return decoder.to(device).eval()


def load_vla_model(
    vla_path: str,
    device: str = "cuda",
    max_pixels: int = 50176,
    min_pixels: int = 784,
    **policy_kwargs,
) -> VLAPolicy:
    """Load a stage-3 checkpoint (LoRA adapter + ``action_decoder.pt`` + ``action_stats.json``)."""
    from peft import PeftConfig, PeftModel

    path = Path(vla_path)
    model, processor = load_model_and_processor(str(path), device)
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    resize_embeddings_if_needed(model, processor.tokenizer)

    cfg = PeftConfig.from_pretrained(str(path))
    cfg.ensure_weight_tying = True
    model = PeftModel.from_pretrained(model, str(path), config=cfg).merge_and_unload(safe_merge=True)

    with open(path / "action_stats.json", "r", encoding="utf-8") as f:
        stats = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(f).items()}

    return VLAPolicy(
        model.eval(),
        processor,
        _load_action_decoder(path, device),
        stats,
        device,
        **policy_kwargs,
    )
