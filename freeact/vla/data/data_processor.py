"""VLA training data: frame pairs + instructions rendered into the Qwen-VL chat template.

The dataset and the packing collator follow the Qwen3-VL finetuning code
(https://github.com/QwenLM/Qwen3-VL, Apache-2.0).

The assistant response is ``num_codes`` latent action tokens.  Because the frozen LAM runs on GPU,
the dataset emits placeholder tokens together with the raw frame pair, and the Trainer replaces
the placeholders with real codes batch-wise (``VLATrainer._encode_lam_batch``).
"""

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from torch.utils.data import Dataset

from freeact.data import MixedLeRobotDataset

from .latent_action_tokenizer import LatentActionTokenizer
from .rope2d import get_rope_index_3
from .utils import (
    build_vla_training_message,
    extract_action_stats,
    normalize_action,
    rank0_print,
    tensor_to_pil,
)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
ASSISTANT_TOKEN_INDEX = 77091  # "assistant" in the Qwen3-VL vocabulary
IM_END_TOKEN_INDEX = 151645  # <|im_end|>


def parse_dataset_specs(dataset_use: str) -> List[Tuple[str, float, bool]]:
    """Parse ``'repo_id:weight:cam_calib,...'`` into ``[(repo_id, weight, cam_calib), ...]``."""
    specs = []
    for item in dataset_use.split(","):
        if not (item := item.strip()):
            continue
        parts = item.split(":")
        specs.append(
            (
                parts[0],
                float(parts[1]) if len(parts) > 1 else 1.0,
                len(parts) > 2 and parts[2].lower() == "true",
            )
        )
    return specs


def update_processor_pixels(processor, data_args):
    """Fix the image resolution budget of the Qwen-VL image processor."""
    ip = processor.image_processor
    ip.min_pixels = data_args.min_pixels
    ip.max_pixels = data_args.max_pixels
    if isinstance(getattr(ip, "size", None), dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
    rank0_print(f"Image processor pixels: [{ip.min_pixels}, {ip.max_pixels}]")
    return processor


class LeRobotSupervisedDataset(Dataset):
    """Wraps :class:`MixedLeRobotDataset` into Qwen-VL training samples.

    pretrain (stage 2): latent-token cross-entropy only, no action labels needed.
    finetune (stage 3): additionally returns the normalized action chunk over ``delta_time`` for
    the continuous action head; video-only samples get a dummy (masked) action.
    """

    def __init__(self, processor, data_args, training_phase: str = "pretrain"):
        super().__init__()
        assert training_phase in ("pretrain", "finetune"), training_phase
        self.training_phase = training_phase

        dataset_specs = parse_dataset_specs(data_args.dataset_use)
        rank0_print(f"LeRobot dataset specs: {dataset_specs}")

        # Action-free (generated) videos, stage 3 only
        video_only_specs = None
        if data_args.video_dataset_use:
            video_only_specs = [
                (repo_id, weight, data_args.lerobot_root)
                for repo_id, weight, _ in parse_dataset_specs(data_args.video_dataset_use)
            ]
            rank0_print(f"Video-only dataset specs: {video_only_specs}")

        self.skip_action = training_phase == "pretrain"
        self.lerobot_dataset = MixedLeRobotDataset(
            dataset_specs=dataset_specs,
            resolution=data_args.lerobot_resolution,
            image_aug=data_args.lerobot_image_aug,
            action_mode="sequence",
            root=data_args.lerobot_root,
            filter_identical_frames=True,
            delta_time=data_args.lerobot_delta_time,
            skip_action=self.skip_action,
            video_only_specs=video_only_specs,
        )
        rank0_print(f"LeRobot dataset: {len(self.lerobot_dataset)} samples")

        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = self.processor.tokenizer
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        self.action_dim = data_args.action_dim

        # q01/q99 action normalization (stage 3)
        self.action_stats = {}
        if not self.skip_action:
            action_datasets = [
                ds
                for ds, vo in zip(self.lerobot_dataset.datasets, self.lerobot_dataset.video_only_flags)
                if not vo
            ]
            for ds in action_datasets:
                if (name := ds.meta.repo_id) not in self.action_stats:
                    self.action_stats[name] = extract_action_stats(ds)
            # Optionally share one dataset's statistics across all action datasets, e.g. the
            # handful of self-evolution rollouts reuse the source-domain (libero_spatial) scale.
            if ref := data_args.action_norm_dataset:
                ref_stats = self.action_stats[ref]
                self.action_stats = {name: ref_stats for name in self.action_stats}
            for name, s in self.action_stats.items():
                rank0_print(f"Action stats [{name}]: q01={s['q01']}, q99={s['q99']}")

            fps_set = {ds.meta.fps for ds in action_datasets}
            assert len(fps_set) == 1, f"Finetune requires the same fps across action datasets, got {fps_set}"
            fps = fps_set.pop()
            self.action_window_size = math.ceil(data_args.lerobot_delta_time * fps)
            rank0_print(f"Action chunk: fps={fps}, delta_time={data_args.lerobot_delta_time}s -> W={self.action_window_size}")

        self.latent_action_tokenizer = LatentActionTokenizer(
            tokenizer=self.tokenizer,
            checkpoint_path=data_args.lam_checkpoint,
        )
        tok = self.latent_action_tokenizer
        rank0_print(f"LAM tokenizer [{tok.backend}]: {tok.num_latents} latents x {tok.num_codes} codes")

        self._len = len(self.lerobot_dataset)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        for _ in range(20):
            if (s := self.lerobot_dataset[idx]).get("task_instruction", "").strip():
                break
            idx = random.randrange(self._len)
        else:
            raise RuntimeError("No sample with non-empty task after 20 retries")

        messages = build_vla_training_message(
            tensor_to_pil(s["initial_pixel_values"]),
            s["task_instruction"],
            self.latent_action_tokenizer.placeholder_str,
        )
        data_dict = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = data_dict["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)
        data_dict["input_ids"] = input_ids
        data_dict["labels"] = self._create_labels(input_ids)

        grid_thw = data_dict.get("image_grid_thw")
        if grid_thw is not None and not isinstance(grid_thw, Sequence):
            grid_thw = [grid_thw]
        position_ids, _ = get_rope_index_3(
            self.merge_size,
            input_ids,
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        )
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [input_ids.size(1)]

        video_only = s.get("video_only", False)
        if not self.skip_action:
            if video_only:
                # Dummy action, masked out of the action loss
                data_dict["continuous_action"] = torch.zeros(
                    self.action_window_size, self.action_dim, dtype=torch.float32
                )
            else:
                stats = self.action_stats[s["dataset_name"]]
                data_dict["continuous_action"] = torch.from_numpy(
                    normalize_action(s["action"], stats["q01"], stats["q99"])
                ).float()
        if video_only:
            data_dict["is_video_only"] = True

        # Frame pair for batched LAM encoding in the Trainer
        data_dict["_lam_frame1"] = s["initial_pixel_values"]
        data_dict["_lam_frame2"] = s["target_pixel_values"]
        return data_dict

    @staticmethod
    def _create_labels(input_ids: torch.Tensor) -> torch.Tensor:
        """Supervise only the assistant response (latent tokens + <|im_end|>\\n)."""
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        ids = input_ids[0].tolist()
        pos = 0
        while pos < len(ids):
            if ids[pos] == ASSISTANT_TOKEN_INDEX:
                start = end = pos + 2
                while end < len(ids) and ids[end] != IM_END_TOKEN_INDEX:
                    end += 1
                if end < len(ids):
                    labels[0, start : end + 2] = input_ids[0, start : end + 2]
                pos = end
            pos += 1
        return labels


def _pop_lam_frames(instances: Sequence[Dict]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if "_lam_frame1" not in instances[0]:
        return None
    return (
        torch.stack([inst.pop("_lam_frame1") for inst in instances]),
        torch.stack([inst.pop("_lam_frame2") for inst in instances]),
    )


def _collate_extras(instances: Sequence[Dict], batch: Dict) -> None:
    batch["pixel_values"] = torch.cat([inst["pixel_values"] for inst in instances], dim=0)
    batch["image_grid_thw"] = torch.cat([inst["image_grid_thw"] for inst in instances])
    if any("continuous_action" in inst for inst in instances):
        batch["continuous_action"] = torch.stack([inst["continuous_action"] for inst in instances])
    # Always present so every rank has a consistent batch structure (gradient projection all_reduce)
    batch["video_only_mask"] = torch.tensor(
        [inst.get("is_video_only", False) for inst in instances], dtype=torch.bool
    )


@dataclass
class FlattenedDataCollatorForSupervisedDataset:
    """Packs a batch into a single sequence for varlen FlashAttention."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        lam_frames = _pop_lam_frames(instances)
        seq_lens = list(itertools.chain(*[inst["attention_mask"] for inst in instances]))
        cumsum_seq_lens = torch.cumsum(
            torch.tensor([0] + seq_lens, dtype=torch.int32), dim=0, dtype=torch.int32
        )
        batch = dict(
            input_ids=torch.cat([inst["input_ids"] for inst in instances], dim=1),
            labels=torch.cat([inst["labels"] for inst in instances], dim=1),
            attention_mask=cumsum_seq_lens,
            position_ids=torch.cat([inst["position_ids"] for inst in instances], dim=2),
        )
        _collate_extras(instances, batch)
        if lam_frames is not None:
            batch["_lam_frames"] = lam_frames
        return batch


def make_supervised_data_module(processor, data_args, training_phase: str = "pretrain") -> Dict:
    train_dataset = LeRobotSupervisedDataset(processor, data_args=data_args, training_phase=training_phase)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=FlattenedDataCollatorForSupervisedDataset(processor.tokenizer),
    )
