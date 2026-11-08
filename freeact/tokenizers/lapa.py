"""LAPA latent action tokenizer (LAQ, ``laq_openx.pt``).

LAPA's LAM differs from UniVLA's in four ways that all have to be handled here,
because none of them is visible from the checkpoint's file name:

===================  =========================  ==========================
                     UniVLA                     LAPA LAQ
===================  =========================  ==========================
video layout         ``(B, T, C, H, W)``        ``(B, C, T, H, W)``
input resolution     224 (DINOv2 ViT-B/14)      **256** (asserted, /32)
pixel preprocessing  ImageNet mean/std inside   raw ``[0, 1]``, no norm
codebook             16 codes x 4 positions     **8** codes x 4 positions
===================  =========================  ==========================

Frames arrive from ``MixedLeRobotDataset`` as ``[0, 1]`` floats.  Inputs at another
resolution (e.g. the policy's 224 px frames) are resized to 256 here.  LAPA's own
labeling script also feeds the model through a plain ``Resize((256, 256))``.
"""

import re
from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .laq import LatentActionQuantization


class LapaLatentActionModel(nn.Module):
    """Wraps LAPA's LAQ with the common tokenizer interface of :mod:`freeact.tokenizers`."""

    def __init__(self, laq, image_size: int, num_latents: int, num_codes: int):
        super().__init__()
        self.laq = laq
        self.input_resolution = image_size
        self.num_latents = num_latents
        self.num_codes = num_codes

    @torch.no_grad()
    def encode_pairs(self, frame1: Tensor, frame2: Tensor) -> Tensor:
        """``(B, C, H, W)`` pairs of ``[0, 1]`` RGB -> ``(B, num_codes)`` indices."""
        if frame1.shape[-1] != self.input_resolution:
            size = (self.input_resolution, self.input_resolution)
            frame1 = F.interpolate(frame1, size=size, mode="bilinear", align_corners=False)
            frame2 = F.interpolate(frame2, size=size, mode="bilinear", align_corners=False)
        # LAQ wants (B, C, T, H, W), not (B, T, C, H, W).
        video = torch.stack([frame1, frame2], dim=2)
        # ``inference`` and ``forward`` pick the codes with the same argmin; the
        # former skips the noise-substitution sampling and the ``.cpu()`` sync on
        # the usage counters, so it is both faster and side-effect free.
        return self.laq.inference(video, return_only_codebook_ids=True)

    @property
    def codebook(self) -> Tensor:
        return self.laq.vq.codebooks

    @property
    def device(self):
        return next(self.parameters()).device


def _infer_code_seq_len(state: Dict[str, Tensor], grid: int) -> int:
    """Recover ``code_seq_len`` by replaying the NSVQ cnn_encoder on ``grid``."""
    convs = sorted(
        {int(m.group(1)) for k in state if (m := re.match(r"vq\.cnn_encoder\.(\d+)\.weight$", k))}
    )
    h = grid
    for i in convs:
        w = state[f"vq.cnn_encoder.{i}.weight"]
        k = w.shape[-1]
        # The published configs use stride 2 + padding 1 for the first conv and
        # stride 1 + padding 0 for the rest.
        stride, pad = (2, 1) if i == convs[0] else (1, 0)
        h = (h + 2 * pad - k) // stride + 1
    return h * h


def build_lapa_lam_from_checkpoint(
    path: str,
    device: Optional[str] = None,
    dim_head: int = 64,
    image_size: int = 256,
    code_seq_len: int = 4,
) -> LapaLatentActionModel:
    """Load ``laq_openx.pt`` into a frozen :class:`LapaLatentActionModel`.

    ``image_size`` and ``code_seq_len`` default to the values LAPA published for
    ``laq_openx.pt`` (see LAPA's ``laq/inference_sthv2.py``) and are *authoritative*:
    neither is recoverable from tensor shapes, so a wrong guess loads cleanly and
    silently changes every token.  The shapes are still used, but only to
    cross-check -- a mismatch raises instead of being adopted.

    ``dim_head`` likewise cannot be recovered, but a wrong value does change the
    shapes of the attention projections, so it fails loudly by itself.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    num_latents, quant_dim = state["vq.codebooks"].shape
    dim = state["to_patch_emb_first_frame.2.weight"].shape[0]
    patch_size = int(round((state["to_patch_emb_first_frame.2.weight"].shape[1] / 3) ** 0.5))
    if image_size % patch_size:
        raise ValueError(
            f"image_size={image_size} is not divisible by the checkpoint's "
            f"patch_size={patch_size}"
        )
    grid = image_size // patch_size
    num_codes = code_seq_len
    # Cross-check against the conv stack.  _infer_code_seq_len assumes the
    # published stride/padding pattern, so it can be wrong on an unseen config;
    # treat a disagreement as "tell the user", not "trust the shapes".
    replayed = _infer_code_seq_len(state, grid)
    if replayed != num_codes:
        raise ValueError(
            f"code_seq_len={code_seq_len} with image_size={image_size} "
            f"(grid {grid}x{grid}) disagrees with the cnn_encoder in {path}, "
            f"which replays to {replayed}. One of them is wrong -- a mismatch "
            f"here loads fine and silently changes every token."
        )

    def _n_layers(prefix: str) -> int:
        idx = {int(m.group(1)) for k in state if (m := re.match(rf"{prefix}\.layers\.(\d+)\.", k))}
        return max(idx) + 1 if idx else 8

    build_ctx = (
        torch.cuda.device(device)
        if (device and "cuda" in str(device))
        else nullcontext()
    )
    with build_ctx:
        # NB: LatentActionQuantization hard-codes device='cuda' for NSVQ, so the
        # construction is scoped to this rank's device instead of cuda:0.
        laq = LatentActionQuantization(
            dim=dim,
            quant_dim=quant_dim,
            codebook_size=num_latents,
            image_size=image_size,
            patch_size=patch_size,
            spatial_depth=_n_layers("enc_spatial_transformer"),
            temporal_depth=_n_layers("enc_temporal_transformer"),
            dim_head=dim_head,
            heads=dim // dim_head,
            code_seq_len=num_codes,
        )

    # LatentActionQuantization.load_state_dict forces strict=False.  Nothing is legitimately
    # absent (AlibiPositionalBias's buffers are non-persistent, and spatial_rel_pos_bias.net.*
    # are trained weights), so any missing key would leave part of the encoder random.
    missing, unexpected = laq.load_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(
            f"LAPA LAQ checkpoint mismatch\n  missing: {list(missing)[:20]}\n"
            f"  unexpected: {list(unexpected)[:20]}"
        )

    laq.eval().requires_grad_(False)
    model = LapaLatentActionModel(laq, image_size, num_latents, num_codes)
    if device is not None:
        model.to(device)
        laq.vq.device = torch.device(device)
    return model
