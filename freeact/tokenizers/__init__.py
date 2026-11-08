"""Frozen latent action models used as tokenizers for the VLA (stages 2-3) and for the RQ1 analysis.

Three backends share one interface (``encode_pairs``, ``codebook``, ``num_latents``, ``num_codes``):

===========  ==========================================  ============  ==========
backend      checkpoint                                  vocabulary    input
===========  ==========================================  ============  ==========
``freeact``  ``Mo-ZheHan/FreeAct``: ``lam/lam.ckpt``     64 x 4 codes  224 px
``univla``   ``qwbu/univla-latent-action-model``         16 x 4 codes  224 px
``lapa``     ``latent-action-pretraining/LAPA-7B-openx`` 8 x 4 codes   256 px
===========  ==========================================  ============  ==========

All consume ``[0, 1]`` RGB frame pairs and handle their own resizing.  The backend is detected
from the checkpoint's parameter names.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

BACKENDS = ("freeact", "univla", "lapa")


class FreeActTokenizer(nn.Module):
    """Wraps the stage-1 :class:`DINOLatentActionModel` with the common tokenizer interface."""

    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        self.num_latents = lam.vq.num_latents
        self.num_codes = lam.num_codes

    @torch.no_grad()
    def encode_pairs(self, frame1: Tensor, frame2: Tensor) -> Tensor:
        """``(B, C, H, W)`` pairs of ``[0, 1]`` RGB -> ``(B, num_codes)`` indices."""
        return self.lam.vq_encode(torch.stack([frame1, frame2], dim=1))["indices"]

    @property
    def codebook(self) -> Tensor:
        return self.lam.vq.codebook.weight


def detect_backend(path: str) -> str:
    """Identify the latent action model family from a checkpoint's parameter names."""
    if backend := os.environ.get("LAM_BACKEND", "").lower():
        if backend not in BACKENDS:
            raise ValueError(f"LAM_BACKEND={backend!r} must be one of {BACKENDS}")
        return backend
    raw = torch.load(path, map_location="meta", weights_only=False, mmap=True)
    keys = set((raw.get("state_dict", raw) if isinstance(raw, dict) else raw).keys())
    if any(k.endswith("vq_action.codebook.weight") for k in keys):
        return "univla"
    if any(k.endswith("vq.codebooks") for k in keys):
        return "lapa"
    if any(k.endswith("action_decoder.net.0.weight") for k in keys):
        return "freeact"
    raise ValueError(f"Cannot tell which latent action model {path} is; set LAM_BACKEND explicitly.")


def load_tokenizer(path: str, device: Optional[str] = None) -> nn.Module:
    """Load a frozen latent action tokenizer from ``path`` (backend auto-detected)."""
    backend = detect_backend(path)
    if backend == "freeact":
        from freeact.lam import load_lam

        model = FreeActTokenizer(load_lam(path, device))
    elif backend == "univla":
        from .univla import build_univla_lam_from_checkpoint

        model = build_univla_lam_from_checkpoint(path, device=device)
    else:
        from .lapa import build_lapa_lam_from_checkpoint

        model = build_lapa_lam_from_checkpoint(path, device=device)
    model.backend = backend
    return model.eval().requires_grad_(False)
