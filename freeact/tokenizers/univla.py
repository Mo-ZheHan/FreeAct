"""UniVLA's task-centric latent action model (the output of its second training stage).

Vendored from ``OpenDriveLab/UniVLA`` (Apache-2.0; ``latent_action_model/genie/modules/lam.py``,
class ``ControllableDINOLatentActionModel``) so that the released
``lam-stage-2.ckpt`` can be used as a frozen tokenizer inside this repository.

Differences from the upstream file:

* the transformer / VQ blocks are imported from ``freeact.lam.modules.blocks``
  (structurally identical);
* ``num_codes`` is a constructor argument instead of a hard-coded ``4``;
* the DINOv2 backbone can be built without downloading pretrained weights,
  because the released checkpoint already carries them;
* training-only code (losses, optimizer setup) has been dropped.

Parameter names are kept identical to upstream so that the released ``state_dict``
loads without renaming; missing and unexpected keys are checked explicitly.
"""

import os
import re
from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torchvision import transforms

from freeact.lam.modules.blocks import (
    SpatioTemporalTransformer,
    SpatioTransformer,
    VectorQuantizer,
)

# timm's names, matching upstream UniVLA
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DINOV2_HUB_REPO = "facebookresearch/dinov2"
DINOV2_ENTRYPOINT = "dinov2_vitb14_reg"
DINOV2_DIM = 768


def _build_dinov2(pretrained: bool = False) -> nn.Module:
    """Instantiate DINOv2 ViT-B/14 (with registers).

    ``UNIVLA_DINOV2_DIR`` may point at a local clone of ``facebookresearch/dinov2``
    to avoid hitting the network from every DDP rank.
    """
    local_dir = os.environ.get("UNIVLA_DINOV2_DIR")
    if local_dir and os.path.isdir(local_dir):
        return torch.hub.load(
            local_dir, DINOV2_ENTRYPOINT, source="local", pretrained=pretrained
        )
    return torch.hub.load(DINOV2_HUB_REPO, DINOV2_ENTRYPOINT, pretrained=pretrained)


class UniVLALatentActionModel(nn.Module):
    """UniVLA's task-centric latent action VQ-VAE.

    Two codebooks operate on the same DINOv2 features: ``vq`` holds the frozen
    stage-1 "uncontrollable" codes, ``vq_action`` holds the task-centric codes
    that UniVLA feeds to the policy.  ``vq_encode`` returns the latter as
    ``indices``.
    """

    def __init__(
        self,
        in_dim: int = 3,
        model_dim: int = 768,
        latent_dim: int = 128,
        num_latents: int = 16,
        num_codes: int = 4,
        patch_size: int = 14,
        enc_blocks: int = 12,
        dec_blocks: int = 12,
        num_heads: int = 12,
        dropout: float = 0.0,
        uncontrol_num_latents: int = 16,
        dino_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_codes = num_codes
        # DINOv2 ViT-B/14 at 224 gives the 16x16 patch grid the LAM was trained on.
        self.input_resolution = 224

        self.dino_transform = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )
        self.dino_encoder = _build_dinov2(pretrained=dino_pretrained)
        self.dino_encoder.requires_grad_(False)

        dino_dim = DINOV2_DIM

        # Stage-1 ("uncontrollable") query tokens + codebook, frozen in stage-2.
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=uncontrol_num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )

        # Decoder: spatial transformer over [action, uncontrol, patches]
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Task-centric codebook -- this is what the policy consumes.
        self.vq_action = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        self.action_latent_controllable = nn.Parameter(
            torch.empty(1, 1, self.num_codes, dino_dim)
        )
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

    def vq_encode(self, videos: Tensor) -> Dict:
        """Encode a ``(B, T, C, H, W)`` clip of ``[0, 1]`` RGB frames.

        Returns the task-centric ``indices`` of shape ``(B * (T - 1), num_codes)``
        alongside the intermediate quantities upstream exposes.
        """
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dino_features = self.dino_encoder.forward_features(videos)[
            "x_norm_patchtokens"
        ]
        dino_features = rearrange(dino_features, "(b T) l d -> b T l d", T=T)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dino_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        z = self.encoder(padded_patches)

        # "Uncontrollable" latents (stage-1 codebook, frozen)
        z_uncontrol = self.to_codebook_uncontrol(
            z[:, 1:, self.num_codes : self.num_codes * 2]
        )
        z_uncontrol = z_uncontrol.reshape(
            B * (T - 1), self.num_codes, self.latent_dim
        )
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(
            z_uncontrol
        )
        z_q_uncontrol = z_q_uncontrol.reshape(
            B, T - 1, self.num_codes, self.latent_dim
        )

        # Task-centric latents
        z_action = self.to_codebook(z[:, 1:, : self.num_codes])
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": dino_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices": indices,
            "indices_uncontrol": indices_uncontrol,
        }

    @property
    def num_latents(self) -> int:
        """Size of the task-centric codebook (the policy's token vocabulary)."""
        return self.vq_action.num_latents

    @property
    def codebook(self) -> Tensor:
        return self.vq_action.codebook.weight

    @torch.no_grad()
    def encode_pairs(self, frame1: Tensor, frame2: Tensor) -> Tensor:
        """``(B, C, H, W)`` pairs of ``[0, 1]`` RGB -> ``(B, num_codes)`` indices."""
        if frame1.shape[-1] != self.input_resolution:
            size = (self.input_resolution, self.input_resolution)
            frame1 = nn.functional.interpolate(
                frame1, size=size, mode="bilinear", align_corners=False
            )
            frame2 = nn.functional.interpolate(
                frame2, size=size, mode="bilinear", align_corners=False
            )
        videos = torch.stack([frame1, frame2], dim=1)  # (B, T, C, H, W)
        return self.vq_encode(videos)["indices"]

    def forward(self, batch: Dict) -> Dict:
        outputs = self.vq_encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        video_action_patches = torch.cat(
            [
                self.action_up(outputs["z_q"]),
                self.action_up_uncontrol(outputs["z_q_uncontrol"]),
                video_patches,
            ],
            dim=2,
        )
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, -video_patches.shape[2] :]
        outputs.update(
            {"recon": video_recon, "target": outputs["patches"][:, [-1]]}
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device


def _no_usage_tracking(*args, **kwargs) -> None:
    """Replacement for ``VectorQuantizer.update_usage`` on a frozen codebook.

    The usage counters only serve dead-code restarts during LAM training, which
    a frozen tokenizer never performs.
    """


def _strip_prefix(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Drop the Lightning ``lam.`` wrapper prefix, as UniVLA's loaders do."""
    return {k[4:] if k.startswith("lam.") else k: v for k, v in state.items()}


def _count_blocks(state: Dict[str, Tensor], prefix: str) -> int:
    pat = re.compile(rf"^{re.escape(prefix)}\.transformer_blocks\.(\d+)\.")
    idx = {int(m.group(1)) for k in state if (m := pat.match(k))}
    return max(idx) + 1 if idx else 12


def build_univla_lam_from_checkpoint(
    path: str,
    device: Optional[str] = None,
    num_heads: Optional[int] = None,
    patch_size: Optional[int] = None,
) -> "UniVLALatentActionModel":
    """Load ``lam-stage-2.ckpt`` into a frozen :class:`UniVLALatentActionModel`.

    Architecture dimensions come from the checkpoint's Lightning
    ``hyper_parameters`` where available, and are otherwise inferred from tensor
    shapes.  ``num_heads`` and ``patch_size`` in particular *must* come from the
    checkpoint: neither changes any tensor shape, so a wrong value loads with
    ``strict=True`` and silently produces different tokens.  Passing them
    explicitly overrides the checkpoint, which is only useful for a release that
    stores no hyper_parameters.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters") or {}
    state = _strip_prefix(ckpt.get("state_dict", ckpt))

    if num_heads is None:
        num_heads = hp.get("lam_num_heads")
    if patch_size is None:
        patch_size = hp.get("lam_patch_size")
    if num_heads is None or patch_size is None:
        raise ValueError(
            f"{path} stores no lam_num_heads/lam_patch_size in hyper_parameters "
            f"(got num_heads={num_heads}, patch_size={patch_size}). Neither can be "
            f"recovered from tensor shapes, and guessing silently changes the "
            f"tokens -- pass them explicitly."
        )

    num_latents, latent_dim = state["vq_action.codebook.weight"].shape
    uncontrol_num_latents = state["vq.codebook.weight"].shape[0]
    model_dim = state["to_codebook.weight"].shape[1]
    num_codes = state["action_latent_controllable"].shape[2]
    n_dino = sum(1 for k in state if k.startswith("dino_encoder."))

    lam = UniVLALatentActionModel(
        in_dim=3,
        model_dim=model_dim,
        latent_dim=latent_dim,
        num_latents=num_latents,
        num_codes=num_codes,
        patch_size=patch_size,
        enc_blocks=_count_blocks(state, "encoder"),
        dec_blocks=_count_blocks(state, "decoder"),
        num_heads=num_heads,
        dropout=0.0,
        uncontrol_num_latents=uncontrol_num_latents,
        dino_pretrained=n_dino == 0,
    )

    # The DINO backbone is either fully in the checkpoint or fully absent (and
    # then downloaded).  A partial set would be silently half-random, so require
    # all-or-nothing rather than exempting the whole prefix from the check.
    n_dino_expected = sum(
        1 for k in lam.state_dict() if k.startswith("dino_encoder.")
    )
    if n_dino not in (0, n_dino_expected):
        raise RuntimeError(
            f"{path} holds {n_dino} of {n_dino_expected} dino_encoder tensors. "
            f"A partial backbone would leave the rest randomly initialized."
        )

    missing, unexpected = lam.load_state_dict(state, strict=False)
    exempt = (lambda k: k.startswith("dino_encoder.")) if n_dino == 0 else (lambda k: False)
    hard_missing = [k for k in missing if not exempt(k)]
    if hard_missing or unexpected:
        raise RuntimeError(
            f"UniVLA LAM checkpoint mismatch\n  missing: {hard_missing[:20]}\n"
            f"  unexpected: {list(unexpected)[:20]}"
        )

    # Frozen tokenizer: skip the VQ usage counters (``VectorQuantizer.forward``
    # updates them in eval mode regardless of ``code_restart``).
    for module in lam.modules():
        if isinstance(module, VectorQuantizer):
            module.code_restart = False
            # A bound lambda here would make the whole LAM unpicklable, which
            # breaks any dataloader using spawn/forkserver.
            module.update_usage = _no_usage_tracking

    lam.eval().requires_grad_(False)
    if device is not None:
        lam.to(device)
    return lam
