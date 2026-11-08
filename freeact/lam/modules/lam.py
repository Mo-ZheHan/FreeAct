"""Physically grounded latent action model (FreeAct stage 1).

A frozen DINOv3 ViT encodes a frame pair; a spatio-temporal transformer distills ``num_codes``
latent action tokens that are vector-quantized against a ``num_latents``-entry codebook.  A
spatial decoder reconstructs the DINO features of the future frame from the current frame and the
quantized latents, and an auxiliary MLP Action Decoder regresses the camera-frame end-effector
action from the quantized latents.
"""

from typing import Dict

import timm
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torchvision import transforms

from .blocks import SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Weights released by Meta under the DINOv3 License (see THIRD_PARTY_NOTICES.md), loaded through timm
DINO_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
DINO_DIM = 768
DINO_NUM_PREFIX_TOKENS = 5  # CLS + 4 register tokens


class ActionDecoder(nn.Module):
    """MLP that maps the quantized latent actions ``(B, num_codes, latent_dim)`` to a 7D action."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, action_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z.reshape(z.shape[0], -1))


class HomoUncActionLoss(nn.Module):
    """
    SmoothL1 action loss with homoscedastic uncertainty weighting (Kendall et al., 2018).

    The 7D action is split into translation [0:3], rotation [3:6] and gripper [6], and each
    component is weighted as ``exp(-s_i) * L_i + s_i`` with a learnable ``s_i = log(sigma_i^2)``.
    """

    def __init__(self):
        super().__init__()
        self.log_sigma2_pos = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2_rot = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2_grip = nn.Parameter(torch.tensor(0.0))
        self.base_loss = nn.SmoothL1Loss(reduction="mean")

    def forward(self, pred_action: Tensor, target_action: Tensor) -> tuple[Tensor, Dict[str, Tensor]]:
        L_pos = self.base_loss(pred_action[..., 0:3], target_action[..., 0:3])
        L_rot = self.base_loss(pred_action[..., 3:6], target_action[..., 3:6])
        L_grip = self.base_loss(pred_action[..., 6:7], target_action[..., 6:7])

        total = (
            torch.exp(-self.log_sigma2_pos) * L_pos + self.log_sigma2_pos
            + torch.exp(-self.log_sigma2_rot) * L_rot + self.log_sigma2_rot
            + torch.exp(-self.log_sigma2_grip) * L_grip + self.log_sigma2_grip
        )

        loss_dict = {
            "L_pos": L_pos.detach(),
            "L_rot": L_rot.detach(),
            "L_grip": L_grip.detach(),
            "log_sigma2_pos": self.log_sigma2_pos.detach(),
            "log_sigma2_rot": self.log_sigma2_rot.detach(),
            "log_sigma2_grip": self.log_sigma2_grip.detach(),
        }
        return total, loss_dict


class DINOLatentActionModel(nn.Module):
    """Latent action VQ-VAE operating in the DINOv3 feature space."""

    def __init__(
        self,
        model_dim: int,
        latent_dim: int,
        num_latents: int,
        num_codes: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_codes = num_codes

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = timm.create_model(DINO_MODEL_NAME, pretrained=pretrained_backbone, num_classes=0)
        self.dino_encoder.requires_grad_(False)

        # Learnable query tokens that become the latent actions
        self.action_latent = nn.Parameter(torch.empty(1, 1, num_codes, DINO_DIM))
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=DINO_DIM,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )
        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(num_latents=num_latents, latent_dim=latent_dim, code_restart=True)

        # Reconstruction decoder
        self.patch_up = nn.Linear(DINO_DIM, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=DINO_DIM,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Auxiliary Action Decoder (physical grounding)
        action_input_dim = num_codes * latent_dim
        self.action_decoder = ActionDecoder(
            input_dim=action_input_dim, hidden_dim=action_input_dim // 2, action_dim=7
        )

    def vq_encode(self, videos: Tensor) -> Dict:
        """Encode ``(B, 2, C, H, W)`` frame pairs in ``[0, 1]`` into quantized latent actions."""
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        dino_output = self.dino_encoder.forward_features(self.dino_transform(videos))
        dino_features = dino_output[:, DINO_NUM_PREFIX_TOKENS:]  # (B*T, 196, 768) patch tokens
        dino_features = rearrange(dino_features, "(b T) l d -> b T l d", T=T)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        z = self.encoder(torch.cat([action_pad, dino_features], dim=2))

        # Latent actions are read out at the future frame
        z = self.to_codebook(z[:, 1:, : self.num_codes])  # (B, T-1, num_codes, latent_dim)
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z_code, z_e, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {"patches": dino_features, "z_q": z_q, "z_code": z_code, "z_e": z_e, "indices": indices}

    def forward(self, batch: Dict) -> Dict:
        outputs = self.vq_encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])

        video_recon = self.decoder(torch.cat([action_patches, video_patches], dim=2))
        video_recon = video_recon[:, :, self.num_codes : self.num_codes + video_patches.shape[2]]

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]],
                "pred_action": self.action_decoder(outputs["z_q"].squeeze(1)),
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
