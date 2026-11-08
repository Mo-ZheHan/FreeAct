"""Lightning module for FreeAct stage 1: the physically grounded latent action model (LAM)."""

from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.distributed
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

from .modules import DINOLatentActionModel, HomoUncActionLoss

OptimizerCallable = Callable[[Iterable], Optimizer]


class DINO_LAM(LightningModule):
    """
    Loss: L_rec + L_VQ + lambda_act * 1[calib] * L_act   (Eq. 1 of the paper)

    ``action_std_*`` rescale the camera-frame action targets per component before the
    uncertainty-weighted SmoothL1 loss (Eq. 2).
    """

    def __init__(
        self,
        lam_model_dim: int = 768,
        lam_latent_dim: int = 128,
        lam_num_latents: int = 64,
        lam_num_codes: int = 4,
        lam_enc_blocks: int = 12,
        lam_dec_blocks: int = 12,
        lam_num_heads: int = 12,
        lam_dropout: float = 0.0,
        vq_beta: float = 0.25,
        action_loss_weight: float = 0.01,
        action_std_pos: float = 1.0,
        action_std_rot: float = 1.0,
        action_std_grip: float = 1.0,
        optimizer: OptimizerCallable = AdamW,
        lam_patch_size: int = 16,  # recorded in the checkpoint; fixed by the DINOv3 ViT-B/16 backbone
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.lam = DINOLatentActionModel(
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            num_latents=lam_num_latents,
            num_codes=lam_num_codes,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
            pretrained_backbone=pretrained_backbone,
        )

        self.lam_num_latents = lam_num_latents
        self.vq_beta = vq_beta
        self.action_loss_weight = action_loss_weight
        self.optimizer = optimizer
        self.action_loss_fn = HomoUncActionLoss()

        action_scale = torch.tensor(
            [action_std_pos] * 3 + [action_std_rot] * 3 + [action_std_grip]
        )
        self.register_buffer("action_scale", action_scale)
        self.save_hyperparameters(ignore=["pretrained_backbone"])

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.lam(batch)

        # Reconstruction + VQ losses
        rec_loss = ((outputs["target"] - outputs["recon"]) ** 2).mean()
        codebook_loss = ((outputs["z_e"].detach() - outputs["z_code"]) ** 2).mean()
        commit_loss = ((outputs["z_e"] - outputs["z_code"].detach()) ** 2).mean()
        vq_loss = codebook_loss + self.vq_beta * commit_loss

        # Action loss, only on samples whose action is expressed in the camera frame
        pred_action = outputs["pred_action"]
        gt_action = batch["action"].float().to(pred_action.device)
        cam_calib_mask = batch["cam_calib_mask"].to(pred_action.device)

        if cam_calib_mask.any():
            action_loss, action_logs = self.action_loss_fn(
                pred_action[cam_calib_mask], gt_action[cam_calib_mask] / self.action_scale
            )
        else:
            # Keep the action branch in the graph so DDP sees every parameter
            dummy, action_logs = self.action_loss_fn(pred_action[:1], pred_action[:1].detach())
            action_loss = dummy * 0.0
            zero = torch.zeros((), device=pred_action.device)
            action_logs.update(L_pos=zero, L_rot=zero, L_grip=zero)

        loss = rec_loss + vq_loss + self.action_loss_weight * action_loss

        # Codebook usage / perplexity (global across ranks)
        counts = outputs["indices"].flatten().bincount(minlength=self.lam_num_latents).float()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        code_perplexity = (-(probs_nz * probs_nz.log()).sum()).exp()

        logs = {
            "rec_loss": rec_loss,
            "vq_loss": vq_loss,
            "action_loss": action_loss,
            "action_loss_pos": action_logs["L_pos"],
            "action_loss_rot": action_logs["L_rot"],
            "action_loss_grip": action_logs["L_grip"],
            "weight_pos": torch.exp(-action_logs["log_sigma2_pos"]),
            "weight_rot": torch.exp(-action_logs["log_sigma2_rot"]),
            "weight_grip": torch.exp(-action_logs["log_sigma2_grip"]),
            "code_usage": (counts > 0).float().mean(),
            "code_perplexity": code_perplexity,
        }
        return outputs, loss, logs

    def _log_step(self, prefix: str, loss: Tensor, logs: Dict, **kwargs):
        self.log_dict(
            {f"{prefix}/loss": loss, **{f"{prefix}/{k}": v for k, v in logs.items()}},
            prog_bar=True,
            sync_dist=True,
            **kwargs,
        )

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        _, loss, logs = self.shared_step(batch)
        self._log_step("train", loss, logs, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        _, loss, logs = self.shared_step(batch)
        self._log_step("val", loss, logs, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.lam.vq.random_restart(epoch=self.current_epoch)
        self.lam.vq.reset_usage()

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters())


def load_lam(path: str, device: Optional[str] = None) -> DINOLatentActionModel:
    """Load a frozen :class:`DINOLatentActionModel` from a stage-1 checkpoint.

    Accepts both full Lightning checkpoints and the slim ``lam.ckpt`` released on the Hub
    (``{"hyper_parameters": ..., "state_dict": ...}``).  The DINOv3 weights are part of the
    checkpoint, so the backbone is not downloaded again.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    module = DINO_LAM(
        lam_model_dim=hp["lam_model_dim"],
        lam_latent_dim=hp["lam_latent_dim"],
        lam_num_latents=hp["lam_num_latents"],
        lam_num_codes=hp["lam_num_codes"],
        lam_enc_blocks=hp["lam_enc_blocks"],
        lam_dec_blocks=hp["lam_dec_blocks"],
        lam_num_heads=hp["lam_num_heads"],
        pretrained_backbone=False,
    )
    missing, unexpected = module.load_state_dict(ckpt["state_dict"], strict=False)
    missing = [k for k in missing if k != "action_scale"]
    if missing or unexpected:
        raise RuntimeError(f"LAM checkpoint mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    lam = module.lam.eval().requires_grad_(False)
    return lam.to(device) if device is not None else lam
