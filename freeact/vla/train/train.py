"""FreeAct stages 2 and 3: latent action-guided VLA pretraining and deployment adaptation.

    stage 2 (``--vla_training_phase pretrain``): Qwen3-VL predicts the frozen LAM's latent action
        tokens for real robot frame pairs (cross-entropy only).
    stage 3 (``--vla_training_phase finetune``): a MAP action head decodes the hidden states of the
        latent tokens into a continuous action chunk.  Source data supervises tokens + actions,
        generated videos (``--video_dataset_use``) supervise tokens only, and the two gradients
        are reconciled by gradient projection (``--grad_proj``).

Training code adapted from the Qwen3-VL finetuning scripts (https://github.com/QwenLM/Qwen3-VL,
Apache-2.0), which retain the following notice of Stanford Alpaca (via FastChat):
    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
    Licensed under the Apache License, Version 2.0.
"""

import json
import os
from collections import deque
from contextlib import contextmanager, nullcontext
from datetime import datetime
from functools import partial
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from accelerate.utils import DistributedType
from torch.utils.data import DataLoader
from transformers import Trainer, set_seed
from transformers.trainer_utils import seed_worker

from freeact.data import DistributedWeightedSampler, print_sampling_summary
from freeact.vla.data import utils as data_utils
from freeact.vla.data.data_processor import IMAGE_TOKEN_INDEX, make_supervised_data_module
from freeact.vla.model_utils import load_model_and_processor, resize_embeddings_if_needed
from freeact.vla.train.action_decoder import VLAActionDecoder
from freeact.vla.train.argument import DataArguments, ModelArguments, TrainingArguments
try:
    from freeact.vla.train.packing import replace_qwen3_vl_attention_class
except ModuleNotFoundError as e:
    if e.name != "flash_attn":
        raise
    raise SystemExit("VLA training needs FlashAttention; install it with: uv sync --extra train") from None

LAM_FINGERPRINT = "lam_backend.json"


def _lam_fingerprint(train_dataset) -> Optional[Dict]:
    tok = getattr(train_dataset, "latent_action_tokenizer", None)
    if tok is None:
        return None
    return {"backend": tok.backend, "num_latents": tok.num_latents, "num_codes": tok.num_codes}


def _save_lam_fingerprint(train_dataset, output_dir: str) -> None:
    """Record which latent action model produced this checkpoint's vocabulary."""
    if (fp := _lam_fingerprint(train_dataset)) is None:
        return
    with open(os.path.join(output_dir, LAM_FINGERPRINT), "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2)


def _check_lam_matches_checkpoint(train_dataset, model_path) -> None:
    """Refuse to fine-tune a stage-2 checkpoint with a different LAM.

    Latent tokens are named ``<latent_action_i>`` regardless of which LAM emitted them, so
    pairing e.g. a UniVLA-pretrained backbone with the FreeAct tokenizer would load cleanly and
    train on silently mismatched targets.
    """
    if (fp := _lam_fingerprint(train_dataset)) is None or not model_path:
        return
    path = os.path.join(os.path.abspath(str(model_path)), LAM_FINGERPRINT)
    if not os.path.isfile(path):
        if os.path.isdir(str(model_path)):
            data_utils.rank0_print(
                f"[LAM] WARNING: no {LAM_FINGERPRINT} in {model_path}; cannot verify that it was "
                f"pretrained with the same latent action model as --lam_checkpoint ({fp})."
            )
        return
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    if saved != fp:
        raise ValueError(
            f"LAM mismatch: {path} records {saved}, but --lam_checkpoint gives {fp}. Point "
            f"--lam_checkpoint at the same latent action model as the stage-2 run."
        )
    data_utils.rank0_print(f"[LAM] fingerprint matches stage-2 checkpoint: {fp}")


def save_action_decoder(trainer, output_dir: str) -> None:
    decoder = trainer.action_decoder
    torch.save(decoder.state_dict(), os.path.join(output_dir, "action_decoder.pt"))
    config = {
        "embed_dim": decoder.latent_action_pool.projection.in_features,
        "hidden_dim": decoder.latent_action_pool.projection.out_features,
        "action_dim": decoder.action_dim,
        "window_size": decoder.window_size,
        "num_codes": decoder.num_codes,
    }
    with open(os.path.join(output_dir, "action_decoder_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def save_action_stats(train_dataset, output_dir: str) -> None:
    """Store the q01/q99 statistics used to normalize actions, needed to denormalize at test time."""
    stats = getattr(train_dataset, "action_stats", None)
    if not stats:
        return
    # All action datasets share one normalization; store it.
    q = next(iter(stats.values()))
    with open(os.path.join(output_dir, "action_stats.json"), "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in q.items()}, f, indent=2)


class VLATrainer(Trainer):
    """HF Trainer with weighted multi-dataset sampling, batched LAM tokenization, and (stage 3)
    the continuous action head with gradient projection between the source and generated streams."""

    def __init__(self, *args, action_decoder=None, training_phase="pretrain", **kwargs):
        super().__init__(*args, **kwargs)
        self.training_phase = training_phase
        self.action_decoder = action_decoder
        assert not (training_phase == "finetune" and action_decoder is None), "Finetune phase needs action_decoder"

        tok = self.train_dataset.latent_action_tokenizer
        self.latent_action_tokenizer = tok
        self._min_lam_id = min(tok.latent_action_token_ids)

        # Gradient projection requires non-reentrant checkpointing (autograd.grad + retain_graph)
        if self.args.grad_proj and self.args.gradient_checkpointing:
            kw = self.args.gradient_checkpointing_kwargs or {}
            kw["use_reentrant"] = False
            self.args.gradient_checkpointing_kwargs = kw

        n = self.args.gradient_accumulation_steps
        self._metric_bufs = {
            name: deque(maxlen=n)
            for name in (
                "latent_action_accuracy",
                "action_loss",
                "action_loss_first_step",
                "ce_src",
                "ce_gen",
                "grad_conflict_ratio",
            )
        }

    # ------------------------------------------------------------------ data

    def _get_train_sampler(self, train_dataset=None):
        train_dataset = train_dataset or self.train_dataset
        ds = train_dataset.lerobot_dataset
        sampler = DistributedWeightedSampler(
            dataset=train_dataset,
            dataset_lengths=ds.dataset_lengths,
            dataset_weights=ds.weights,
            shuffle=True,
            seed=self.args.seed,
        )
        if self.args.local_rank <= 0:
            print_sampling_summary(ds, sampler.samples_per_dataset)
            print(f"Train: {len(self.train_dataset):,} samples | Samples/epoch/GPU: ~{len(sampler):,}\n")
        return sampler

    def get_train_dataloader(self) -> DataLoader:
        """DataLoader with the pre-sharded weighted sampler (bypasses Accelerate's re-sharding)."""
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            worker_init_fn=partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        )
        dataloader._is_accelerate_prepared = True
        return dataloader

    def _encode_lam_batch(self, inputs):
        """Batched LAM inference: replace placeholder tokens with the real latent action codes."""
        tok = self.latent_action_tokenizer
        indices = tok.encode_frames(*inputs.pop("_lam_frames"))
        ids, labels = inputs["input_ids"], inputs["labels"]
        mask = torch.isin(ids, torch.tensor(tok.latent_action_token_ids, device=ids.device))
        real = torch.from_numpy(tok.indices_to_ids(indices)).to(dtype=ids.dtype, device=ids.device)
        ids[mask], labels[mask] = real.flatten(), real.flatten()

    # ------------------------------------------------------- checkpointing

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if not self.args.should_save:
            return
        ckpt_dir = os.path.join(self._get_output_dir(trial), f"checkpoint-{self.state.global_step}")
        if self.action_decoder is not None:
            save_action_decoder(self, ckpt_dir)
            save_action_stats(self.train_dataset, ckpt_dir)
        _save_lam_fingerprint(self.train_dataset, ckpt_dir)

    def restore_action_decoder(self, ckpt_dir):
        if self.action_decoder is None or not ckpt_dir:
            return
        path = os.path.join(str(ckpt_dir), "action_decoder.pt")
        if os.path.isfile(path):
            device = next(self.action_decoder.parameters()).device
            self.action_decoder.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            print(f"[action head] Restored from {path}")

    def create_optimizer(self):
        """Optimizer with the action head in its own param group (lr / weight decay).

        The head is temporarily frozen so that the default Trainer grouping skips it, then
        re-enabled and added as a separate group (DeepSpeed-compatible).
        """
        if self.optimizer is not None:
            return self.optimizer
        if self.action_decoder is not None:
            for p in self.action_decoder.parameters():
                p.requires_grad = False
        super().create_optimizer()
        if self.action_decoder is not None:
            for p in self.action_decoder.parameters():
                p.requires_grad = True
            self.optimizer.add_param_group(
                {
                    "params": list(self.action_decoder.parameters()),
                    "lr": self.args.action_decoder_lr,
                    "weight_decay": self.args.action_decoder_weight_decay,
                }
            )
        return self.optimizer

    # ---------------------------------------------------------------- loss

    @staticmethod
    def _fix_ds_output_hooks(outputs):
        """Remove cascading per-tensor 1/gas hooks from DeepSpeed ``engine.forward()``.

        DeepSpeed 0.18+ registers gradient scaling hooks on ALL output tensors (logits,
        hidden_states, ...), causing 1/gas^K scaling instead of 1/gas.  Keep only the hook on the
        scalar loss.
        """
        if not isinstance(outputs, dict):
            return
        for val in outputs.values():
            if not (isinstance(val, torch.Tensor) and hasattr(val, "_backward_hook_managers")):
                continue
            for mgr in val._backward_hook_managers:
                mgr.remove_hooks()
                loss = outputs.get("loss")
                if isinstance(loss, torch.Tensor) and getattr(loss, "grad_fn", None):
                    mgr.hook_handles.append(loss.register_hook(mgr._make_backward_hook(loss)))
            break

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._encode_lam_batch(inputs)

        # Gradient projection only applies to stage-3 batches that mix source and generated samples
        if self.args.grad_proj and self.training_phase == "finetune":
            vom = inputs["video_only_mask"]
            # All ranks must agree (different ranks see different data)
            flag = torch.tensor(
                [vom.any().item() and not vom.all().item()], dtype=torch.int32, device=self.args.device
            )
            if self.args.world_size > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                return self._grad_proj_training_step(model, inputs)

        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**inputs, output_hidden_states=(self.training_phase == "finetune"))
        self._fix_ds_output_hooks(outputs)
        self._update_lam_metrics(outputs.logits, labels)

        if self.training_phase == "pretrain":
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Stage 3 batches without gradient projection: L_CE + L_ACT (video-only samples masked out of L_ACT)
        action_loss, first_step_loss = self._compute_action_loss(outputs.hidden_states[-1], labels, inputs)
        self._metric_bufs["action_loss"].append(action_loss.item())
        self._metric_bufs["action_loss_first_step"].append(first_step_loss.item())
        total = action_loss + outputs.loss
        return (total, outputs) if return_outputs else total

    def _grad_proj_training_step(self, model, inputs):
        """Split the loss into the source stream (CE + action) and the generated stream (CE on
        video-only samples), project conflicting gradients, then backward."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]

        with self.compute_loss_context_manager():
            outputs = model(**inputs, output_hidden_states=True)
            self._fix_ds_output_hooks(outputs)
            ce_src, ce_gen = self._split_ce_loss(outputs.logits, labels, inputs)
            action_loss, first_step_loss = self._compute_action_loss(outputs.hidden_states[-1], labels, inputs)

        loss_src = ce_src + action_loss
        loss_gen = ce_gen

        self._metric_bufs["action_loss"].append(action_loss.detach().item())
        self._metric_bufs["action_loss_first_step"].append(first_step_loss.item())
        self._metric_bufs["ce_src"].append(ce_src.detach().item())
        self._metric_bufs["ce_gen"].append(ce_gen.detach().item())
        self._update_lam_metrics(outputs.logits, labels)
        reported_loss = (outputs.loss + action_loss).detach()
        # Free large tensors before backward
        del outputs, inputs, labels, ce_src, ce_gen, action_loss, first_step_loss

        self._grad_proj_backward(model, loss_src, loss_gen)
        return reported_loss

    def _split_ce_loss(self, logits, labels, inputs):
        """Token CE split by ``video_only_mask`` (handles packed sequences).

        Both losses always traverse the full graph so that all ranks fire identical NCCL
        collectives during ``autograd.grad``.
        """
        vom = inputs["video_only_mask"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        per_tok_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.shape)

        # Expand per-sample mask to per-token (packed: attention_mask holds cu_seqlens)
        attn_mask = inputs["attention_mask"]
        tok_vom = torch.zeros(labels.shape[1], dtype=torch.bool, device=vom.device)
        for k in range(vom.shape[0]):
            tok_vom[attn_mask[k] : attn_mask[k + 1]] = vom[k]
        tok_vom = tok_vom[1:].unsqueeze(0)

        valid = shift_labels != -100
        pm = (valid & ~tok_vom).float()
        vm = (valid & tok_vom).float()
        ce_src = (per_tok_ce * pm).sum() / pm.sum().clamp(min=1)
        ce_gen = (per_tok_ce * vm).sum() / vm.sum().clamp(min=1)
        return ce_src, ce_gen

    @contextmanager
    def _ds_hooks_suspended(self):
        """Neutralize DeepSpeed backward hooks so ``autograd.grad`` neither corrupts IPG buffers
        nor triggers a premature allreduce."""
        eng = self.accelerator.deepspeed_engine_wrapped.engine
        opt = eng.optimizer
        _noop = lambda *a, **kw: None
        saved = (opt.enter_backward, opt.queue_post_backward_callback, eng._scale_wrt_gas)
        opt.enter_backward = opt.queue_post_backward_callback = _noop
        eng._scale_wrt_gas = False
        try:
            yield
        finally:
            opt.enter_backward, opt.queue_post_backward_callback, eng._scale_wrt_gas = saved

    def _grad_proj_backward(self, model, loss_src, loss_gen):
        """Two-pass gradient projection (Eq. 6).

        Pass 1 computes ``g_gen`` with ``autograd.grad``; pass 2 runs the regular (DeepSpeed)
        backward of ``loss_src`` with per-parameter hooks that return
        ``g_src + g_gen - min(0, <g_src, g_gen> / ||g_src||^2) * g_src``.
        """
        is_ds = self.accelerator.distributed_type == DistributedType.DEEPSPEED
        params = [p for p in model.parameters() if p.requires_grad]

        with self._ds_hooks_suspended() if is_ds else nullcontext():
            if loss_gen.requires_grad:
                grads_gen = torch.autograd.grad(loss_gen, params, retain_graph=True, allow_unused=True)
            else:
                grads_gen = (None,) * len(params)

        scale = 1.0 / self.args.gradient_accumulation_steps
        n_total = 0
        n_conflict = torch.zeros((), device=self.args.device)
        hooks = []
        for p, gv in zip(params, grads_gen):

            def _hook(gp, _gv=gv, _nc=n_conflict):
                if _gv is None:
                    return gp * scale
                dot = (gp * _gv).sum()
                coeff = torch.clamp(dot / (gp.norm().square() + 1e-12), max=0.0)
                _nc.add_((dot < 0).float())  # in-place GPU accumulate, no sync
                return (gp + _gv - coeff * gp) * scale

            n_total += gv is not None
            hooks.append(p.register_hook(_hook))
        del grads_gen

        # scale_wrt_gas=False avoids DeepSpeed's cascading 1/gas^K; the hooks apply 1/gas.
        try:
            if is_ds:
                eng = self.accelerator.deepspeed_engine_wrapped.engine
                eng.set_gradient_accumulation_boundary(is_boundary=self.accelerator.sync_gradients)
                eng.optimizer.enter_backward()
                eng.optimizer.queue_post_backward_callback()
                eng.backward(loss_src, scale_wrt_gas=False)
            else:
                self.accelerator.backward(loss_src)
        finally:
            for h in hooks:
                h.remove()

        if n_total > 0:
            self._metric_bufs["grad_conflict_ratio"].append(n_conflict.item() / n_total)
        if is_ds and self.accelerator.sync_gradients:
            self.accelerator.deepspeed_engine_wrapped.engine.step()

    def _compute_action_loss(self, hidden_states, labels, inputs):
        """L1 loss between the decoded action chunk and the normalized ground truth (Eq. 4).

        Video-only samples are forwarded through the head too (keeps every parameter in the graph
        for DDP/ZeRO) but masked out of the loss.
        """
        target = inputs.get("continuous_action")
        if target is None:
            dummy = sum(p.sum() for p in self.action_decoder.parameters()) * 0.0
            return dummy, torch.tensor(0.0, device=dummy.device)

        hs, ids, labs = hidden_states[0], inputs["input_ids"][0], labels[0]
        cu = inputs["attention_mask"]  # packed: cumulative sequence lengths
        visual_embeds, action_embeds = [], []
        for i in range(cu.shape[0] - 1):
            s, e = cu[i].item(), cu[i + 1].item()
            visual_embeds.append(hs[s:e][ids[s:e] == IMAGE_TOKEN_INDEX])
            act_mask = labs[s:e] >= self._min_lam_id
            action_embeds.append(
                hs[s:e][act_mask] if act_mask.any() else hs[s:e][-self.latent_action_tokenizer.num_codes :]
            )

        dtype = next(self.action_decoder.parameters()).dtype
        pred = self.action_decoder(
            self._pad_and_stack(action_embeds).to(dtype), self._pad_and_stack(visual_embeds).to(dtype)
        )
        loss = F.l1_loss(pred, target.to(pred.device), reduction="none")

        mask = (~inputs["video_only_mask"]).float().to(loss.device)  # 1 = source, 0 = generated
        n = mask.sum().clamp(min=1)
        per_sample = loss.flatten(1).mean(1)
        first_step = loss[:, 0].mean(-1)
        return (per_sample * mask).sum() / n, ((first_step * mask).sum() / n).detach()

    @staticmethod
    def _pad_and_stack(embeds):
        max_len = max(e.shape[0] for e in embeds)
        return torch.stack([F.pad(e, (0, 0, 0, max_len - e.shape[0])) for e in embeds])

    # ------------------------------------------------------------- logging

    @torch.no_grad()
    def _update_lam_metrics(self, logits, labels):
        preds, targets = logits[:, :-1].argmax(-1), labels[:, 1:]
        mask = targets >= self._min_lam_id
        if mask.any():
            self._metric_bufs["latent_action_accuracy"].append(
                ((preds == targets) & mask).sum().float().item() / mask.sum().item()
            )

    def _gather_mean(self, buf) -> Optional[float]:
        t = torch.tensor(sum(buf) / len(buf), device=self.args.device)
        if self.args.local_rank != -1:
            t = self.accelerator.gather(t).mean()
        return t.item() if self.state.is_world_process_zero else None

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None):
        for name, buf in self._metric_bufs.items():
            if buf and (val := self._gather_mean(buf)) is not None:
                logs[name] = val
            buf.clear()
        super().log(logs, start_time)


def _apply_lora(model, training_args, adapter_dir: Optional[str]):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if adapter_dir:
        print(f"[LoRA] Loading adapter from: {adapter_dir}")
        return PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    for p in model.parameters():
        p.requires_grad = False
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # The new latent action tokens need trainable embeddings / output rows
        modules_to_save=["embed_tokens", "lm_head"],
    )
    lora_config.ensure_weight_tying = True
    return get_peft_model(model, lora_config)


def train(attn_implementation="flash_attention_2"):
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    data_utils.local_rank = training_args.local_rank
    training_phase = training_args.vla_training_phase

    # --resume_from_checkpoint <run>/checkpoint-N continues that run in its directory (set
    # WANDB_RUN_ID to continue the same wandb run); otherwise each run gets a timestamped directory.
    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt and not os.path.isdir(str(resume_ckpt)):
        raise FileNotFoundError(f"--resume_from_checkpoint {resume_ckpt} is not a directory")
    continue_run = bool(resume_ckpt)
    if continue_run:
        training_args.output_dir = os.path.dirname(os.path.abspath(resume_ckpt))
        if os.environ.get("WANDB_RUN_ID"):
            os.environ["WANDB_RESUME"] = "allow"
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(obj := [run_id], src=0)
            run_id = obj[0]
        training_args.output_dir = os.path.join(training_args.output_dir, run_id)
    os.makedirs(training_args.output_dir, exist_ok=True)

    model, processor = load_model_and_processor(
        model_args.model_name_or_path,
        device=None,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
        attn_implementation=attn_implementation,
    )
    print(f"Loaded model: {model_args.model_name_or_path} ({model.__class__.__name__})")

    replace_qwen3_vl_attention_class()  # varlen attention over packed samples
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Adds the latent action tokens to the vocabulary
    data_module = make_supervised_data_module(processor, data_args=data_args, training_phase=training_phase)
    train_dataset = data_module["train_dataset"]
    _check_lam_matches_checkpoint(train_dataset, model_args.model_name_or_path)
    resize_embeddings_if_needed(model, processor.tokenizer)

    adapter_dir = next(
        (
            str(c)
            for c in [resume_ckpt, model_args.model_name_or_path]
            if c and os.path.isfile(os.path.join(str(c), "adapter_config.json"))
        ),
        None,
    )
    model = _apply_lora(model, training_args, adapter_dir)

    action_decoder = None
    if training_phase == "finetune":
        action_decoder = VLAActionDecoder(
            embed_dim=model.config.text_config.hidden_size,
            hidden_dim=training_args.action_decoder_hidden_dim,
            action_dim=data_args.action_dim,
            window_size=train_dataset.action_window_size,
            num_codes=train_dataset.latent_action_tokenizer.num_codes,
        ).to(next(model.parameters()).device)
        # Registered on the model so that DeepSpeed ZeRO can map its parameter names
        model.action_decoder = action_decoder

    trainer = VLATrainer(
        model=model,
        processing_class=processor.tokenizer,
        args=training_args,
        action_decoder=action_decoder,
        training_phase=training_phase,
        **data_module,
    )

    ckpt = resume_ckpt if continue_run else None
    trainer.restore_action_decoder(ckpt)
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_state()

    model.config.use_cache = True
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if training_args.should_save:
        if action_decoder is not None:
            save_action_decoder(trainer, training_args.output_dir)
            save_action_stats(train_dataset, training_args.output_dir)
        _save_lam_fingerprint(train_dataset, training_args.output_dir)


if __name__ == "__main__":
    try:
        train()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
