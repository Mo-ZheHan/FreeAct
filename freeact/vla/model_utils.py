"""Qwen3-VL loading shared by training and inference."""

import json
import logging
import os

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration


def resolve_base_model(path: str) -> str:
    """If ``path`` is a LoRA adapter directory, return its base model id."""
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            if base := json.load(f).get("base_model_name_or_path"):
                return base
    return path


def load_model_and_processor(model_name: str, device="cuda", dtype=torch.bfloat16, **kw):
    """Load the Qwen3-VL base model and processor.

    When ``model_name`` is a FreeAct checkpoint (LoRA adapter), the base model is loaded and the
    checkpoint's tokenizer (with the latent action tokens) replaces the stock one.
    """
    base_model = resolve_base_model(model_name)
    logging.info("Loading Qwen3-VL: %s", base_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, dtype=dtype, device_map=device, **kw)
    processor = AutoProcessor.from_pretrained(base_model)
    if base_model != model_name and os.path.isfile(os.path.join(model_name, "tokenizer.json")):
        processor.tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, processor


def resize_embeddings_if_needed(model, tokenizer) -> bool:
    """Resize embeddings if the tokenizer vocabulary exceeds the model's reserved rows."""
    vocab_size = model.config.text_config.vocab_size
    if len(tokenizer) > vocab_size:
        logging.info("Resizing embeddings: %d -> %d", vocab_size, len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
        return True
    return False
