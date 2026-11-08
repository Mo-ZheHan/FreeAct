"""Maps frame pairs to ``<latent_action_i>`` tokens of the VLA vocabulary via a frozen LAM."""

import os
from contextlib import contextmanager
from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from freeact.tokenizers import load_tokenizer


class LatentActionTokenizer:
    """Adds ``num_latents`` latent action tokens to the text vocabulary and encodes frame pairs."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        checkpoint_path: str,
        device: str = None,
    ):
        self.tokenizer = tokenizer
        if device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.lam = load_tokenizer(checkpoint_path, device=device)
        self.backend = self.lam.backend
        # num_latents = codebook size (vocabulary), num_codes = tokens per frame pair
        self.num_latents = self.lam.num_latents
        self.num_codes = self.lam.num_codes

        self._token_ids = self._setup_tokens()

    def _setup_tokens(self) -> List[int]:
        tokens = [f"<latent_action_{i}>" for i in range(self.num_latents)]
        self.tokenizer.add_tokens(tokens, special_tokens=True)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @property
    def latent_action_token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def placeholder_str(self) -> str:
        """Placeholder of ``num_codes`` tokens, replaced by real codes in the Trainer (batched on GPU)."""
        return "".join(f"<latent_action_{i}>" for i in range(self.num_codes))

    @contextmanager
    def _tf32(self):
        """TF32 matmuls for the baseline tokenizers only; restores the global flags afterwards.

        The FreeAct LAM runs in fp32; the UniVLA and LAPA baseline tokenizers use TF32 matmuls.
        """
        if self.backend == "freeact":
            yield
            return
        saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved

    @torch.no_grad()
    def encode_frames(self, frame1: torch.Tensor, frame2: torch.Tensor) -> np.ndarray:
        """Encode frame pairs ``[C,H,W]`` or ``[B,C,H,W]`` in ``[0, 1]`` into VQ indices."""
        squeeze = frame1.dim() == 3
        if squeeze:
            frame1, frame2 = frame1.unsqueeze(0), frame2.unsqueeze(0)
        frame1 = frame1.to(self.device, non_blocking=True)
        frame2 = frame2.to(self.device, non_blocking=True)
        with self._tf32():
            indices = self.lam.encode_pairs(frame1, frame2)
        return indices.squeeze(0).cpu().numpy() if squeeze else indices.cpu().numpy()

    def indices_to_ids(self, indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """Convert VQ indices to token IDs (supports batch input)."""
        return np.array(self._token_ids)[np.asarray(indices)]
