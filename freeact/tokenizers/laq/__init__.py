"""LAPA's Latent Action Quantization model; only ``inference`` is used here.

Vendored from https://github.com/LatentActionPretraining/LAPA (``laq/laq_model``, MIT license,
see LICENSE in this directory) so that the released ``laq_openx.pt`` can serve as a baseline
tokenizer without checking out the LAPA repository.  LAPA's LAQ builds on phenaki-pytorch
(https://github.com/lucidrains/phenaki-pytorch, MIT), and ``nsvq.py`` comes from NSVQ
(https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization, MIT); LICENSE lists all
copyright notices.
"""

from .latent_action_quantization import LatentActionQuantization

__all__ = ["LatentActionQuantization"]
