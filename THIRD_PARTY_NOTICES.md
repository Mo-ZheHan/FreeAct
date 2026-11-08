# Third-party notices

FreeAct is released under the [Apache-2.0 License](LICENSE). Parts of it are adapted from other open-source projects, and it uses third-party models and data that are not covered by that license. The license terms of each original work continue to apply to the corresponding parts. Every adapted file names its source in its header.

## Code adapted in this repository

| Source | License | Files |
|---|---|---|
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) finetuning code | Apache-2.0 | `freeact/vla/train/train.py`, `argument.py`, `packing.py`; `freeact/vla/data/data_processor.py`, `rope2d.py` |
| [UniVLA](https://github.com/OpenDriveLab/UniVLA) | Apache-2.0 | `freeact/lam/modules/blocks.py`, `freeact/tokenizers/univla.py`, `freeact/vla/train/action_decoder.py` |
| [LAPA](https://github.com/LatentActionPretraining/LAPA), building on [phenaki-pytorch](https://github.com/lucidrains/phenaki-pytorch) and [NSVQ](https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization) | MIT | `freeact/tokenizers/laq/` (license and copyright notices in [`freeact/tokenizers/laq/LICENSE`](freeact/tokenizers/laq/LICENSE)) |

Copyright notices retained from the adapted code:

- Qwen3-VL finetuning code, from Stanford Alpaca via FastChat: Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li (Apache-2.0).
- LAPA: Copyright (c) 2024 Seonghyeon Ye and Joel Jang (MIT).
- phenaki-pytorch: Copyright (c) 2022 Phil Wang (MIT).
- NSVQ: Copyright (c) 2022-present Mohammad Hassan Vali (MIT).

## Models

- **DINOv3 ViT-B/16** (`vit_base_patch16_dinov3.lvd1689m`, loaded through [timm](https://github.com/huggingface/pytorch-image-models)) is the frozen backbone of the latent action model. Stage-1 training downloads it, and the released `lam/lam.ckpt` contains it. The weights are released by Meta under the [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), whose terms apply to their use and redistribution.
- **Qwen3-VL-4B-Instruct** ([Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), Apache-2.0) is the base model of the policy and is downloaded at runtime.
- **Baseline tokenizers**, downloaded by the user: UniVLA [`lam-stage-2.ckpt`](https://huggingface.co/qwbu/univla-latent-action-model) (Apache-2.0), whose backbone is [DINOv2](https://github.com/facebookresearch/dinov2) (Apache-2.0), and LAPA [`laq_openx.pt`](https://huggingface.co/latent-action-pretraining/LAPA-7B-openx) (MIT).

## Simulation, libraries and data

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT) and [robosuite](https://github.com/ARISE-Initiative/robosuite) (MIT) provide the simulated environments. `freeact/envs` patches LIBERO's `libero_spatial` scenes at runtime and adds its own initial states and texture.
- [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0) provides the dataset format, data loading and the LIBERO environment wrapper.
- [DROID](https://droid-dataset.github.io/) (CC BY 4.0) and the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) demonstrations are used for training. The LIBERO demonstrations are OpenVLA's regeneration ([openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds), MIT) and, for LIBERO-90, [IPEC-COMMUNITY/libero_90_no_noops_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_90_no_noops_lerobot) (Apache-2.0). The released conversions and their licenses are described on the [data cards](https://huggingface.co/datasets/Mo-ZheHan/FreeAct-Data).
