"""Stage-1 training entry point (Lightning CLI).

    python -m freeact.lam.train fit --config configs/lam.yaml
"""

import torch
from lightning.pytorch.cli import LightningCLI

from freeact.lam import DINO_LAM, LightningLeRobotDataset


def main():
    torch.set_float32_matmul_precision("high")
    LightningCLI(
        DINO_LAM,
        LightningLeRobotDataset,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
