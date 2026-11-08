"""Per-component std of the camera-frame action targets, for ``action_std_*`` in ``configs/lam.yaml``.

    python scripts/compute_action_stats.py --config configs/lam.yaml [--num_samples 10000]

Only the calibrated datasets of the mixture are used (the action loss is gated on them).  The
values shipped in ``configs/lam.yaml`` were computed this way with delta_time = 1.0 s.
"""

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from freeact.data import MixedLeRobotDataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/lam.yaml")
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    data_cfg = yaml.safe_load(open(args.config))["data"]
    specs = [tuple(s) for s in data_cfg["dataset_mix"] if s[2]]
    ds = MixedLeRobotDataset(specs, root=data_cfg["root"], delta_time=data_cfg["delta_time"], action_mode="delta")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), min(args.num_samples, len(ds)), replace=False).tolist()
    loader = DataLoader(Subset(ds, idx), batch_size=1, num_workers=args.num_workers, collate_fn=lambda b: b[0])
    actions = np.stack([torch.as_tensor(s["action"]).numpy() for s in tqdm(loader)])

    std = actions.std(axis=0)
    print(f"{len(actions)} actions")
    print(f"action_std_pos: {std[0:3].mean()}")
    print(f"action_std_rot: {std[3:6].mean()}")
    print(f"action_std_grip: {std[6]}")


if __name__ == "__main__":
    main()
