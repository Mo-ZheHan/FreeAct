"""Lightning DataModule for stage-1 latent action model training."""

import hashlib
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from ..data import (
    CollatorForLatentAction,
    DistributedWeightedSampler,
    MixedLeRobotDataset,
    print_sampling_summary,
)


class LightningLeRobotDataset(LightningDataModule):
    """Lightning DataModule for training with mixed LeRobot datasets."""

    def __init__(
        self,
        dataset_mix: List[Tuple[str, float, bool]],
        batch_size: int = 64,
        resolution: int = 224,
        image_aug: bool = True,
        num_workers: int = 4,
        seed: int = 42,
        root: Optional[str] = None,
        val_split: float = 0.0,
        delta_time: float = 1.0,
        shuffle: bool = True,
    ):
        super().__init__()
        self.dataset_mix = dataset_mix
        self.batch_size = batch_size
        self.resolution = resolution
        self.image_aug = image_aug
        self.num_workers = num_workers
        self.seed = seed
        self.root = root
        self.val_split = val_split
        self.delta_time = delta_time
        self.shuffle = shuffle
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.collate_fn = CollatorForLatentAction()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        """Setup datasets for training and optional validation."""
        if stage != "fit":
            return

        self._create_datasets()

        self.train_sampler = DistributedWeightedSampler(
            dataset=self.train_dataset,
            dataset_lengths=self._train_lens,
            dataset_weights=self._full_dataset.weights,
            shuffle=self.shuffle,
            seed=self.seed,
        )
        self._print_sampling_summary()

    def _create_datasets(self) -> None:
        """Create full dataset and train/val splits (without distributed sampler)."""
        self._full_dataset = MixedLeRobotDataset(
            dataset_specs=self.dataset_mix,
            resolution=self.resolution,
            image_aug=self.image_aug,
            action_mode="delta",
            seed=self.seed,
            root=self.root,
            delta_time=self.delta_time,
        )
        ds = self._full_dataset

        if self.val_split > 0:
            train_idx, val_idx, self._train_lens = self._stratified_split(
                ds, self.val_split
            )
            self.train_dataset = Subset(ds, train_idx)
            self.val_dataset = Subset(ds, val_idx)
        else:
            self.train_dataset = ds
            self.val_dataset = None
            self._train_lens = ds.dataset_lengths

    def get_split_datasets(
        self,
    ) -> Tuple[Subset, Optional[Subset], "MixedLeRobotDataset"]:
        """
        Get train/val splits without creating distributed sampler.
        Useful for inference where distributed training is not needed.

        Returns:
            (train_dataset, val_dataset, full_dataset)
        """
        self._create_datasets()
        return self.train_dataset, self.val_dataset, self._full_dataset

    def _stratified_split(
        self, dataset: MixedLeRobotDataset, val_ratio: float
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Stratified split by episode. Returns indices into the valid index space.
        Same repo_id shares splits to prevent data leakage.
        """
        train_idx, val_idx, train_lens = [], [], []
        episode_perms: Dict[str, np.ndarray] = {}

        def get_episode_perm(repo_id: str, n_episodes: int) -> np.ndarray:
            if repo_id not in episode_perms:
                h = int(hashlib.md5(repo_id.encode()).hexdigest(), 16) % (2**31)
                g = torch.Generator().manual_seed(self.seed + h)
                episode_perms[repo_id] = torch.randperm(n_episodes, generator=g).numpy()
            return episode_perms[repo_id]

        for i, ds in enumerate(dataset.datasets):
            repo_id = dataset.repo_ids[i]
            global_offset = dataset.cumulative_lengths[i]

            # Recompute valid lengths to map episodes to global indices
            fps = ds.meta.fps
            exclude = math.ceil(dataset.delta_time * fps)

            episodes = ds.meta.episodes
            ep_from = np.array(episodes["dataset_from_index"])
            ep_to = np.array(episodes["dataset_to_index"])

            valid_lens = np.maximum(0, ep_to - exclude - ep_from)
            ep_starts = np.cumsum(np.concatenate(([0], valid_lens[:-1])))

            perm = get_episode_perm(repo_id, len(ep_from))
            n_val = max(1, int(len(perm) * val_ratio))

            def append_indices(ep_list: np.ndarray, target_list: List[int]) -> int:
                count = 0
                for ep in ep_list:
                    length = valid_lens[ep]
                    if length > 0:
                        start = global_offset + ep_starts[ep]
                        target_list.extend(range(start, start + length))
                        count += length
                return count

            append_indices(perm[:n_val], val_idx)

            train_count = append_indices(perm[n_val:], train_idx)
            train_lens.append(train_count)

        return train_idx, val_idx, train_lens

    def _print_sampling_summary(self) -> None:
        print_sampling_summary(self._full_dataset, self.train_sampler.samples_per_dataset)
        n_train = len(self.train_dataset)
        n_val = len(self.val_dataset) if self.val_dataset else 0
        print(
            f"Train: {n_train:,} samples | Val: {n_val:,} samples ({n_val/(n_train+n_val)*100:.1f}%)"
            if n_val
            else f"Train: {n_train:,} samples"
        )
        print(
            f"Samples per epoch: ~{len(self.train_sampler):,}/GPU (±1) | Workers: {self.num_workers}\n"
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with weighted sampling and multi-worker support."""
        if self.train_dataset is None:
            raise RuntimeError(
                "Training dataset not initialized. Call setup('fit') first."
            )

        def worker_init_fn(worker_id: int):
            rank = self.train_sampler.rank
            worker_seed = self.seed + worker_id + rank * self.num_workers
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation DataLoader if val_split > 0."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.val_dataset, shuffle=False),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
