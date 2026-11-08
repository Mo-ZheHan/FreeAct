"""LeRobot-based datasets shared by the latent action model (stage 1) and the VLA (stages 2-3).

Every sample is a frame pair ``(o_t, o_{t+Δ})`` plus, for action-labeled data, the
camera-frame end-effector action between the two frames.
"""

import math
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2

# Fixed LIBERO agent-view camera rotation (quat wxyz -> scipy xyzw).
# LIBERO_R_CAM: world -> camera (training), LIBERO_R_CAM_INV: camera -> world (inference).
_LIBERO_QUAT = [
    0.3048497438430786,
    0.30484986305236816,
    0.6380177736282349,
    0.6380177736282349,
]
LIBERO_R_CAM = R.from_quat(_LIBERO_QUAT).inv()
LIBERO_R_CAM_INV = R.from_quat(_LIBERO_QUAT)


def libero_to_camera_frame(action: np.ndarray) -> np.ndarray:
    """Transform a 7D LIBERO action from the world frame to the camera frame (training)."""
    dp, dr, dg = action[:3], action[3:6], action[6]
    return np.concatenate([-LIBERO_R_CAM.apply(dp), -LIBERO_R_CAM.apply(dr), [dg]])


def libero_from_camera_frame(action: np.ndarray) -> np.ndarray:
    """Transform a 7D LIBERO action from the camera frame back to the world frame (inference)."""
    dp, dr, dg = action[:3], action[3:6], action[6]
    return np.concatenate([-LIBERO_R_CAM_INV.apply(dp), -LIBERO_R_CAM_INV.apply(dr), [dg]])


_DROID_IMAGE_KEYS = {
    "dual_cam": ["observation.images.exterior_1_left", "observation.images.exterior_2_left"],
    "dual": ["observation.images.exterior_1_left", "observation.images.exterior_2_left"],
    "single_cam": ["observation.images.exterior_0_left"],
    "single": ["observation.images.exterior_0_left"],
}
LIBERO_IMAGE_KEY = "observation.images.image"
VIDEO_IMAGE_KEY = "observation.images.video"


def get_image_keys(dataset_name: str, video_only: bool = False) -> List[str]:
    """Camera keys used for a dataset. Each key becomes its own sub-dataset."""
    if video_only:
        return [VIDEO_IMAGE_KEY]
    name = dataset_name.lower()
    if "droid" in name:
        return _DROID_IMAGE_KEYS[name.rstrip("/").split("/")[-1]]
    if "libero" in name:
        return [LIBERO_IMAGE_KEY]
    raise ValueError(
        f"Unknown dataset '{dataset_name}': names must contain 'droid' or 'libero', which selects the format "
        "(e.g. rollouts/libero_target_A for converted rollouts)"
    )


def get_action_key(dataset_name: str) -> str:
    """DROID stores its raw action under 'action.original', LIBERO under 'action'."""
    return "action.original" if "droid" in dataset_name.lower() else "action"


class DistributedWeightedSampler(DistributedSampler):
    """
    Distributed sampler with epoch-multiplier weights for multi-dataset mixing.

    shuffle=True: random sampling with interleaved distribution across ranks.
    shuffle=False: sequential sampling with contiguous chunks per rank.
    Weight: 1.0 = full pass, 2.0 = 2x samples, 0.5 = half samples per epoch.
    """

    def __init__(
        self,
        dataset: Dataset,
        dataset_lengths: List[int],
        dataset_weights: List[float],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)

        self._lengths = np.asarray(dataset_lengths, dtype=np.int64)
        self._offsets = np.zeros(len(dataset_lengths) + 1, dtype=np.int64)
        np.cumsum(self._lengths, out=self._offsets[1:])

        self._counts = np.array(
            [int(n * w) for n, w in zip(dataset_lengths, dataset_weights)],
            dtype=np.int64,
        )
        self.samples_per_dataset = self._counts.tolist()
        effective_total = int(self._counts.sum())

        if shuffle:
            self.num_samples = math.ceil(effective_total / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            q, r = divmod(effective_total, self.num_replicas)
            self._seq_start = self.rank * q + min(self.rank, r)
            self.num_samples = q + (1 if self.rank < r else 0)
            self.total_size = effective_total

    def _build_random_indices(self, rng: np.random.Generator) -> np.ndarray:
        """Build shuffled global indices array using numpy (memory-efficient)."""
        total = int(self._counts.sum())
        indices = np.empty(total, dtype=np.int64)
        ptr = 0

        for i, count in enumerate(self._counts):
            if count == 0:
                continue
            length, offset = int(self._lengths[i]), int(self._offsets[i])
            full, rem = divmod(int(count), length)

            for _ in range(full):
                indices[ptr : ptr + length] = offset + rng.permutation(length)
                ptr += length
            if rem:
                indices[ptr : ptr + rem] = offset + rng.permutation(length)[:rem]
                ptr += rem

        rng.shuffle(indices)

        if (pad := self.total_size - total) > 0:
            indices = np.concatenate([indices, rng.choice(indices, pad, replace=True)])

        return indices[self.rank :: self.num_replicas]

    def _iter_sequential(self, start: int, end: int) -> Iterator[int]:
        """Yield indices for [start, end) with per-epoch rotation for full coverage."""
        pos = 0
        for i, count in enumerate(self._counts):
            count = int(count)
            if count == 0:
                continue
            if pos >= end:
                break
            if pos + count <= start:
                pos += count
                continue

            length, offset = int(self._lengths[i]), int(self._offsets[i])
            shift = (self.epoch * count) % length
            for k in range(max(start, pos), min(end, pos + count)):
                yield offset + (k - pos + shift) % length
            pos += count

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            rng = np.random.Generator(np.random.PCG64(self.seed + self.epoch))
            return iter(self._build_random_indices(rng).tolist())

        rot = (self.epoch * (self.total_size // self.num_replicas)) % self.total_size
        start = (self._seq_start + rot) % self.total_size
        end = start + self.num_samples

        if end <= self.total_size:
            return self._iter_sequential(start, end)
        return chain(
            self._iter_sequential(start, self.total_size),
            self._iter_sequential(0, end - self.total_size),
        )

    def __len__(self) -> int:
        return self.num_samples


def print_sampling_summary(dataset: "MixedLeRobotDataset", samples_per_dataset: List[int], width: int = 85) -> None:
    """Print the per-dataset size / weight / samples-per-epoch table of a weighted mixture."""
    total = sum(samples_per_dataset)
    print(f"\n{'=' * width}\nDataset Sampling Summary\n{'=' * width}")
    print(f"{'Dataset':<35} {'Samples':>12} {'Weight':>8} {'Sampled':>12} {'Ratio':>10}")
    print("-" * width)
    for repo, n, w, s in zip(dataset.repo_ids, dataset.dataset_lengths, dataset.weights, samples_per_dataset):
        name = repo[:32] + "..." if len(repo) > 35 else repo
        ratio = s / total * 100 if total > 0 else 0
        print(f"{name:<35} {n:>12,} {w:>8.2f} {s:>12,} {ratio:>9.2f}%")
    print(f"{'-' * width}\n{'TOTAL':<35} {sum(dataset.dataset_lengths):>12,} {'':<8} {total:>12,} {'100.00%':>10}")
    print("=" * width)


class MixedLeRobotDataset(Dataset):
    """
    Mixture of LeRobot datasets that yields frame pairs separated by ``delta_time`` seconds.

    Each ``dataset_spec`` is ``(repo_id, weight, cam_calib)``; ``cam_calib`` marks datasets whose
    actions can be expressed in the camera frame (and therefore supervise the LAM's Action Decoder).
    ``video_only_specs`` are ``(repo_id, weight, root)`` triples for action-free (e.g. generated)
    videos, used only for latent-token supervision in stage 3.

    action_mode:
        "delta":    7D displacement between the two frames (stage 1)
        "sequence": the per-step actions over ``delta_time`` (stage 3 action chunk)
    """

    # Filter thresholds for static frames (no visible motion)
    MIN_CHANGE_RATIO = 0.001  # Min fraction of significantly changed pixels (0.1%)
    PIXEL_DIFF_TOLERANCE = 0.05  # Min per-pixel diff to count as "changed" (5%)
    MAX_RESAMPLE_ATTEMPTS = 1024  # Max retries when sample is invalid

    def __init__(
        self,
        dataset_specs: List[Tuple[str, float, bool]],
        resolution: int = 224,
        image_aug: bool = False,
        action_mode: str = "delta",
        seed: int = 42,
        root: Optional[str] = None,
        filter_identical_frames: bool = True,
        delta_time: float = 1.0,
        skip_action: bool = False,
        video_only_specs: Optional[List[Tuple[str, float, str]]] = None,
    ):
        super().__init__()
        assert action_mode in ("delta", "sequence"), action_mode
        self.dataset_specs = dataset_specs
        self.resolution = resolution
        self.action_mode = action_mode
        self.seed = seed
        self.root = root
        self.filter_identical_frames = filter_identical_frames
        self.delta_time = delta_time
        self.skip_action = skip_action

        # Expand dataset specs: each (repo_id, weight, cam_calib) with N image keys
        # becomes N entries (one per image key), preserving original weight and cam_calib
        self.repo_ids: List[str] = []
        self.weights: List[float] = []
        self.image_keys: List[str] = []
        self.cam_calib_flags: List[bool] = []
        self.video_only_flags: List[bool] = []
        self._per_dataset_root: List[Optional[str]] = []

        # Normalize all specs to (repo_id, weight, cam_calib, root_override, video_only)
        unified = [(r, w, c, None, False) for r, w, c in dataset_specs]
        for r, w, vroot in video_only_specs or []:
            unified.append((r, w, False, vroot, True))

        for repo_id, weight, cam_calib, root_ovr, video_only in unified:
            for key in get_image_keys(repo_id, video_only):
                self.repo_ids.append(repo_id)
                self.weights.append(weight)
                self.image_keys.append(key)
                self.cam_calib_flags.append(cam_calib)
                self.video_only_flags.append(video_only)
                self._per_dataset_root.append(root_ovr)

        if image_aug:
            self.image_aug = v2.Compose(
                [
                    v2.RandomResizedCrop(
                        size=resolution, scale=(0.9, 0.9), ratio=(1.0, 1.0)
                    ),
                    v2.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    ),
                ]
            )
        else:
            self.image_aug = v2.Compose(
                [
                    v2.Resize(size=resolution, antialias=True),
                    v2.CenterCrop(size=resolution),
                ]
            )

        # Cache (root, repo_id) -> (dataset, valid_indices) to avoid duplicate loading
        self._cache: Dict[str, Tuple[LeRobotDataset, np.ndarray]] = {}

        self.datasets: List[LeRobotDataset] = []
        self.dataset_lengths: List[int] = []
        self._valid_indices: List[np.ndarray] = []
        self.cumulative_lengths: List[int] = [0]

        print(f"Initializing mixed dataset: {len(self.repo_ids)} camera streams from {len(set(self.repo_ids))} datasets...")
        for i, (repo_id, image_key) in enumerate(zip(self.repo_ids, self.image_keys)):
            effective_root = self._per_dataset_root[i] or self.root
            dataset, valid_indices = self._get_cached(
                repo_id, effective_root, self.video_only_flags[i]
            )

            self.datasets.append(dataset)
            self._valid_indices.append(valid_indices)
            self.dataset_lengths.append(len(valid_indices))
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(valid_indices)
            )

            vo_tag = " [video-only]" if self.video_only_flags[i] else ""
            print(
                f"  - {repo_id} [{image_key.split('.')[-1]}]{vo_tag}: {len(valid_indices):,} valid / "
                f"{len(dataset):,} total, weight={self.weights[i]:.2f}, Δt={self.delta_time:.1f}s"
            )

        self._total_length = self.cumulative_lengths[-1]
        print(f"Total valid samples: {self._total_length:,}")

    def _get_cached(
        self, repo_id: str, root: Optional[str], video_only: bool
    ) -> Tuple[LeRobotDataset, np.ndarray]:
        """Get or create cached (dataset, valid_indices) for repo_id."""
        cache_key = f"{root}:{repo_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        dt = self.delta_time
        delta_timestamps = {k: [0.0, dt] for k in get_image_keys(repo_id, video_only)}

        # Delta mode needs state pairs for displacement computation
        if self.action_mode == "delta" and not video_only:
            if "droid" in repo_id.lower():
                delta_timestamps["observation.state.cartesian_position"] = [0.0, dt]
                delta_timestamps["observation.state.gripper_position"] = [0.0, dt]
            elif "libero" in repo_id.lower():
                delta_timestamps["observation.state"] = [0.0, dt]

        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=Path(root).expanduser() / repo_id if root else None,
            delta_timestamps=delta_timestamps,
            image_transforms=self.image_aug,
            revision="local",
            video_backend="pyav",
        )
        valid_indices = self._compute_valid_indices(dataset, dt)
        self._cache[cache_key] = (dataset, valid_indices)
        return dataset, valid_indices

    def _compute_valid_indices(
        self, dataset: LeRobotDataset, delta_time: float
    ) -> np.ndarray:
        """
        Compute valid frame indices by excluding the last N frames of each episode,
        where N = ceil(delta_time * fps). These tail frames cannot form valid pairs.
        Returns a sorted NumPy array of valid global indices.
        """
        fps = dataset.meta.fps
        exclude = math.ceil(delta_time * fps)

        episodes = dataset.meta.episodes
        ep_from = np.array(episodes["dataset_from_index"])
        ep_to = np.array(episodes["dataset_to_index"])

        # Valid range for episode is [from, to - exclude)
        valid_ends = np.maximum(ep_from, ep_to - exclude)

        ranges = [
            np.arange(s, e, dtype=np.int64)
            for s, e in zip(ep_from, valid_ends)
            if e > s
        ]
        return np.concatenate(ranges) if ranges else np.array([], dtype=np.int64)

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample, resampling up to MAX_RESAMPLE_ATTEMPTS times on failure."""
        for i in range(self.MAX_RESAMPLE_ATTEMPTS + 1):
            target = idx if i == 0 else np.random.randint(0, self._total_length)
            if (result := self._try_get_sample(target)) is not None:
                return result
        raise RuntimeError(
            f"Sample {idx}: failed after {self.MAX_RESAMPLE_ATTEMPTS} retries"
        )

    def _try_get_sample(self, idx: int) -> Optional[Dict[str, Any]]:
        """Attempt to get a sample by global index. Returns None on failure."""
        # Binary search to find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1

        # Map to actual dataset index via valid_indices
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        actual_idx = int(self._valid_indices[dataset_idx][local_idx])

        dataset = self.datasets[dataset_idx]
        dataset_name = self.repo_ids[dataset_idx]
        image_key = self.image_keys[dataset_idx]

        try:
            sample = dataset[actual_idx]
        except Exception as e:
            print(f"[Warning] Failed to load sample {idx} from {dataset_name}: {e}")
            return None

        # Skip padded samples
        padding_key = f"{image_key}_is_pad"
        if padding_key in sample and sample[padding_key].any():
            return None

        return self._process_sample(
            sample,
            dataset_name,
            image_key,
            self.cam_calib_flags[dataset_idx],
            dataset,
            actual_idx,
            video_only=self.video_only_flags[dataset_idx],
        )

    def _process_sample(
        self,
        sample: Dict[str, Any],
        dataset_name: str,
        image_key: str,
        cam_calib: bool,
        dataset: LeRobotDataset,
        actual_idx: int,
        video_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Process a sample into training format."""
        img0, img1 = sample[image_key]  # [T=2, C, H, W]
        if self.filter_identical_frames and not self._has_visible_motion(img0, img1):
            return None

        if img0.dtype == torch.uint8:
            img0, img1 = img0.float() / 255.0, img1.float() / 255.0

        result = {
            "initial_pixel_values": img0,
            "target_pixel_values": img1,
            "cam_calib": cam_calib,
            "task_instruction": sample.get("task") or "",
            "dataset_name": dataset_name,
            "video_only": video_only,
        }

        # Stage-2 pretraining and generated videos need no action labels
        if self.skip_action or video_only:
            return result

        if self.action_mode == "sequence":
            action = self._get_action_sequence(dataset, actual_idx, dataset_name)
        elif "droid" in dataset_name.lower():
            action = self._compute_droid_action(sample)
        elif "libero" in dataset_name.lower():
            action = self._compute_libero_action(sample)
        else:
            raise ValueError(f"Unknown dataset for delta mode: {dataset_name}")

        if cam_calib:
            transform = lambda a: self._to_camera_frame(a, dataset_name, sample, image_key)
            action = (
                np.stack([transform(a) for a in action])
                if self.action_mode == "sequence"
                else transform(action)
            )

        result["action"] = action
        return result

    def _has_visible_motion(self, frame1: torch.Tensor, frame2: torch.Tensor) -> bool:
        """Check if there is visible motion between two frames."""
        diff = (frame1 - frame2).abs()
        change_ratio = (diff > self.PIXEL_DIFF_TOLERANCE).float().mean().item()
        return change_ratio >= self.MIN_CHANGE_RATIO

    def _to_camera_frame(
        self,
        action: np.ndarray,
        dataset_name: str,
        sample: Dict[str, Any],
        image_key: str,
    ) -> np.ndarray:
        """Transform 7D action [dp, dr, gripper] from base/world to camera frame."""
        dp, dr, dg = action[:3], action[3:6], action[6]

        if "droid" in dataset_name.lower():
            ext = sample[f"camera_extrinsics.{image_key.split('.')[-1]}"].cpu().numpy()
            rot = R.from_euler("xyz", ext[3:6]).inv()
            return np.concatenate([rot.apply(dp), rot.apply(dr), [dg]])

        if "libero" in dataset_name.lower():
            return libero_to_camera_frame(action)

        raise ValueError(f"Unknown dataset for camera transform: {dataset_name}")

    def _get_action_sequence(
        self, dataset: LeRobotDataset, start_idx: int, dataset_name: str
    ) -> np.ndarray:
        """Per-step raw actions over ``delta_time``: ``[ceil(delta_time * fps), 7]``."""
        num_frames = math.ceil(self.delta_time * dataset.meta.fps)
        action_key = get_action_key(dataset_name)

        actions = []
        for i in range(num_frames):
            action = dataset.hf_dataset[start_idx + i][action_key]
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()
            actions.append(action)
        return np.stack(actions)

    def _compute_droid_action(self, sample: Dict[str, Any]) -> np.ndarray:
        """Compute 7D action [dp, dr, gripper] for DROID in base frame."""
        cart = sample["observation.state.cartesian_position"].cpu().numpy()  # [2, 6]
        grip = sample["observation.state.gripper_position"].cpu().numpy()  # [2]

        dp = np.diff(cart[:, :3], axis=0)[0]
        dr = R.from_euler("xyz", cart[1, 3:6]) * R.from_euler("xyz", cart[0, 3:6]).inv()
        dg = np.diff(grip)[0]

        return np.concatenate([dp, dr.as_rotvec(), [dg]])

    def _compute_libero_action(self, sample: Dict[str, Any]) -> np.ndarray:
        """Compute 7D action [dp, dr, gripper] for LIBERO in world frame."""
        s = sample["observation.state"].cpu().numpy()  # [2, 8]

        dp = np.diff(s[:, :3], axis=0)[0]
        dr = (R.from_rotvec(s[1, 3:6]) * R.from_rotvec(s[0, 3:6]).inv()).as_rotvec()
        dg = ((s[1, -1] - s[1, -2]) - (s[0, -1] - s[0, -2])) * 12.5  # change in finger gap (up to 0.08 m) scaled to about [-1, 1]

        return np.concatenate([dp, dr, [dg]])


class CollatorForLatentAction:
    """Collator for batching frame pairs for latent action model training."""

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "videos": torch.stack(
                [
                    torch.stack([x["initial_pixel_values"], x["target_pixel_values"]])
                    for x in instances
                ]
            ),
            "action": torch.stack([torch.from_numpy(x["action"]) for x in instances]),
            "cam_calib_mask": torch.tensor(
                [x["cam_calib"] for x in instances], dtype=torch.bool
            ),
            "dataset_names": [x["dataset_name"] for x in instances],
        }
