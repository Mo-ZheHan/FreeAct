"""Convert saved LIBERO-format rollouts (``freeact.eval.run --save_rollouts``) into a LeRobot dataset.

    python -m freeact.selfevolve.to_lerobot --rollouts a.hdf5 b.hdf5 ... --output data/rollouts/libero_target_A

The resulting dataset has the LIBERO schema (agent-view + wrist images, 8D state, 7D action) at
20 fps and can be passed to stage 3 like any other action-labeled dataset.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

LIBERO_FPS = 20
LIBERO_FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "rgb"]},
    "observation.images.wrist_image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "rgb"]},
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]},
    },
}


def load_episodes(path: Path):
    """Yield ``(instruction, frames)`` for every demo in a LIBERO-format HDF5 file."""
    with h5py.File(path, "r") as f:
        instruction = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
        for key in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1])):
            demo = f["data"][key]
            # Inverse of the env-side mapping a_libero = 1 - 2 * clip(a_vla, 0, 1)
            action = np.array(demo["actions"])
            action = np.concatenate([action[:, :6], np.clip((1.0 - action[:, -1:]) / 2.0, 0.0, 1.0)], axis=1)
            state = np.concatenate([np.array(demo["obs/ee_states"]), np.array(demo["obs/gripper_states"])], axis=1)
            episode = {
                "observation.images.image": np.array(demo["obs/agentview_rgb"]),
                "observation.images.wrist_image": np.array(demo["obs/eye_in_hand_rgb"]),
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
            }
            yield instruction, [{k: v[i] for k, v in episode.items()} for i in range(len(action))]


def convert(rollouts: list[Path], output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    dataset = LeRobotDataset.create(
        repo_id=f"{output.parent.name}/{output.name}",
        root=output,
        fps=LIBERO_FPS,
        robot_type="franka",
        features=LIBERO_FEATURES,
    )
    for path in tqdm(rollouts, desc="HDF5 -> LeRobot"):
        for instruction, frames in load_episodes(path):
            for frame in frames:
                dataset.add_frame({**frame, "task": instruction})
            dataset.save_episode()
    dataset.finalize()
    shutil.rmtree(output / "images", ignore_errors=True)  # temporary frames, already encoded into videos/
    logger.info("Wrote %d episodes to %s", dataset.meta.total_episodes, output)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if missing := [str(r) for r in args.rollouts if not r.is_file()]:
        p.error(f"rollout files not found: {', '.join(missing)}")
    convert(sorted(args.rollouts), args.output)


if __name__ == "__main__":
    main()
