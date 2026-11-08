"""Score rollout videos with a multimodal LLM and select the best ones per task (RQ3).

    export OPENAI_API_KEY=...
    python -m freeact.selfevolve.judge --run_dir outputs/eval/<run> --top_k 5

Reads ``<run_dir>/rollouts/*/*.hdf5`` (saved with ``--save_rollouts``), renders each rollout's
agent-view frames, asks the judge for the five-dimension score of Appendix C, and writes
``scores.json`` plus ``selected.json`` (top-k per task).

Frames are subsampled to ``--fps`` (at most ``--max_frames``, spread over the whole rollout) and sent as images through the OpenAI-compatible chat API
(``--model``, ``--base_url``).
"""

import argparse
import base64
import io
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text()
SOURCE_FPS = 20


def _frames(path: Path, fps: float, max_frames: int) -> tuple[str, list[str]]:
    with h5py.File(path, "r") as f:
        instruction = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
        rgb = np.array(f["data/demo_0/obs/agentview_rgb"])
    idx = np.arange(0, len(rgb), max(1, round(SOURCE_FPS / fps)))
    if len(idx) > max_frames:  # keep the end of long rollouts
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]
    encoded = []
    for i in idx:
        buf = io.BytesIO()
        Image.fromarray(rgb[i]).save(buf, format="JPEG", quality=90)
        encoded.append(base64.b64encode(buf.getvalue()).decode())
    return instruction, encoded


def score_rollout(client, model: str, path: Path, fps: float, max_frames: int) -> dict:
    instruction, frames = _frames(path, fps, max_frames)
    content = [{"type": "text", "text": PROMPT.replace("<task objective>", f'task "{instruction}"')}]
    content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in frames]
    reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}])
    text = reply.choices[0].message.content
    m = re.search(r"Total Score:\s*(\d+(?:\.\d+)?)\s*/\s*20", text)
    return {"instruction": instruction, "score": float(m.group(1)) if m else None, "response": text}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--base_url", default=None)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--max_frames", type=int, default=64)
    args = p.parse_args()
    if not (args.run_dir / "rollouts").is_dir():
        p.error(f"{args.run_dir / 'rollouts'} not found; collect rollouts with scripts/eval.sh ... --save_rollouts")
    if not os.environ.get("OPENAI_API_KEY"):
        p.error("set OPENAI_API_KEY for the OpenAI-compatible API (see --base_url)")

    try:
        from openai import OpenAI
    except ImportError:
        p.error("the judge needs the openai package; install it with uv sync --extra judge")

    client = OpenAI(base_url=args.base_url)
    success = {}
    if (res := args.run_dir / "results.json").exists():
        for e in json.loads(res.read_text())["episodes"]:
            success[f"{e['task_group']}/task{e['task_id']}_ep{e['episode']}.hdf5"] = e["success"]

    scores_path = args.run_dir / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    for path in sorted((args.run_dir / "rollouts").glob("*/*.hdf5")):
        key = f"{path.parent.name}/{path.name}"
        if key in scores and scores[key]["score"] is not None:
            continue  # resumable
        scores[key] = score_rollout(client, args.model, path, args.fps, args.max_frames)
        print(f"{key}: {scores[key]['score']}")
        scores_path.write_text(json.dumps(scores, indent=2))

    by_task = defaultdict(list)
    for key, s in scores.items():
        if s["score"] is not None:
            by_task[key.rsplit("_ep", 1)[0]].append((s["score"], key))
    selected = {task: [k for _, k in sorted(v, reverse=True)[: args.top_k]] for task, v in sorted(by_task.items())}
    (args.run_dir / "selected.json").write_text(json.dumps(selected, indent=2))

    for task, keys in selected.items():
        hits = [success.get(k) for k in keys]
        print(f"{task}: selected {len(keys)}, successful {sum(bool(h) for h in hits)}/{len(keys)}")
    print(f"-> {args.run_dir / 'selected.json'}")


if __name__ == "__main__":
    main()
