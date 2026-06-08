#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Wan-Trainer maze WebDataset JSON sidecars to verl-omni parquet.

The Wan-Trainer RL shards store precomputed latents plus one JSON sidecar per
sample. verl-omni online DanceGRPO samples from prompt parquet and computes
rewards from generated pixels, so this script keeps the prompt and the minimal
maze metadata needed by the pixel tracker reward.
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json_sidecars(webdataset_dir: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tar_paths = sorted(webdataset_dir.glob("shard-*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No shard-*.tar files found under {webdataset_dir}")

    for tar_path in tar_paths:
        with tarfile.open(tar_path) as tar:
            members = sorted((m for m in tar.getmembers() if m.name.endswith(".json")), key=lambda m: m.name)
            for member in members:
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                rows.append(json.loads(fh.read().decode("utf-8")))
                if max_samples is not None and len(rows) >= max_samples:
                    return rows
    return rows


def _maze_reward_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    maze = raw["maze"]
    render_metadata = maze.get("render_metadata", {})
    palette = maze.get("palette", {})
    return {
        "frame_positions_pix": maze["frame_positions_pix"],
        "grid": maze["grid"],
        "goal": maze["goal"],
        "ball_rgb": render_metadata.get("ball_rgb", palette.get("ball_rgb", [220, 40, 40])),
        "cell_px": maze["cell_px"],
        "image_hw": [maze["image_h"], maze["image_w"]],
        "path": maze["path"],
        "path_len": maze.get("path_len", len(maze["path"])),
        "difficulty": maze.get("difficulty"),
        "difficulty_id": maze.get("difficulty_id"),
    }


def _make_record(raw: dict[str, Any], split: str, index: int, data_source: str) -> dict[str, Any]:
    prompt = raw["prompt"]
    global_index = raw.get("global_index", index)
    maze_meta = _maze_reward_metadata(raw)
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "negative_prompt": [
            {"role": "system", "content": ""},
            {"role": "user", "content": " "},
        ],
        "ability": "t2v",
        "reward_model": {"style": "rule", "ground_truth": prompt},
        "extra_info": {
            "split": split,
            "index": index,
            "global_index": global_index,
            "prompt": prompt,
            "maze": maze_meta,
        },
    }


def _split_rows(rows: list[dict[str, Any]], val_size: int, val_ratio: float, seed: int):
    if not rows:
        raise ValueError("No samples loaded from WebDataset")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    if val_size < 0:
        val_size = max(1, int(len(rows) * val_ratio))
    val_size = max(1, min(val_size, len(rows) - 1))
    val_idx = set(order[:val_size])
    train_rows = [row for i, row in enumerate(rows) if i not in val_idx]
    val_rows = [row for i, row in enumerate(rows) if i in val_idx]
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webdataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", default="dance_grpo/maze_tracker")
    args = parser.parse_args()

    raw_rows = _load_json_sidecars(args.webdataset_dir.expanduser(), args.max_samples)
    train_raw, val_raw = _split_rows(raw_rows, args.val_size, args.val_ratio, args.seed)
    train_records = [_make_record(row, "train", i, args.data_source) for i, row in enumerate(train_raw)]
    val_records = [_make_record(row, "test", i, args.data_source) for i, row in enumerate(val_raw)]

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "test.parquet"
    pd.DataFrame(train_records).to_parquet(train_path)
    pd.DataFrame(val_records).to_parquet(val_path)

    info = {
        "source": str(args.webdataset_dir),
        "max_samples": args.max_samples,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "data_source": args.data_source,
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Wrote {len(train_records)} train samples to {train_path}")
    print(f"Wrote {len(val_records)} val samples to {val_path}")


if __name__ == "__main__":
    main()
