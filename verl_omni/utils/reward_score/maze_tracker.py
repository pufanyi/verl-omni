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
"""Pixel-space maze tracker reward for Wan2.2 DanceGRPO."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _as_tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _maybe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return _maybe_json_dict(value.item())
    if not isinstance(value, dict):
        raise TypeError(f"Expected maze metadata dict, got {type(value).__name__}")
    return value


def _to_bcthw(video: torch.Tensor | np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(video).detach().float().cpu()
    if x.ndim == 3:
        if x.shape[0] in (1, 3):
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        else:
            raise ValueError(f"Cannot infer image channel dimension from shape {tuple(x.shape)}")
    elif x.ndim == 4:
        if x.shape[0] in (1, 3):
            x = x.unsqueeze(0)
        elif x.shape[1] in (1, 3):
            x = x.permute(1, 0, 2, 3).unsqueeze(0)
        elif x.shape[-1] in (1, 3):
            x = x.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            raise ValueError(f"Cannot infer video channel dimension from shape {tuple(x.shape)}")
    elif x.ndim == 5:
        if x.shape[1] in (1, 3):
            pass
        elif x.shape[2] in (1, 3):
            x = x.permute(0, 2, 1, 3, 4)
        elif x.shape[-1] in (1, 3):
            x = x.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError(f"Cannot infer batched video channel dimension from shape {tuple(x.shape)}")
    else:
        raise ValueError(f"Expected image/video tensor with 3, 4, or 5 dims, got shape {tuple(x.shape)}")

    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1, -1)
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3 color channels, got shape {tuple(x.shape)}")
    if x.min().item() < -0.1:
        x = (x + 1.0) * 127.5
    elif x.max().item() <= 1.5:
        x = x * 255.0
    return x.clamp(0.0, 255.0)


def _weighted_best_xy(dist: torch.Tensor, *, color_slack: float, x_offset: int = 0, y_offset: int = 0):
    min_val = dist.min()
    mask = dist <= min_val + float(color_slack) * float(color_slack)
    ys, xs = torch.nonzero(mask, as_tuple=True)
    weights = 1.0 / (dist[ys, xs] + 1.0)
    weight_sum = weights.sum().clamp(min=1e-8)
    x = ((xs.float() + float(x_offset)) * weights).sum() / weight_sum
    y = ((ys.float() + float(y_offset)) * weights).sum() / weight_sum
    return torch.stack((x, y)), min_val


def _track_color_object(
    pixel_video: torch.Tensor,
    ball_rgb: torch.Tensor,
    initial_xy: torch.Tensor,
    *,
    search_radius: int,
    color_slack: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, _, num_frames, height, width = pixel_video.shape
    positions = pixel_video.new_empty((bsz, num_frames, 2))
    confidences = pixel_video.new_empty((bsz, num_frames))
    prev_xy = initial_xy.float()

    for t in range(num_frames):
        frame = pixel_video[:, :, t]
        dist = (frame - ball_rgb.view(bsz, 3, 1, 1).float()).pow(2).sum(dim=1)
        next_prev = []
        for b in range(bsz):
            global_xy, global_min = _weighted_best_xy(dist[b], color_slack=color_slack)
            px, py = prev_xy[b]
            x0 = max(0, int(round(float(px.item()))) - int(search_radius))
            x1 = min(width, int(round(float(px.item()))) + int(search_radius) + 1)
            y0 = max(0, int(round(float(py.item()))) - int(search_radius))
            y1 = min(height, int(round(float(py.item()))) + int(search_radius) + 1)

            xy = global_xy
            min_val = global_min
            if x1 > x0 and y1 > y0:
                local_xy, local_min = _weighted_best_xy(
                    dist[b, y0:y1, x0:x1],
                    color_slack=color_slack,
                    x_offset=x0,
                    y_offset=y0,
                )
                local_limit = torch.maximum(global_min.sqrt() + color_slack, global_min.new_tensor(color_slack * 3.0))
                if local_min.sqrt() <= local_limit:
                    xy = local_xy
                    min_val = local_min

            positions[b, t] = xy
            confidences[b, t] = torch.exp(-min_val.sqrt() / 80.0)
            next_prev.append(xy)
        prev_xy = torch.stack(next_prev, dim=0)

    return positions, confidences


def _path_progress_score(
    det_xy: torch.Tensor,
    path_ij: torch.Tensor,
    path_len: torch.Tensor,
    cell_px_x: torch.Tensor,
    cell_px_y: torch.Tensor,
) -> torch.Tensor:
    bsz, selected_frames, _ = det_xy.shape
    _, max_path, _ = path_ij.shape
    path_len = path_len.long().clamp(min=1, max=max_path)
    path_x = (path_ij[..., 1].float() + 0.5) * cell_px_x.view(bsz, 1)
    path_y = (path_ij[..., 0].float() + 0.5) * cell_px_y.view(bsz, 1)
    path_xy = torch.stack((path_x, path_y), dim=-1)

    dists = torch.cdist(det_xy.float(), path_xy.float())
    valid = torch.arange(max_path).view(1, 1, max_path) < path_len.view(bsz, 1, 1)
    dists = dists.masked_fill(~valid, float("inf"))
    nearest = dists.argmin(dim=-1).float()
    if selected_frames > 1:
        monotonic_fraction = (nearest[:, 1:] - nearest[:, :-1] >= -2).float().mean(dim=1)
    else:
        monotonic_fraction = torch.ones(bsz)
    progress = ((nearest[:, -1] - nearest[:, 0]) / (path_len.float() - 1.0).clamp(min=1.0)).clamp(0.0, 1.0)
    return progress * monotonic_fraction


def _score_one(pixel_video: torch.Tensor, maze: dict[str, Any]) -> dict[str, float]:
    _, _, total_frames, height, width = pixel_video.shape
    image_hw = _as_tensor(maze["image_hw"], dtype=torch.float32)
    source_h, source_w = float(image_hw[0].item()), float(image_hw[1].item())
    scale_x = float(width) / max(source_w, 1.0)
    scale_y = float(height) / max(source_h, 1.0)

    fp_pix = _as_tensor(maze["frame_positions_pix"], dtype=torch.float32)
    fp_pix = fp_pix.clone()
    fp_pix[:, 0] *= scale_x
    fp_pix[:, 1] *= scale_y
    fp_pix = fp_pix.unsqueeze(0)

    grid = _as_tensor(maze["grid"], dtype=torch.long).unsqueeze(0)
    goal_ij = _as_tensor(maze["goal"], dtype=torch.long).unsqueeze(0)
    ball_rgb = _as_tensor(maze["ball_rgb"], dtype=torch.float32).view(1, 3)
    path_ij = _as_tensor(maze["path"], dtype=torch.long).unsqueeze(0)
    path_len = _as_tensor(maze.get("path_len", len(maze["path"])), dtype=torch.long).view(1)

    cell_px = float(maze["cell_px"])
    cell_px_x = torch.tensor([cell_px * scale_x], dtype=torch.float32)
    cell_px_y = torch.tensor([cell_px * scale_y], dtype=torch.float32)
    cell_px_avg = ((cell_px_x + cell_px_y) * 0.5).clamp(min=1.0)

    num_score_frames = min(max(1, _env_int("MAZE_TRACKER_REWARD_NUM_FRAMES", 21)), total_frames)
    frame_idxs = torch.linspace(0, total_frames - 1, num_score_frames).round().long()

    expected_source_idxs = torch.linspace(0, fp_pix.shape[1] - 1, total_frames).round().long()
    expected_video_xy = fp_pix.index_select(1, expected_source_idxs)
    expected_xy = expected_video_xy.index_select(1, frame_idxs)

    search_radius = int(round(_env_int("MAZE_TRACKER_REWARD_SEARCH_RADIUS", 96) * (scale_x + scale_y) * 0.5))
    det_all, _confidence_all = _track_color_object(
        pixel_video,
        ball_rgb,
        expected_video_xy[:, 0],
        search_radius=max(1, search_radius),
        color_slack=_env_float("MAZE_TRACKER_REWARD_COLOR_SLACK", 28.0),
    )
    det_xy = det_all.index_select(1, frame_idxs)

    dists = (det_xy - expected_xy).pow(2).sum(dim=-1).sqrt()
    mean_error_px = dists.mean(dim=1)
    max_mean_error_px = _env_float("MAZE_TRACKER_REWARD_MAX_MEAN_ERROR_CELLS", 4.0) * cell_px_avg
    traj_score = (1.0 - mean_error_px / max_mean_error_px).clamp(min=0.0)

    cell_i = (det_xy[..., 1] / cell_px_y.view(1, 1)).long().clamp(0, grid.shape[1] - 1)
    cell_j = (det_xy[..., 0] / cell_px_x.view(1, 1)).long().clamp(0, grid.shape[2] - 1)
    on_path_fraction = (grid[0, cell_i[0], cell_j[0]] == 0).float().mean().view(1)

    goal_x = (goal_ij[:, 1].float() + 0.5) * cell_px_x
    goal_y = (goal_ij[:, 0].float() + 0.5) * cell_px_y
    end_error_px = ((det_xy[:, -1, 0] - goal_x).pow(2) + (det_xy[:, -1, 1] - goal_y).pow(2)).sqrt()
    goal_score = (end_error_px <= _env_float("MAZE_TRACKER_REWARD_GOAL_TOLERANCE_CELLS", 1.0) * cell_px_avg).float()

    progress_score = _path_progress_score(det_xy, path_ij, path_len, cell_px_x, cell_px_y)
    overall = (
        _env_float("MAZE_TRACKER_REWARD_W_TRAJ", 0.35) * traj_score
        + _env_float("MAZE_TRACKER_REWARD_W_ONPATH", 0.25) * on_path_fraction
        + _env_float("MAZE_TRACKER_REWARD_W_GOAL", 0.25) * goal_score
        + _env_float("MAZE_TRACKER_REWARD_W_PROGRESS", 0.15) * progress_score
    )
    return {
        "score": float(overall.item()),
        "traj": float(traj_score.item()),
        "onpath": float(on_path_fraction.item()),
        "goal": float(goal_score.item()),
        "progress": float(progress_score.item()),
        "mean_error_cells": float((mean_error_px / cell_px_avg).item()),
        "end_error_cells": float((end_error_px / cell_px_avg).item()),
    }


def compute_score_maze_tracker(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
) -> dict[str, float]:
    """Score a generated maze video against per-sample maze metadata."""
    del data_source, ground_truth, kwargs
    extra_info = _maybe_json_dict(extra_info)
    maze = _maybe_json_dict(extra_info["maze"])
    pixel_video = _to_bcthw(solution_image)
    if pixel_video.shape[0] != 1:
        raise ValueError(f"Expected a single sample in reward scorer, got batch shape {tuple(pixel_video.shape)}")
    return _score_one(pixel_video, maze)
