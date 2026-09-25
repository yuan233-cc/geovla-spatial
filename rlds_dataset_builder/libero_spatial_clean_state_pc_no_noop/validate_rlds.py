#!/usr/bin/env python3
"""Exhaustively compare the clean LIBERO-Spatial RLDS with its NPZ source."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-image-mae", type=float, default=5.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or int(manifest.get("episodes", -1)) != 500:
        raise AssertionError("NPZ source does not have a complete 500-episode manifest")

    builder_dir = args.data_root / "libero_spatial_state_pc_no_noop" / "1.1.0"
    builder = tfds.builder_from_directory(builder_dir)
    dataset = tfds.as_numpy(builder.as_dataset(split="train", shuffle_files=False))

    episode_count = 0
    step_count = 0
    task_counts: Counter[int] = Counter()
    max_image_mae = 0.0
    pc_min = np.full(3, np.inf, dtype=np.float64)
    pc_max = np.full(3, -np.inf, dtype=np.float64)
    gripper_actions: set[float] = set()

    for episode in dataset:
        metadata = episode["episode_metadata"]
        task_id = int(metadata["task_id"])
        source_episode = int(metadata["source_episode"])
        episode_dir = args.source / f"task_{task_id:02d}" / f"episode_{source_episode:06d}"
        summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
        frame_paths = sorted((episode_dir / "frames").glob("*.npz"))
        if len(frame_paths) != int(summary["saved_frames"]):
            raise AssertionError(f"Source frame-count mismatch: {episode_dir}")
        if _text(metadata["task"]) != str(summary["task"]):
            raise AssertionError(f"Task metadata mismatch: {episode_dir}")

        local_steps = 0
        for index, pair in enumerate(zip_longest(episode["steps"], frame_paths)):
            step, frame_path = pair
            if step is None or frame_path is None:
                raise AssertionError(f"RLDS/source step-count mismatch: {episode_dir}")
            with np.load(frame_path) as source:
                expected_image = np.asarray(source["image"], dtype=np.float32)
                expected_pc = np.asarray(source["base_pc"], dtype=np.float32)
                expected_state = np.asarray(source["state"], dtype=np.float32)
                expected_action = np.asarray(source["action"], dtype=np.float32)

            observation = step["observation"]
            point_cloud = observation["base_pc"]
            if not np.array_equal(point_cloud, expected_pc):
                raise AssertionError(f"base_pc mismatch: {frame_path}")
            if not np.array_equal(observation["state"], expected_state):
                raise AssertionError(f"state mismatch: {frame_path}")
            if not np.array_equal(step["action"], expected_action):
                raise AssertionError(f"action mismatch: {frame_path}")
            if not np.isfinite(point_cloud).all():
                raise AssertionError(f"non-finite base_pc: {frame_path}")

            image_mae = float(
                np.mean(np.abs(observation["image"].astype(np.float32) - expected_image))
            )
            max_image_mae = max(max_image_mae, image_mae)
            if image_mae > args.max_image_mae:
                raise AssertionError(f"JPEG MAE {image_mae:.3f}: {frame_path}")

            is_first = index == 0
            is_last = index == len(frame_paths) - 1
            if bool(step["is_first"]) != is_first:
                raise AssertionError(f"is_first mismatch: {frame_path}")
            if bool(step["is_last"]) != is_last or bool(step["is_terminal"]) != is_last:
                raise AssertionError(f"terminal flag mismatch: {frame_path}")
            expected_reward = 1.0 if is_last else 0.0
            expected_discount = 0.0 if is_last else 1.0
            if float(step["reward"]) != expected_reward or float(step["discount"]) != expected_discount:
                raise AssertionError(f"reward/discount mismatch: {frame_path}")
            if _text(step["language_instruction"]) != str(summary["language"]):
                raise AssertionError(f"language mismatch: {frame_path}")

            pc_min = np.minimum(pc_min, point_cloud.reshape(-1, 3).min(axis=0))
            pc_max = np.maximum(pc_max, point_cloud.reshape(-1, 3).max(axis=0))
            gripper_actions.add(float(step["action"][-1]))
            local_steps += 1
            step_count += 1

        if local_steps != len(frame_paths):
            raise AssertionError(f"RLDS/source step-count mismatch: {episode_dir}")
        task_counts[task_id] += 1
        episode_count += 1

    if episode_count != 500 or step_count != 62153:
        raise AssertionError(f"Expected 500 episodes/62153 steps, got {episode_count}/{step_count}")
    if task_counts != Counter({task_id: 50 for task_id in range(10)}):
        raise AssertionError(f"Unexpected per-task counts: {dict(task_counts)}")
    if not gripper_actions.issubset({-1.0, 1.0}):
        raise AssertionError(f"Unexpected gripper values: {sorted(gripper_actions)}")

    report = {
        "status": "passed",
        "episodes": episode_count,
        "steps": step_count,
        "episodes_per_task": {str(key): task_counts[key] for key in sorted(task_counts)},
        "max_jpeg_mae": max_image_mae,
        "base_pc_min_xyz": pc_min.tolist(),
        "base_pc_max_xyz": pc_max.tolist(),
        "gripper_action_values": sorted(gripper_actions),
        "source_format": manifest["format"],
        "rlds_version": "1.1.0",
    }
    report_path = args.report or (args.data_root / "validation_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
