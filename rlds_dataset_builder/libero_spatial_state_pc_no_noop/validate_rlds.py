#!/usr/bin/env python3
"""Validate the converted LIBERO-Spatial RLDS episode against source NPZs."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


def _load_builder_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("libero_spatial_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    # Construct a read-only builder from the generated metadata before importing
    # the source builder.  Importing the source builder first registers its
    # dynamically loaded module name with TFDS, which can make metadata lookup
    # fail even though the generated dataset itself is valid.
    builder_dir = (
        args.data_root / "libero_spatial_state_pc_no_noop" / "1.0.0"
    )
    builder = tfds.builder_from_directory(builder_dir)
    helper_path = Path(__file__).with_name(
        "libero_spatial_state_pc_no_noop_dataset_builder.py"
    )
    helpers = _load_builder_helpers(helper_path)
    episodes = list(tfds.as_numpy(builder.as_dataset(split="train")))
    if len(episodes) != 1:
        raise AssertionError(f"Expected one test episode, got {len(episodes)}")
    steps = list(episodes[0]["steps"])
    source_frames = sorted(args.source.glob("task_*/episode_*/frames/*.npz"))
    if len(steps) != len(source_frames):
        raise AssertionError(f"RLDS/source length mismatch: {len(steps)} vs {len(source_frames)}")

    max_image_mae = 0.0
    for index, (step, source_path) in enumerate(zip(steps, source_frames)):
        with np.load(source_path) as source:
            expected_pc = helpers._unproject_depth(
                source["depth_m"], source["camera_intrinsics"], source["camera_to_world"]
            )[:, ::-1]
            expected_polar = np.stack(
                (
                    source["DoLP"],
                    source["cos2AoLP"],
                    source["sin2AoLP"],
                    source["polar_valid_mask"].astype(np.float32),
                ),
                axis=-1,
            )[:, ::-1]
            expected_state = helpers._state8(source["state"])
            expected_action = helpers._openvla_action(source["action"])
            expected_image = source["rgb"][:, ::-1].astype(np.float32)

        observation = step["observation"]
        if not np.array_equal(observation["base_pc"], expected_pc):
            raise AssertionError(f"base_pc mismatch at frame {index}")
        if not np.array_equal(observation["polar"], expected_polar):
            raise AssertionError(f"polar mismatch at frame {index}")
        if not np.array_equal(observation["state"], expected_state):
            raise AssertionError(f"state mismatch at frame {index}")
        if not np.array_equal(step["action"], expected_action):
            raise AssertionError(f"action mismatch at frame {index}")
        image_mae = float(
            np.mean(np.abs(observation["image"].astype(np.float32) - expected_image))
        )
        max_image_mae = max(max_image_mae, image_mae)
        if image_mae > 5.0:
            raise AssertionError(f"Unexpected JPEG image error {image_mae:.3f} at frame {index}")
        if bool(step["is_first"]) != (index == 0):
            raise AssertionError(f"is_first mismatch at frame {index}")
        if bool(step["is_last"]) != (index == len(steps) - 1):
            raise AssertionError(f"is_last mismatch at frame {index}")

    actions = np.stack([step["action"] for step in steps])
    states = np.stack([step["observation"]["state"] for step in steps])
    if not set(np.unique(actions[:, -1])).issubset({-1.0, 1.0}):
        raise AssertionError("Converted gripper actions are not binary")
    print(
        {
            "episodes": len(episodes),
            "steps": len(steps),
            "max_jpeg_mae": max_image_mae,
            "base_pc_shape": steps[0]["observation"]["base_pc"].shape,
            "state_shape": states.shape,
            "action_shape": actions.shape,
            "gripper_action_values": np.unique(actions[:, -1]).tolist(),
        }
    )


if __name__ == "__main__":
    main()
