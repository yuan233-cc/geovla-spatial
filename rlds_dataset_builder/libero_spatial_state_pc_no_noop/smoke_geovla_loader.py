#!/usr/bin/env python3
"""Load one batch through GeoVLA's actual OXE/RLDS input pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Prefer this checkout over another installed `prismatic` package when the
# script is launched through its path from the workspace parent directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prismatic.vla.datasets.datasets import RLDSDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--future-action-window-size", type=int, default=15)
    args = parser.parse_args()

    dataset = RLDSDataset(
        data_root_dir=args.data_root,
        data_mix="libero_spatial_state_pc_no_noop",
        batch_transform=lambda batch: batch,
        resize_resolution=(224, 224),
        shuffle_buffer_size=1,
        future_action_window_size=args.future_action_window_size,
        past_action_window_size=0,
        train=True,
        image_aug=False,
        load_all_data_for_training=True,
        load_depth=True,
        proprio_type="shift_ee",
        load_wrist=False,
        wrist_first=False,
    )
    batch = next(iter(dataset))
    observation = batch["observation"]

    image = observation["image_primary"]
    point_cloud = observation["depth_primary"]
    proprio = observation["proprio"]
    actions = batch["action"]
    action_mask = batch["action_mask"]

    expected_actions = args.future_action_window_size + 1
    assert image.shape == (1, 224, 224, 3), image.shape
    assert point_cloud.shape == (1, 224, 224, 3), point_cloud.shape
    assert proprio.shape == (1, 8), proprio.shape
    assert actions.shape == (expected_actions, 7), actions.shape
    assert action_mask.shape == (expected_actions,), action_mask.shape
    for name, value in {
        "point_cloud": point_cloud,
        "proprio": proprio,
        "actions": actions,
    }.items():
        if not np.isfinite(value).all():
            raise AssertionError(f"{name} contains non-finite values")

    print(
        {
            "dataset_length": len(dataset),
            "image": (image.shape, str(image.dtype)),
            "point_cloud": (point_cloud.shape, str(point_cloud.dtype)),
            "proprio": (proprio.shape, str(proprio.dtype)),
            "actions": (actions.shape, str(actions.dtype)),
            "valid_action_slots": int(action_mask.sum()),
        }
    )


if __name__ == "__main__":
    main()
