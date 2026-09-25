#!/usr/bin/env python3
"""Run one full future-action window through GeoVLA's actual RLDS loader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

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
    expected_actions = args.future_action_window_size + 1
    checks = {
        "image": (observation["image_primary"], (1, 224, 224, 3)),
        "point_cloud": (observation["depth_primary"], (1, 224, 224, 3)),
        "proprio": (observation["proprio"], (1, 8)),
        "actions": (batch["action"], (expected_actions, 7)),
        "action_mask": (batch["action_mask"], (expected_actions,)),
    }
    result = {"dataset_length": len(dataset)}
    for name, (value, expected_shape) in checks.items():
        if value.shape != expected_shape:
            raise AssertionError(f"{name}: expected {expected_shape}, got {value.shape}")
        if name != "image" and not np.isfinite(value).all():
            raise AssertionError(f"{name} contains non-finite values")
        result[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    result["valid_action_slots"] = int(batch["action_mask"].sum())
    print(result)


if __name__ == "__main__":
    main()
