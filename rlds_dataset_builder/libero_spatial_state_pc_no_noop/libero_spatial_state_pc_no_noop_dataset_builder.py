"""Build a GeoVLA-compatible RLDS sample from LIBERO polar replay NPZs.

The source NPZ remains the auditable superset.  This builder emits the dense
organized XYZ representation consumed by stock GeoVLA and preserves a compact
four-channel polarization map for the later polar-fusion model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow_datasets as tfds


_DESCRIPTION = """LIBERO-Spatial demonstrations with RGB, organized world-frame
point clouds, robot state, actions, and aligned polarization.  RGB, point-grid,
and polar images use the 180-degree LIBERO/OpenVLA training convention.  The
gripper action uses the OpenVLA convention: +1 is open and -1 is closed."""


def _unproject_depth(
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Unproject MuJoCo z-depth into an organized world-frame XYZ grid."""
    depth = np.asarray(depth_m, dtype=np.float32)
    k = np.asarray(intrinsics, dtype=np.float32)
    pose = np.asarray(camera_to_world, dtype=np.float32)
    height, width = depth.shape
    v, u = np.indices((height, width), dtype=np.float32)
    camera = np.stack(
        (
            (u + 0.5 - k[0, 2]) * depth / k[0, 0],
            (v + 0.5 - k[1, 2]) * depth / k[1, 1],
            depth,
        ),
        axis=-1,
    )
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(world).all():
        raise ValueError("Generated base_pc contains non-finite values")
    return np.ascontiguousarray(world, dtype=np.float32)


def _state8(raw_state: np.ndarray) -> np.ndarray:
    """Convert EEF XYZ + quaternion + two finger joints into POS_QUAT state8.

    LIBERO raw gripper commands use +1=close, -1=open.  We use the same sign
    semantics for the binary state value.  The 0.04-radian width threshold lies
    between the observed open (~0.08) and closed (~0.007) finger separation.
    """
    raw = np.asarray(raw_state, dtype=np.float32)
    if raw.shape != (9,):
        raise ValueError(f"Expected raw state shape (9,), got {raw.shape}")
    finger_width = float(raw[7] - raw[8])
    gripper_closed = 1.0 if finger_width < 0.04 else -1.0
    return np.ascontiguousarray(np.concatenate((raw[:7], [gripper_closed])), dtype=np.float32)


def _openvla_action(raw_action: np.ndarray) -> np.ndarray:
    """Change raw LIBERO -1=open/+1=close to OpenVLA +1=open/-1=close."""
    action = np.asarray(raw_action, dtype=np.float32).copy()
    if action.shape != (7,):
        raise ValueError(f"Expected action shape (7,), got {action.shape}")
    if not np.isin(action[-1], (-1.0, 1.0)):
        raise ValueError(f"Expected binary raw gripper action, got {action[-1]}")
    action[-1] *= -1.0
    return action


def _source_root(manual_dir: Path) -> Path:
    manual_dir = Path(manual_dir)
    if (manual_dir / "manifest.json").is_file():
        return manual_dir
    candidates = [p.parent for p in manual_dir.glob("*/manifest.json")]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "manual_dir must contain manifest.json directly or exactly one child dataset; "
            f"found {len(candidates)} candidates in {manual_dir}"
        )
    return candidates[0]


class LiberoSpatialStatePcNoNoop(tfds.core.GeneratorBasedBuilder):
    """One RLDS episode per successfully replayed LIBERO demonstration."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial organized-PC and polar conversion."}
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "Pass the LIBERO polar replay dataset root using --manual_dir."
    )

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "base_pc": tfds.features.Tensor(
                                        shape=(256, 256, 3), dtype=np.float32
                                    ),
                                    "polar": tfds.features.Tensor(
                                        shape=(256, 256, 4), dtype=np.float32
                                    ),
                                    "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "source": tfds.features.Text(),
                            "task": tfds.features.Text(),
                            "task_id": tfds.features.Scalar(dtype=np.int32),
                            "source_episode": tfds.features.Scalar(dtype=np.int32),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://libero-project.github.io/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        source = _source_root(Path(dl_manager.manual_dir))
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete", False):
            raise ValueError(f"Refusing incomplete/debug source dataset: {source}")
        if manifest.get("resolution") != 256:
            raise ValueError(f"Expected 256px source, got {manifest.get('resolution')}")
        if manifest.get("state_source") != "replay":
            raise ValueError("GeoVLA reproduction builder requires replay state_source")
        return {"train": self._generate_examples(source)}

    def _generate_examples(self, source: Path) -> Iterator[tuple[str, dict]]:
        for summary_path in sorted(source.glob("task_*/episode_*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not summary.get("complete", False) or not summary.get("success", False):
                continue
            frame_paths = sorted((summary_path.parent / "frames").glob("*.npz"))
            if len(frame_paths) != int(summary["saved_frames"]):
                raise ValueError(f"Frame count mismatch under {summary_path.parent}")

            language = str(summary["language"])

            def steps():
                last_index = len(frame_paths) - 1
                for index, frame_path in enumerate(frame_paths):
                    with np.load(frame_path) as frame:
                        rgb = np.ascontiguousarray(frame["rgb"][:, ::-1])
                        base_pc = _unproject_depth(
                            frame["depth_m"],
                            frame["camera_intrinsics"],
                            frame["camera_to_world"],
                        )[:, ::-1]
                        polar = np.stack(
                            (
                                frame["DoLP"],
                                frame["cos2AoLP"],
                                frame["sin2AoLP"],
                                frame["polar_valid_mask"].astype(np.float32),
                            ),
                            axis=-1,
                        )[:, ::-1]
                        is_last = index == last_index
                        yield {
                            "observation": {
                                "image": rgb,
                                "base_pc": np.ascontiguousarray(base_pc, dtype=np.float32),
                                "polar": np.ascontiguousarray(polar, dtype=np.float32),
                                "state": _state8(frame["state"]),
                            },
                            "action": _openvla_action(frame["action"]),
                            "discount": np.float32(0.0 if is_last else 1.0),
                            "reward": np.float32(1.0 if is_last else 0.0),
                            "is_first": index == 0,
                            "is_last": is_last,
                            "is_terminal": is_last,
                            "language_instruction": language,
                        }

            episode_key = f"task_{int(summary['task_id']):02d}_episode_{int(summary['source_episode']):06d}"
            yield episode_key, {
                "steps": steps(),
                "episode_metadata": {
                    "source": str(summary["source"]),
                    "task": str(summary["task"]),
                    "task_id": np.int32(summary["task_id"]),
                    "source_episode": np.int32(summary["source_episode"]),
                },
            }
