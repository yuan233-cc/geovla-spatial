"""Build stock-GeoVLA RLDS from the clean LIBERO-Spatial NPZ export."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow_datasets as tfds


def _load_frame(frame_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompress one frame; called by the per-episode read-ahead pool."""
    with np.load(frame_path) as frame:
        return (
            np.ascontiguousarray(frame["image"], dtype=np.uint8),
            np.ascontiguousarray(frame["base_pc"], dtype=np.float32),
            np.ascontiguousarray(frame["state"], dtype=np.float32),
            np.ascontiguousarray(frame["action"], dtype=np.float32),
        )


class LiberoSpatialStatePcNoNoop(tfds.core.GeneratorBasedBuilder):
    """Successful LIBERO-Spatial demonstrations with organized world XYZ."""

    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {"1.1.0": "Full clean Spatial dataset without unused polar tensors."}
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Pass the geovla_clean_npz_v1 root with --manual_dir."

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            description=(
                "LIBERO-Spatial RGB, organized world-frame point clouds, POS_QUAT state, "
                "OpenVLA-convention actions, and language instructions. Observations are "
                "rendered at each official recorded simulator state and rotated 180 degrees "
                "to match the OpenVLA LIBERO training/evaluation convention."
            ),
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
        source = Path(dl_manager.manual_dir)
        manifest_path = source / "manifest.json"
        if not manifest_path.is_file():
            candidates = list(source.glob("*/manifest.json"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Cannot identify one clean manifest under {source}")
            source = candidates[0].parent
            manifest_path = candidates[0]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete", False):
            raise ValueError(f"Refusing incomplete dataset: {source}")
        if manifest.get("format") != "geovla_clean_npz_v1":
            raise ValueError(f"Unexpected source format: {manifest.get('format')}")
        if manifest.get("resolution") != 256:
            raise ValueError(f"Expected 256 px, got {manifest.get('resolution')}")
        if int(manifest.get("episodes", -1)) != 500:
            raise ValueError(f"Expected 500 episodes, got {manifest.get('episodes')}")
        return {"train": self._generate_examples(source)}

    def _generate_examples(self, source: Path) -> Iterator[tuple[str, dict]]:
        summaries = sorted(source.glob("task_*/episode_*/summary.json"))
        if len(summaries) != 500:
            raise ValueError(f"Expected 500 episode summaries, got {len(summaries)}")
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not summary.get("complete", False) or not summary.get("success", False):
                raise ValueError(f"Incomplete or failed episode: {summary_path.parent}")
            frame_paths = sorted((summary_path.parent / "frames").glob("*.npz"))
            if len(frame_paths) != int(summary["saved_frames"]):
                raise ValueError(f"Frame-count mismatch: {summary_path.parent}")
            language = str(summary["language"])

            def steps():
                last_index = len(frame_paths) - 1
                # NPZ inflate releases the GIL. Read-ahead overlaps it with TFDS's
                # canonical JPEG encoding and TFRecord serialization in the main thread.
                with ThreadPoolExecutor(max_workers=4) as readers:
                    loaded = readers.map(_load_frame, frame_paths)
                    for index, (image, base_pc, state, action) in enumerate(loaded):
                        if not np.isfinite(base_pc).all() or not np.isfinite(state).all() or not np.isfinite(action).all():
                            raise ValueError(f"Non-finite values in {frame_paths[index]}")
                        is_last = index == last_index
                        yield {
                            "observation": {"image": image, "base_pc": base_pc, "state": state},
                            "action": action,
                            "discount": np.float32(0.0 if is_last else 1.0),
                            "reward": np.float32(1.0 if is_last else 0.0),
                            "is_first": index == 0,
                            "is_last": is_last,
                            "is_terminal": is_last,
                            "language_instruction": language,
                        }

            key = f"task_{int(summary['task_id']):02d}_episode_{int(summary['source_episode']):06d}"
            yield key, {
                "steps": steps(),
                "episode_metadata": {
                    # Keep public RLDS metadata portable and avoid embedding a
                    # machine-specific absolute path from the conversion host.
                    "source": Path(str(summary["source"])).name,
                    "task": str(summary["task"]),
                    "task_id": np.int32(summary["task_id"]),
                    "source_episode": np.int32(summary["source_episode"]),
                },
            }
