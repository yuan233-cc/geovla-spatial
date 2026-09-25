# LIBERO-Spatial reproduction

The public RLDS release is available at
[`yuan1119/libero-spatial-clean-geovla-rlds-v1`](https://huggingface.co/datasets/yuan1119/libero-spatial-clean-geovla-rlds-v1).
It contains all 10 LIBERO-Spatial tasks, 50 successful demonstrations per
task, and 62,153 transitions after the OpenVLA no-op filter.
The dataset config explicitly loads the packaged 500-trajectory statistics
artifact, so moving or mounting the archive at another absolute path does not
trigger a full statistics recomputation.

After extracting or mounting the archive, the data root must contain:

```text
libero_spatial_state_pc_no_noop/1.1.0/
```

Use the official OpenVLA-Prismatic initialization checkpoint from
`openvla/openvla-7b-prismatic`:

```text
config.json
dataset_statistics.json
checkpoints/step-295000-epoch-40-loss=0.2200.pt
```

The Spatial-only launch script is `scripts/train_libero_spatial_h200.sh`.
Its architecture and semantic parameters follow the repository's LIBERO 3D
MoE recipe: full point cloud, `dit_condition_self`, DiT-B, MoE enabled,
`shift_ee` proprioception, 16-action chunks, eight repeated diffusion steps,
learning rate `2e-5`, and global batch 256. It defaults to eight epochs,
matching the epoch of the released repository's referenced LIBERO 3D-MoE
checkpoint.

The original shell recipe assumes 8 GPUs and per-device batch 32. On one H200,
the launcher defaults to per-device batch 1 and preserves global batch 256 via
gradient accumulation. The VLA loop in this branch adds accumulation support;
the original loop rejected any accumulation factor other than one. This is
numerically close but not an exact hardware or throughput reproduction, and a
single-GPU run will be much slower than the paper's eight-GPU run.

Because the RLDS input pipeline repeats indefinitely, the launcher converts the
requested eight epochs into 1,944 optimizer steps using the released dataset's
62,153 transitions and global batch 256. Set `DATASET_TRANSITIONS` only when
using a different release. An explicit `MAX_STEPS` still overrides this value.

Required environment variables:

```bash
export DATA_ROOT_DIR=/path/to/rlds/root
export PRETRAINED_CHECKPOINT=/path/to/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt
export LLAMA2_7B_PATH=/path/to/llama2-config-and-tokenizer
export RUN_ROOT_DIR=/persistent/checkpoints
export RUN_ID=libero-spatial-geovla-3dmoe-seed42
export WANDB_ENTITY=verified-entity-slug
export WANDB_PROJECT=geovla_libero_spatial
# HF_TOKEN is optional because every required model asset is local.
export PYTHON_BIN=/path/to/geovla/venv/bin/python

bash scripts/train_libero_spatial_h200.sh
```

`LLAMA2_7B_PATH` only needs the Llama-2 architecture config and tokenizer files.
The OpenVLA Prismatic checkpoint contains the complete LLM and vision state
dicts, so the loader constructs both backbones without downloading their base
weights and then loads those checkpoint tensors. The LLM load remains strict.

Before a full run, use a separate absent `RUN_ID` and `MAX_STEPS=1` to verify
one optimizer step, checkpoint writing, W&B initialization, and GPU memory.

For the CAMP H200 workflow, `scripts/prepare_libero_spatial_h200_runtime.sh`
verifies the exact dataset/model/runtime checksums, publishes the prebuilt
environment, mounts the ZIP without extraction, and runs the real RLDS loader.
After that gate passes, run `scripts/launch_libero_spatial_h200.sh` on the host;
it keeps Enroot in the foreground and places W&B runtime state under job-local
`/tmp`, while checkpoints remain on Aachen storage. Neither script embeds an
access token.
