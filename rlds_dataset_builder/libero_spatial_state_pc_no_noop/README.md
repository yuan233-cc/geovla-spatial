# LIBERO-Spatial GeoVLA RLDS builder

Build the one-episode validation dataset with:

```bash
export PYTHONNOUSERSITE=1
tfds build \
  rlds_dataset_builder/libero_spatial_state_pc_no_noop/libero_spatial_state_pc_no_noop_dataset_builder.py \
  --manual_dir /path/to/libero_spatial_intermediate_npz \
  --data_dir /path/to/libero_spatial_test_rlds \
  --overwrite
```

The RLDS action follows OpenVLA convention (`+1=open`, `-1=close`), so the
raw LIBERO gripper action is inverted.  RGB, organized XYZ and polar maps are
all horizontally flipped relative to the intermediate NPZ because the NPZ has
already applied MuJoCo's vertical framebuffer flip; this produces the same
180-degree convention used by the official LIBERO OpenVLA evaluation wrapper.
The source replay exporter already removed no-op transitions using the official
criterion (near-zero arm command with an unchanged gripper command).

Stock GeoVLA reads `image`, `base_pc`, and `state`.  `polar` is preserved in
RLDS but intentionally ignored until the polar encoder is added.

## Verified test artifact

The one-episode validation build used this layout:

```text
/path/to/libero_spatial_test_rlds/libero_spatial_state_pc_no_noop/1.0.0
```

It contains one successful 98-step episode (LIBERO-Spatial task 0) and occupies
174 MiB.  Full-step validation produced:

```text
episodes=1, steps=98
base_pc=(256, 256, 3), state=(98, 8), action=(98, 7)
gripper_action_values=[-1.0, 1.0]
max_jpeg_mae=1.5858
```

The stock GeoVLA RLDS loader was also tested with a 16-step future-action
window.  Its batch contained RGB `(1, 224, 224, 3)`, organized XYZ
`(1, 224, 224, 3)`, proprioception `(1, 8)`, actions `(16, 7)`, and only finite
values.  The XYZ batch was then passed through `depth_preprocess` and
`DiTDepthEncoder`; all returned feature and position tensors were finite.

Validate every converted step against its source NPZ with:

```bash
python rlds_dataset_builder/libero_spatial_state_pc_no_noop/validate_rlds.py \
  --source /path/to/libero_spatial_intermediate_npz \
  --data-root /path/to/libero_spatial_test_rlds
```

Re-run the stock GeoVLA loader smoke test with:

```bash
python rlds_dataset_builder/libero_spatial_state_pc_no_noop/smoke_geovla_loader.py \
  --data-root /path/to/libero_spatial_test_rlds
```

For the single-episode overfit check on a multi-GPU node, set
`PRETRAINED_CHECKPOINT` and the other variables documented by the launcher,
then run:

```bash
scripts/train_libero_spatial_1ep_overfit.sh
```

This is an integration/overfit test dataset, not a statistically meaningful
training set and not a benchmark result.
