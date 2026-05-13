# VABench / SELD Data Preparation and Training Guide

This repository contains the data preparation scripts and training configuration used to build and run SELD experiments on:

- `STARSS23`
- `SpatialQA / HFData`
- merged `STARSS23 + HFData`

The repository also contains two model families:

- original `DCASE2024 SELD baseline`
- modified `dcase2024-SedHead`

This README is written for someone who did **not** participate in the original setup and needs a practical, end-to-end overview.

## 1. What Is Included Here

### Data preparation

Under [`code/`](code):

- `prepare_real_foa_to_dcase.py`
- `segment_starss23_fixed20s.py`
- `segment_spatialqa_event_based.py`
- `convert_spatialqa_json_to_dcase_csv.py`
- `normalize_hf_json_headings.py`
- `resample_merged_foa_dataset_to_16k.py`

### Training code

Under [`dcase2024-SedHead/`](dcase2024-SedHead):

- `parameters.py`
- `train_seldnet.py`
- `batch_feature_extraction.py`
- modified label / loss handling code
- wait-launch scripts for queued experiments

### Documentation

- dataset processing summary:
  [`data_preparation_summary.md`](data_preparation_summary.md)
- full Chinese reproduction guide for newcomers:
  [`REPRODUCTION_GUIDE_ZH.md`](REPRODUCTION_GUIDE_ZH.md)
- code-specific Chinese README:
  [`code/README.md`](code/README.md)

## 2. What Is Not Included

This repository does **not** include the raw datasets themselves.

You need to prepare:

- raw `STARSS23`
- raw `HFData / SpatialQA`

and place them into the expected directories on your server before running the scripts.

## 3. Final Dataset Roots Used for Training

### STARSS23-only

Final processed dataset root:

- `/data/zhuzhiyuan/starss23/STARSS23_20s_16k`

Used by:

- `247` / `251` / `249` / `253`

### STARSS23 + HFData merged

Final processed dataset root:

- `/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k`

Used by:

- `248` / `252` / `250` / `254`

## 4. High-Level Processing Pipeline

### Step 1. Prepare raw STARSS23

Start from the original STARSS23 dataset root:

- `/data/zhuzhiyuan/starss23/STARSS23`

Then:

1. segment into fixed 20-second clips
2. resample to 16kHz
3. generate a STARSS23-only split manifest

Outputs:

- `/data/zhuzhiyuan/starss23/STARSS23_20s`
- `/data/zhuzhiyuan/starss23/STARSS23_20s_16k`

### Step 2. Prepare raw HFData / SpatialQA

Start from raw audio/video/json files.

Then:

1. flatten and rename files by timestamp
2. normalize `heading` to `[-180, 180)`
3. map / merge event categories
4. remove events mapped to `0`
5. segment by event-aware rules
6. convert JSON annotations into STARSS23-style CSV

Outputs:

- `/data/zhuzhiyuan/starss23/SpatialQA_hf`
- `/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented`
- `/data/zhuzhiyuan/starss23/SpatialQA_hf_csv`
- `/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented_csv`

### Step 3. Merge STARSS23 + HFData

Create a DCASE-style merged dataset root:

- `foa_dev/`
- `metadata_dev/`
- `video_dev/`
- `class_mapping.json`
- `split_manifest.json`

Final root:

- `/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k`

## 5. Final Split Policy

### STARSS23-only

- `train`: 90% of STARSS23 train
- `valid`: 10% of STARSS23 train
- `test`: official STARSS23 test

### STARSS23 + HFData

- `train`: STARSS23 train + HF train + HF test
- `valid`: HF eval
- `test`: STARSS23 test

This means:

- final benchmark score is always reported on `STARSS23 test`

## 6. Experiment Matrix

### Modified dcase2024-SedHead

- `247`: STARSS23-only, `audio-only`
- `248`: STARSS23 + HF, `audio-only`
- `249`: STARSS23-only, `audio-visual`
- `250`: STARSS23 + HF, `audio-visual`

### Original DCASE2024 / SELD baseline

- `251`: STARSS23-only, `audio-only`
- `252`: STARSS23 + HF, `audio-only`
- `253`: STARSS23-only, `audio-visual`
- `254`: STARSS23 + HF, `audio-visual`

## 7. Feature Extraction Directories

Training does not directly read raw `.wav` or `.mp4`. It reads extracted `.npy` feature caches.

### SedHead feature dirs

- `247` -> `/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task247`
- `248` -> `/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task248`
- `249` -> `/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task249_av`
- `250` -> `/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task250_av`

### Baseline feature reuse

To save time, baseline tasks reuse the exact same extracted features:

- `251` reuses `247`
- `252` reuses `248`
- `253` reuses `249`
- `254` reuses `250`

So you do **not** need to extract the same features twice.

## 8. Minimal Training Order

Recommended order:

1. run `247` and `248`
2. run `251` and `252`
3. run `249` and `250`
4. run `253` and `254`

This gives:

- audio-only results first
- then audio-visual results
- and avoids repeated feature extraction

## 9. Queue Scripts

These scripts wait for resources and then launch experiments sequentially.

### STARSS23-only queue

- [`dcase2024-SedHead/run_starss23_only_queue.sh`](dcase2024-SedHead/run_starss23_only_queue.sh)

Order:

- `247 -> 251 -> 249 -> 253`

### STARSS23 + HF queue

- [`dcase2024-SedHead/run_starss23_plus_hf_queue.sh`](dcase2024-SedHead/run_starss23_plus_hf_queue.sh)

Order:

- `248 -> 252 -> 250 -> 254`

## 10. What to Read Next

If you need more detail:

- full data processing notes:
  [`data_preparation_summary.md`](data_preparation_summary.md)
- code-level Chinese explanation:
  [`code/README.md`](code/README.md)
