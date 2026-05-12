# 数据处理与训练输入总结

本文档总结当前项目中，从 `HFData` 与 `STARSS23` 原始数据开始，到最终整理成可用于 `dcase2024-SedHead / DCASE2024 SELD baseline` 训练输入的完整流程、关键目录和最终产物。

## 1. 原始数据来源

### 1.1 STARSS23 原始数据

原始目录：

- [`/data/zhuzhiyuan/starss23/STARSS23`](/data/zhuzhiyuan/starss23/STARSS23)

原始结构核心包括：

- `foa_dev/`：FOA 音频
- `metadata_dev/`：逐帧 CSV 标注
- `video_dev/`：视频

### 1.2 HFData 原始数据

HFData 最初经过下载和整理后，被扁平化到：

- [`/data/zhuzhiyuan/starss23/SpatialQA_hf/audio`](/data/zhuzhiyuan/starss23/SpatialQA_hf/audio)
- [`/data/zhuzhiyuan/starss23/SpatialQA_hf/visual`](/data/zhuzhiyuan/starss23/SpatialQA_hf/visual)
- [`/data/zhuzhiyuan/starss23/SpatialQA_hf/json`](/data/zhuzhiyuan/starss23/SpatialQA_hf/json)

其中：

- 音频按父目录时间重命名为 `YYYYMMDD_HHMMSS.wav`
- 视频按父目录时间重命名为 `YYYYMMDD_HHMMSS.mp4`
- JSON 按父目录时间重命名为 `YYYYMMDD_HHMMSS.json`

## 2. HFData 预处理

### 2.1 角度规范化

HFData JSON 中的 `heading` 被统一归一到 `[-180, 180)`：

- 脚本：
  [`normalize_hf_json_headings.py`](/data/zhuzhiyuan/starss23/code/normalize_hf_json_headings.py:1)

### 2.2 类别映射与清理

为便于训练，对 HFData 原始事件类别进行了人工归并，并生成模板：

- 模板：
  [`spatialqa_event_mapping_template.csv`](/data/zhuzhiyuan/starss23/code/spatialqa_event_mapping_template.csv:1)

如果某些类别被标记为 `0`，则对应 JSON 条目会被删除。最终大小写也做了统一。

映射统计说明：

- [`spatialqa_event_mapping_summary.md`](/data/zhuzhiyuan/starss23/code/spatialqa_event_mapping_summary.md:1)

### 2.3 JSON 转 STARSS23/DCASE 风格 CSV

将 HFData JSON 转成和 STARSS23 一致的逐帧 CSV：

- 脚本：
  [`convert_spatialqa_json_to_dcase_csv.py`](/data/zhuzhiyuan/starss23/code/convert_spatialqa_json_to_dcase_csv.py:1)

输出目录：

- 原始整段 CSV：
  [`/data/zhuzhiyuan/starss23/SpatialQA_hf_csv`](/data/zhuzhiyuan/starss23/SpatialQA_hf_csv)
- 切段后 CSV：
  [`/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented_csv`](/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented_csv)

CSV 格式为：

```text
frame_idx,class_id,track_id,azimuth,elevation,distance_cm
```

### 2.4 `up/down` 粗定位标签策略

HFData 中仅有粗粒度上下方向信息的事件，保留 `elevation = inf / -inf`，但：

- 这些事件仍参与 `SED` 监督
- 不参与 `DOA / distance` 回归 loss

对应代码修改在：

- [`cls_feature_class.py`](/data/zhuzhiyuan/starss23/dcase2024-SedHead/cls_feature_class.py:1)
- [`seldnet_model.py`](/data/zhuzhiyuan/starss23/dcase2024-SedHead/seldnet_model.py:1)

## 3. STARSS23 切段与重采样

### 3.1 固定 20 秒切段

STARSS23 被切成固定 20 秒片段，音频、视频、CSV 同步切分：

- 脚本：
  [`segment_starss23_fixed20s.py`](/data/zhuzhiyuan/starss23/code/segment_starss23_fixed20s.py:1)

输出目录：

- [`/data/zhuzhiyuan/starss23/STARSS23_20s`](/data/zhuzhiyuan/starss23/STARSS23_20s)

### 3.2 重采样到 16kHz

为了统一训练采样率，将切段后的 STARSS23 重采样到 `16kHz`：

输出目录：

- [`/data/zhuzhiyuan/starss23/STARSS23_20s_16k`](/data/zhuzhiyuan/starss23/STARSS23_20s_16k)

## 4. HFData 切段

HFData 不是固定 20 秒切，而是按 `prepare_real_foa_to_dcase.py` 的事件驱动逻辑切段：

- 最短约 `5s`
- 最长约 `20s`
- 优先在事件空白区或事件边界切

脚本：

- [`segment_spatialqa_event_based.py`](/data/zhuzhiyuan/starss23/code/segment_spatialqa_event_based.py:1)

输出目录：

- [`/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented`](/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented)

结构为：

- `audio/`
- `visual/`
- `json/`

后续再转换出切段后的 CSV：

- [`/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented_csv`](/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented_csv)

## 5. 合并训练集

### 5.1 合并后的 16k 训练根目录

最终合并出的主训练数据根目录为：

- [`/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k`](/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k)

结构包括：

- `foa_dev/`
- `metadata_dev/`
- `video_dev/`
- `class_mapping.json`
- `split_manifest.json`
- `source_manifest.json`

### 5.2 当前合并数据的 split 策略

当前主合并集使用的策略是：

- `train` = `STARSS23 train + HF train + HF test`
- `valid` = `HF eval`
- `test` = `STARSS23 test`

这样：

- 训练时尽量多利用 HFData
- `STARSS23 test` 保持为最终 benchmark 分数集合

对应文件：

- [`split_manifest.json`](/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k/split_manifest.json:1)
- [`source_manifest.json`](/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k/source_manifest.json:1)
- [`merge_summary.json`](/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k/merge_summary.json:1)

### 5.3 STARSS23-only 独立 split

为了做纯 STARSS23 实验，还单独对 `STARSS23_20s_16k` 做了 manifest：

- `train`：从 `STARSS23 train` 中按 `90%`
- `valid`：从 `STARSS23 train` 中按 `10%`
- `test`：官方 `STARSS23 test`

对应文件：

- [`/data/zhuzhiyuan/starss23/STARSS23_20s_16k/split_manifest.json`](/data/zhuzhiyuan/starss23/STARSS23_20s_16k/split_manifest.json:1)
- [`/data/zhuzhiyuan/starss23/STARSS23_20s_16k/source_manifest.json`](/data/zhuzhiyuan/starss23/STARSS23_20s_16k/source_manifest.json:1)

## 6. 最终训练输入路径

### 6.1 原始数据根目录

#### 纯 STARSS23

- [`/data/zhuzhiyuan/starss23/STARSS23_20s_16k`](/data/zhuzhiyuan/starss23/STARSS23_20s_16k)

#### STARSS23 + HFData

- [`/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k`](/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k)

### 6.2 实际训练时使用的特征目录

模型训练时真正读取的是 `feat_label_dir` 下的 `.npy` 特征和标签。

#### SedHead

- `247` / `STARSS23-only` / `audio-only`
  - [`/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task247`](/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task247)
- `248` / `STARSS23+HF` / `audio-only`
  - [`/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task248`](/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task248)
- `249` / `STARSS23-only` / `audio-visual`
  - [`/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task249_av`](/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task249_av)
- `250` / `STARSS23+HF` / `audio-visual`
  - [`/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task250_av`](/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task250_av)

#### DCASE2024 / SELDnet baseline

为避免重复提特征，baseline 直接复用已有特征目录：

- `251` / `STARSS23-only` / `audio-only baseline`
  - 复用 `247`
- `252` / `STARSS23+HF` / `audio-only baseline`
  - 复用 `248`
- `253` / `STARSS23-only` / `audio-visual baseline`
  - 复用 `249`
- `254` / `STARSS23+HF` / `audio-visual baseline`
  - 复用 `250`

## 7. 当前实验矩阵

### 7.1 你的 SedHead

- `247`：audio-only，STARSS23-only
- `248`：audio-only，STARSS23 + HF
- `249`：audio-visual，STARSS23-only
- `250`：audio-visual，STARSS23 + HF

### 7.2 原始 DCASE2024 / SELDnet baseline

- `251`：audio-only，STARSS23-only
- `252`：audio-only，STARSS23 + HF
- `253`：audio-visual，STARSS23-only
- `254`：audio-visual，STARSS23 + HF

## 8. 最终评测分数

当前所有实验的最终 benchmark 都应看：

- `STARSS23 test`

也就是：

- 对纯 STARSS23：`STARSS23_20s_16k/split_manifest.json` 中的 `test`
- 对合并数据：`merged_seld_foa_starss23_spatialqa_20s_16k/split_manifest.json` 中的 `test`

## 9. 相关代码与说明文件

- 数据说明：
  [`/data/zhuzhiyuan/starss23/code/README.md`](/data/zhuzhiyuan/starss23/code/README.md:1)
- 模型参数：
  [`parameters.py`](/data/zhuzhiyuan/starss23/dcase2024-SedHead/parameters.py:1)
- 训练入口：
  [`train_seldnet.py`](/data/zhuzhiyuan/starss23/dcase2024-SedHead/train_seldnet.py:1)
- 特征提取入口：
  [`batch_feature_extraction.py`](/data/zhuzhiyuan/starss23/dcase2024-SedHead/batch_feature_extraction.py:1)

