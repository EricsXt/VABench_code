# `code/` 目录说明

本文档介绍 `/data/zhuzhiyuan/starss23/code` 目录中的 3 个 Python 脚本，包括它们的作用、输入、输出，以及推荐的使用顺序。

## 目录中的脚本

- `prepare_real_foa_to_dcase.py`
- `check_real_annotations.py`
- `analyze_prepared_real_foa_dataset.py`

这 3 个脚本对应一个比较清晰的流程：

1. 先检查原始标注质量
2. 再把原始数据整理成 DCASE 风格数据集
3. 最后对整理后的数据集做统计分析

## 1. `prepare_real_foa_to_dcase.py`

### 作用

这是最核心的预处理脚本，用来把原始 RealFOA 音频和 JSON 标注转换成 DCASE 风格的数据集。它会完成以下工作：

- 读取原始 FOA 音频和标注
- 将原始类别名映射到 FSD 风格类别
- 按事件时间切分音频片段
- 将音频重采样到 16 kHz
- 为每个片段生成对应的帧级标注 CSV
- 生成类别映射、切分清单、manifest 等辅助文件

### 输入

主要参数如下：

- `--audio-root`
  原始音频根目录
  默认值：
  `/apdcephfs_cq10/share_1603164/user/schmittzhu/data/RealFOA`
- `--anno-root`
  原始标注根目录
  默认值：
  `/apdcephfs_cq10/share_1603164/user/schmittzhu/data/metadata/real_anno`
- `--fsd-vocab-csv`
  FSD50K 词表 CSV，用来校验和索引目标类别
- `--output-root`
  输出数据集根目录
- `--target-sr`
  目标采样率，默认 `16000`
- `--label-hop-seconds`
  帧级标签时间步长，默认 `0.1`
- `--min-seconds`
  最短切片长度，默认 `5.0`
- `--max-seconds`
  最长切片长度，默认 `20.0`
- `--target-seconds`
  目标切片长度，默认 `12.0`
- `--merge-gap-seconds`
  相邻事件合并成同一时间岛的间隔阈值，默认 `1.5`
- `--split-strategy`
  数据集划分方式，支持 `hash` 和 `all_train`

原始目录结构要求大致如下：

```text
audio-root/
  20260313/
    VID_20260313_120110/
      aligned_foa_bformat.wav

anno-root/
  20260313/
    VID_20260313_120110/
      xxx.json
```

标注 JSON 中，脚本主要依赖这些字段：

- `annotations`
- `event_name`
- `start_time`
- `end_time`
- `location`
- 可选：`location_end`
- 可选：`moving`

### 处理逻辑

脚本的大致逻辑是：

- 在音频目录和标注目录中按 `20*/VID_*` 结构逐个配对
- 跳过缺失音频、缺失 JSON、事件名缺失、时间非法、无法映射类别等无效样本
- 将原始类别映射到统一的 FSD 类别
- 把 `location`、`location_end` 等位置信息转成 DCASE 所需的：
  - `azimuth`
  - `elevation`
  - `distance_cm`
- 根据事件时间把长录音切成多个片段
- 每个片段导出一个 `wav` 和一个 `csv`
- 按录音级别分到 `train`、`valid`、`test`

### 输出

在 `output-root` 下会生成如下内容：

```text
output-root/
  class_mapping.json
  split_manifest.json
  raw_to_fsd_mapping.csv
  segment_manifest.csv
  prep_summary.json
  train.jsonl
  valid.jsonl
  test.jsonl
  foa_dev/
    realfoa/
      *.wav
  metadata_dev/
    realfoa/
      *.csv
  split_lists/
    train.txt
    valid.txt
    test.txt
```

各输出文件含义如下：

- `foa_dev/realfoa/*.wav`
  处理后的音频片段
- `metadata_dev/realfoa/*.csv`
  与每个音频片段对应的帧级标注，格式为：
  `frame_idx,class_id,track_id,azimuth_deg,elevation_deg,distance_cm`
- `class_mapping.json`
  类别名称和类别 ID 的映射关系
- `split_manifest.json`
  各数据集划分下包含哪些片段
- `raw_to_fsd_mapping.csv`
  原始类别到目标类别的映射记录
- `segment_manifest.csv`
  每个导出片段的来源、起止时间、时长、事件数、标注行数
- `train.jsonl`、`valid.jsonl`、`test.jsonl`
  更适合程序读取的 manifest
- `prep_summary.json`
  预处理汇总统计信息

### 示例命令

```bash
python3 prepare_real_foa_to_dcase.py \
  --audio-root /path/to/RealFOA \
  --anno-root /path/to/real_anno \
  --fsd-vocab-csv /path/to/final_vocabulary.csv \
  --output-root /path/to/prepared_dataset \
  --overwrite
```

## 2. `check_real_annotations.py`

### 作用

这个脚本用于检查原始 RealFOA 标注质量。它不会导出训练数据，而是输出一组检查报告，帮助判断原始数据是否存在问题。

它主要检查：

- 音频目录和标注目录是否一一对应
- 每个标注目录下是否恰好有 1 个 JSON
- 标注事件的时间是否合法
- 事件是否缺少位置字段
- 位置字段是否能被解析成空间信息
- 每段音频中被标注覆盖的比例

### 输入

主要参数如下：

- `--audio-root`
  原始音频根目录
  默认值：
  `/apdcephfs_cq10/share_1603164/user/schmittzhu/data/RealFOA`
- `--anno-root`
  原始标注根目录
  默认值：
  `/apdcephfs_cq10/share_1603164/user/schmittzhu/data/metadata/real_anno`
- `--output-dir`
  检查报告输出目录
  默认值：
  `check_reports`
- `--verbose`
  是否打印逐条处理进度

每条录音默认会检查：

- 音频文件：`aligned_foa_bformat.wav`
- 标注文件：目录下唯一的 `*.json`

### 处理逻辑

脚本会：

- 扫描两个根目录下的 `20*/VID_*` 子目录
- 检查哪些录音只存在于音频侧，哪些只存在于标注侧
- 校验每个标注目录下 JSON 文件数量是否为 1
- 读取 wav 时长
- 读取 JSON 中的 `annotations`
- 对每个事件检查：
  - `event_name` 是否存在
  - `start_time`、`end_time` 是否可解析
  - 时长是否大于 0
  - 起止时间是否超出音频范围
  - `location` 是否存在
  - `location` 是否能解析为空间方向或位置
- 根据有效标注估算：
  - 类别分布
  - 方向分布
  - 仰角分布
  - 距离分布
  - 标注覆盖率

### 输出

输出目录结构如下：

```text
output-dir/
  summary.json
  per_audio_summary.csv
  invalid_events.csv
  missing_annotation_details.csv
  parse_errors.csv
  class_distribution.csv
  location_type_distribution.csv
  heading_distribution.csv
  heading_bins.csv
  elevation_distribution.csv
  elevation_bins.csv
  distance_distribution.csv
  distance_bins.csv
```

各输出文件含义如下：

- `summary.json`
  总体汇总信息，包括录音数、配对数、无效事件数、缺失情况、覆盖率等
- `per_audio_summary.csv`
  每段录音一行，记录音频时长、标注数、无效事件数、覆盖时长、覆盖率等
- `invalid_events.csv`
  所有事件级错误和异常
- `missing_annotation_details.csv`
  缺失音频目录、缺失标注目录、JSON 数量不对等问题
- `parse_errors.csv`
  无法读取的 wav 或 JSON 文件
- `class_distribution.csv`
  原始类别的出现次数和总时长
- `location_type_distribution.csv`
  各种位置描述类型的统计
- `heading_distribution.csv` / `heading_bins.csv`
  朝向角及其分桶统计
- `elevation_distribution.csv` / `elevation_bins.csv`
  仰角及其分桶统计
- `distance_distribution.csv` / `distance_bins.csv`
  距离及其分桶统计

### 示例命令

```bash
python3 check_real_annotations.py \
  --audio-root /path/to/RealFOA \
  --anno-root /path/to/real_anno \
  --output-dir check_reports \
  --verbose
```

## 3. `analyze_prepared_real_foa_dataset.py`

### 作用

这个脚本用于分析已经整理好的 DCASE 风格数据集。也就是说，它的输入不是原始数据，而是 `prepare_real_foa_to_dcase.py` 的输出结果。

它主要统计：

- 每个片段的时长分布
- 每个类别的总标注时长
- 方位角分布
- 仰角分布
- 距离分布
- 未知距离、未知仰角、无穷仰角的数量

### 输入

主要参数如下：

- `--dataset-root`
  已整理数据集根目录
  默认值：
  `/apdcephfs_cq10/share_1603164/user/schmittzhu/code/DCASE2024_seld_baseline/prepared_datasets/real_foa_fsd50k_16k`
- `--output-dir`
  输出统计结果目录
  默认值：
  `prepared_real_foa_profile`
- `--workers`
  并行处理线程数

输入目录中至少需要这些文件：

- `class_mapping.json`
- `segment_manifest.csv`
- `metadata_dev/**/*.csv`

### 处理逻辑

脚本会：

- 读取 `class_mapping.json`
- 读取 `segment_manifest.csv`
- 扫描 `metadata_dev` 下所有 CSV
- 按 stem 把每个 CSV 对应回它的片段信息
- 统计片段时长，并分到 1 秒宽的区间
- 统计每个类别的总持续时间
- 将方位角按 20 度分桶
- 将仰角按 10 度分桶
- 将距离按 0.5 米分桶
- 统计未知距离、未知仰角、无穷仰角的行数

### 输出

输出目录结构如下：

```text
output-dir/
  summary.json
  clip_duration_bins_1s.csv
  class_total_duration_sec.csv
  azimuth_bins_20deg.csv
  elevation_bins_10deg.csv
  distance_bins_0p5m.csv
```

各输出文件含义如下：

- `summary.json`
  数据集总体统计信息
- `clip_duration_bins_1s.csv`
  片段时长分布
- `class_total_duration_sec.csv`
  每个映射后类别的总持续时间
- `azimuth_bins_20deg.csv`
  方位角分布
- `elevation_bins_10deg.csv`
  仰角分布
- `distance_bins_0p5m.csv`
  距离分布

### 示例命令

```bash
python3 analyze_prepared_real_foa_dataset.py \
  --dataset-root /path/to/prepared_dataset \
  --output-dir prepared_real_foa_profile
```

## 推荐使用顺序

建议按照下面顺序使用：

1. 先运行 `check_real_annotations.py`
   检查原始音频和标注是否有明显问题
2. 再运行 `prepare_real_foa_to_dcase.py`
   生成训练所需的 DCASE 风格数据
3. 最后运行 `analyze_prepared_real_foa_dataset.py`
   对已经导出的数据集做统计分析

## 总结

- `prepare_real_foa_to_dcase.py` 是“生成数据”的核心脚本
- `check_real_annotations.py` 是“检查原始数据”的质检脚本
- `analyze_prepared_real_foa_dataset.py` 是“分析生成结果”的统计脚本

目前这份中文 README 暂时写在：

`/data/zhuzhiyuan/starss23/README_code.md`

原因是 `/data/zhuzhiyuan/starss23/code` 目录当前仍然是 `root:root`，我没有权限直接写入 `code/README.md`。如果你把这个目录改成 `zhuzhiyuan:zhuzhiyuan`，我可以立刻把这份内容放到 `/data/zhuzhiyuan/starss23/code/README.md`。
