# 复现指南：从数据处理到训练与评测

本文档面向一个**完全没有参与过前期工作的新人**，目标是让对方能够从原始 `STARSS23` 和 `HFData / SpatialQA` 数据开始，复现当前这套实验流程，并理解最终评测结果的含义。

本文档覆盖：

- 原始数据应该放在哪里
- HFData 和 STARSS23 分别如何处理
- 最终训练集是如何组织的
- 训练一共有哪些 task，分别对应什么实验
- 训练日志、模型、评测结果保存在哪里
- 每一个评测指标的定义、公式和解释

如果只想看更简短的概览，请先看根目录的 [`README.md`](README.md)。

---

## 1. 项目目标

当前项目要比较两类模型、两种模态、两种数据配置：

### 1.1 模型类型

1. 原始 `DCASE2024 SELD baseline`
2. 修改后的 `dcase2024-SedHead`

其中：

- `baseline` 是原始 SELDnet / Multi-ACCDOA 方案
- `SedHead` 是在此基础上增加了一个显式 `SED head`

### 1.2 模态

1. `audio-only`
2. `audio-visual`

### 1.3 数据配置

1. 只用 `STARSS23`
2. 用 `STARSS23 + HFData`

最初整理了 8 个实验：

| Task | 模型 | 模态 | 训练数据 |
|---|---|---|---|
| 247 | SedHead | audio-only | STARSS23-only |
| 248 | SedHead | audio-only | STARSS23 + HF |
| 249 | SedHead | audio-visual | STARSS23-only |
| 250 | SedHead | audio-visual | STARSS23 + HF |
| 251 | baseline | audio-only | STARSS23-only |
| 252 | baseline | audio-only | STARSS23 + HF |
| 253 | baseline | audio-visual | STARSS23-only |
| 254 | baseline | audio-visual | STARSS23 + HF |

所有最终 benchmark 分数都统一在 `STARSS23 test` 上汇报。

后来为了把 merged 数据的验证集改成 `STARSS23 val-internal`，又新增了 4 个重跑 task：

| Task | 模型 | 模态 | 训练数据 | split 口径 |
|---|---|---|---|---|
| 255 | SedHead | audio-only | STARSS23 + HF | `train = STARSS23 train-internal + HF train`, `valid = STARSS23 val-internal`, `test = STARSS23 test` |
| 256 | baseline | audio-only | STARSS23 + HF | 同上 |
| 257 | SedHead | audio-visual | STARSS23 + HF | 同上，但过滤无视频样本 |
| 258 | baseline | audio-visual | STARSS23 + HF | 同上，但过滤无视频样本 |

---

## 2. 目录说明

当前实际工作目录主要有两块：

### 2.1 数据与真实训练代码

- 主目录：`/data/zhuzhiyuan/starss23`
- 训练代码目录：`/data/zhuzhiyuan/starss23/dcase2024-SedHead`
- 数据处理脚本目录：`/data/zhuzhiyuan/starss23/code`

### 2.2 GitHub 镜像目录

- `/home/zhuzhiyuan/VABench_code_upload`

这个 GitHub 镜像目录保存：

- 数据处理脚本
- 当前使用的训练代码
- 各种说明文档

也就是说：

- `/data/...` 是**实际运行环境**
- `/home/.../VABench_code_upload` 是**对外同步到 GitHub 的版本**

---

## 3. 原始数据准备

### 3.1 STARSS23 原始数据

原始 STARSS23 数据放在：

- `/data/zhuzhiyuan/starss23/STARSS23`

其关键结构是：

- `foa_dev/`：四通道 FOA 音频
- `metadata_dev/`：逐帧 CSV 标注
- `video_dev/`：同步视频

注意：

- STARSS23 原始数据中有一部分音频没有对应视频
- 这不是本项目处理错误，而是原始数据本身就存在
- 缺视频主要影响 `audio-visual` 训练
- 但最终 `STARSS23 test` 用于 benchmark 的样本是有视频的

### 3.2 HFData / SpatialQA 原始数据

HFData 最终被整理到：

- `/data/zhuzhiyuan/starss23/SpatialQA_hf/audio`
- `/data/zhuzhiyuan/starss23/SpatialQA_hf/visual`
- `/data/zhuzhiyuan/starss23/SpatialQA_hf/json`

这里已经不是最原始下载形态，而是整理后的扁平版本：

- 音频文件命名为 `YYYYMMDD_HHMMSS.wav`
- 视频文件命名为 `YYYYMMDD_HHMMSS.mp4`
- 标注文件命名为 `YYYYMMDD_HHMMSS.json`

这些名字来自原始目录的父级时间戳，如：

- `VID_20260313_120110`

会被整理成：

- `20260313_120110.wav`
- `20260313_120110.mp4`
- `20260313_120110.json`

---

## 4. HFData 数据处理流程

HFData 的处理步骤比 STARSS23 更复杂，因为它最开始不是 DCASE / STARSS23 风格的 SELD 数据。

### 4.1 角度归一化

HFData 的 JSON 中，航向角 `heading` 采用：

- `front = 0`
- `left = +90`

这和 STARSS23 的 azimuth 定义是一致的，但还需要把角度范围统一到：

- `[-180, 180)`

使用脚本：

- [`code/normalize_hf_json_headings.py`](code/normalize_hf_json_headings.py)

作用：

- 把 `270` 变成 `-90`
- 把 `315` 变成 `-45`
- 把所有 `heading` 统一成与 STARSS23 兼容的表示

### 4.2 类别映射模板

HFData 原始事件名比 STARSS23 多，也更杂。

因此需要先把所有事件名导出成模板，再人工映射到更适合训练的类别集合。

模板文件：

- [`code/spatialqa_event_mapping_template.csv`](code/spatialqa_event_mapping_template.csv)

字段包括：

- `raw_event_name`
- `target_class`
- `source_json_files`
- `notes`

其中：

- `raw_event_name`：JSON 里的原始事件名
- `target_class`：你希望归并到的训练类别
- `source_json_files`：这个事件名出现在哪些 JSON 里
- `notes`：人工备注

### 4.3 删除映射为 0 的事件

如果在 `target_class` 中把某类标为 `0`，表示：

- 这个事件不参与训练
- 对应 JSON 条目会被删除

这一步做过实际清理，因此当前数据里已经移除了这些无效事件。

类别映射汇总说明：

- [`code/spatialqa_event_mapping_summary.md`](code/spatialqa_event_mapping_summary.md)

### 4.4 事件驱动切段

HFData 不按固定 20 秒切段，而是按事件驱动逻辑切段。

使用脚本：

- [`code/segment_spatialqa_event_based.py`](code/segment_spatialqa_event_based.py)

切段原则：

- 最短大约 `5s`
- 最长大约 `20s`
- 优先在事件空白区切
- 优先在事件边界切
- 避免把强相关事件硬切断

输出目录：

- `/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented`

结构为：

- `audio/`
- `visual/`
- `json/`

### 4.5 JSON 转 STARSS23 风格 CSV

训练和评测不直接读取原始 JSON，而是读取 STARSS23/DCASE 风格的逐帧 CSV。

使用脚本：

- [`code/convert_spatialqa_json_to_dcase_csv.py`](code/convert_spatialqa_json_to_dcase_csv.py)

输出：

- 整段版：
  - `/data/zhuzhiyuan/starss23/SpatialQA_hf_csv`
- 切段版：
  - `/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented_csv`

CSV 格式：

```text
frame_idx,class_id,track_id,azimuth,elevation,distance_cm
```

解释：

- `frame_idx`：标签帧编号，步长为 `0.1s`
- `class_id`：类别编号
- `track_id`：同类声源实例编号
- `azimuth`：水平角
- `elevation`：垂直角
- `distance_cm`：距离，单位厘米

### 4.6 `up/down` 粗定位标签的特殊处理

HFData 中一部分事件只有：

- 在上方
- 或在下方

但没有精确仰角。

如果强行给一个具体仰角，会引入很大误差，因此这里采用的是：

- 保留 `elevation = inf / -inf`
- 训练时：
  - 保留 `SED` 监督
  - 跳过 `DOA / distance` 回归 loss
- 评测时：
  - 标准 SELD 指标不把它们当作精确定位参考
  - 另外单独统计 `up/down` 的粗垂直方向分数

相关代码：

- [`dcase2024-SedHead/cls_feature_class.py`](dcase2024-SedHead/cls_feature_class.py)
- [`dcase2024-SedHead/seldnet_model.py`](dcase2024-SedHead/seldnet_model.py)
- [`dcase2024-SedHead/SELD_evaluation_metrics.py`](dcase2024-SedHead/SELD_evaluation_metrics.py)
- [`dcase2024-SedHead/cls_compute_seld_results.py`](dcase2024-SedHead/cls_compute_seld_results.py)

这一步非常重要，因为它决定了：

- HFData 粗定位信息不会污染标准 DOA 回归
- 但又不会完全浪费这部分样本

---

## 5. STARSS23 数据处理流程

STARSS23 的原始形式已经比较接近训练需要的格式，但仍然做了两步处理：

### 5.1 固定 20 秒切段

使用脚本：

- [`code/segment_starss23_fixed20s.py`](code/segment_starss23_fixed20s.py)

作用：

- 将 STARSS23 按固定 `20s` 切分
- 保持音频、视频、CSV 三者同步切分

输出：

- `/data/zhuzhiyuan/starss23/STARSS23_20s`

### 5.2 统一到 16kHz

原始训练过程中发现：

- STARSS23 与 HFData 的采样率不一致
- 训练代码不会自动重采样

因此最终统一到：

- `16000 Hz`

最终目录：

- `/data/zhuzhiyuan/starss23/STARSS23_20s_16k`

这一步是训练能否正常进行的关键前提。

---

## 6. 合并训练集的构建

### 6.1 合并根目录

最终合并后的训练根目录为：

- `/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k`

目录结构：

- `foa_dev/`
- `metadata_dev/`
- `video_dev/`
- `class_mapping.json`
- `split_manifest.json`
- `source_manifest.json`
- `merge_summary.json`

这就是 `STARSS23 + HFData` 联合训练时真正使用的数据根目录。

### 6.2 split 策略

当前主合并集的**最新重跑口径**采用：

- `train = STARSS23 train-internal + HF train`
- `valid = STARSS23 val-internal`
- `test = STARSS23 test`

原因：

- 不再使用 `HF eval` 选 best epoch
- 用 `STARSS23` 自己的内部验证集做 early stopping 和调参
- 最终 benchmark 仍然只看 `STARSS23 test`

对应文件：

- [`split_manifest.json`](dcase2024-SedHead/../merged_seld_foa_starss23_spatialqa_20s_16k/split_manifest.json)

### 6.3 AV 专用 manifest

由于 `STARSS23 train` 中有 `60` 条样本没有视频，因此：

- `audio-only` 可直接用普通 `split_manifest.json`
- `audio-visual` 不能直接用同一份 train split

否则会出现：

- 音频样本有
- 视频特征没有
- 最终在 `DataGenerator` 中索引越界

因此当前真正使用的是两份新的 manifest：

- `audio-only`
  - `/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k/split_manifest_starss_internal_val.json`
- `audio-visual`
  - `/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k/split_manifest_starss_internal_val_av.json`

它的特点是：

- `audio-only`：
  - `train = 2766`
  - `valid = 79`
  - `test = 619`
- `audio-visual`：
  - `train = 2713`
  - `valid = 72`
  - `test = 619`

`audio-visual` 的 `valid` 比 `audio-only` 少 `7` 条，是因为：

- `STARSS23 val-internal` 中有 `7` 条没有视频
- AV 训练和验证必须过滤掉这些样本，否则 dataloader 会报错

这一步是 `257/258` 能正常训练的必要条件。

---

## 7. STARSS23-only 的独立 split

为了做纯 STARSS23 实验，还单独在：

- `/data/zhuzhiyuan/starss23/STARSS23_20s_16k`

中构建了 manifest。

策略：

- `train`：`STARSS23 train` 的 `90%`
- `valid`：`STARSS23 train` 的 `10%`
- `test`：官方 `STARSS23 test`

对应文件：

- `/data/zhuzhiyuan/starss23/STARSS23_20s_16k/split_manifest.json`

这样做的原因是：

- 不能直接拿 `STARSS23 test` 做 validation
- 否则最终 benchmark 会被污染

---

## 8. 最终训练时实际使用的数据路径

### 8.1 数据根目录

#### 纯 STARSS23

- `/data/zhuzhiyuan/starss23/STARSS23_20s_16k`

#### STARSS23 + HFData

- `/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k`

### 8.2 训练真正读取的特征目录

模型训练时读取的不是原始 `.wav/.mp4`，而是提取后的 `.npy` 特征缓存。

#### SedHead

- `247`：`/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task247`
- `248`：`/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task248`
- `249`：`/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task249_av`
- `250`：`/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task250_av`

#### baseline

为了避免重复提取特征，baseline 直接复用对应 SedHead 的特征目录：

- `251` 复用 `247`
- `252` 复用 `248`
- `253` 复用 `249`
- `254` 复用 `250`

这意味着：

- baseline 不需要重新提音频特征
- baseline 也不需要重新提视频特征

---

## 9. 特征提取

特征提取入口：

- [`dcase2024-SedHead/batch_feature_extraction.py`](dcase2024-SedHead/batch_feature_extraction.py)

### 9.1 audio-only

示例：

```bash
cd /data/zhuzhiyuan/starss23/dcase2024-SedHead
python3 batch_feature_extraction.py 247
python3 batch_feature_extraction.py 248
```

### 9.2 audio-visual

示例：

```bash
python3 batch_feature_extraction.py 249
python3 batch_feature_extraction.py 250
```

对 `audio-visual` 来说，会额外提取：

- 视频特征

其成本明显高于 `audio-only`。

### 9.3 常见注意事项

1. 特征目录如果已经完整存在，没必要重复提取
2. baseline task 应复用 SedHead 对应特征目录
3. merged AV 一定要保证视频特征完整
4. 如果只看目录数量没有增长，不一定是卡住，可能只是按顺序跳过已有文件

---

## 10. 训练

训练入口：

- [`dcase2024-SedHead/train_seldnet.py`](dcase2024-SedHead/train_seldnet.py)

### 10.1 基本命令

示例：

```bash
cd /data/zhuzhiyuan/starss23/dcase2024-SedHead
python3 train_seldnet.py 247 starss23_only_sedhead_audio_run01
```

### 10.2 当前 8 个 task 的意义

| Task | 模型 | 模态 | 数据 |
|---|---|---|---|
| 247 | SedHead | audio-only | STARSS23-only |
| 248 | SedHead | audio-only | STARSS23 + HF |
| 249 | SedHead | audio-visual | STARSS23-only |
| 250 | SedHead | audio-visual | STARSS23 + HF |
| 251 | baseline | audio-only | STARSS23-only |
| 252 | baseline | audio-only | STARSS23 + HF |
| 253 | baseline | audio-visual | STARSS23-only |
| 254 | baseline | audio-visual | STARSS23 + HF |

### 10.3 当前 batch size

当前训练配置里使用：

- `batch_size = 32`

这是**总 batch size**，不是每张卡各自 32。

例如：

- 一个任务用 2 张 GPU，则大约每卡分到 16
- 一个任务用 4 张 GPU，则大约每卡分到 8

### 10.4 epoch、step、early stopping

默认最大 epoch 数：

- `250`

日志中形如：

- `Train 001 33/82`

表示：

- 当前是第 `1` 个 epoch
- 这个 epoch 一共有 `82` 个 batch
- 当前跑到了第 `33` 个 batch

当前训练还开启了：

- `early stopping`
- `patience = 50`

含义：

- 如果验证集表现连续 `50` 个 epoch 没有刷新最优结果，就提前停止
- 因此并不一定真的跑满 `250 epoch`

---

## 11. 训练输出保存位置

### 11.1 日志

日志主要看：

- `dcase2024-SedHead/run_logs/`

例如：

- `247_auto.log`
- `248_auto.log`
- `249_auto.log`
- `250_auto.log`

如果是自动调度/守护进程拉起的任务，通常看 `*_auto.log` 即可。

当前这套训练**没有接入 `wandb`**。

也就是说：

- 训练过程主要通过日志文件观察
- 结果主要通过 `results_audio/` 或 `results_audio_visual/` 下的 JSON / CSV 查看

### 11.2 模型 checkpoint

保存目录：

- `dcase2024-SedHead/models_audio/`
- `dcase2024-SedHead/models_audio_visual/`

常见文件：

- `*_model.h5`
- `*_best_full_model.h5`

一般可理解为：

- `*_model.h5`：某轮保存的最佳 checkpoint
- `*_best_full_model.h5`：按 full validation 口径选出的最佳 checkpoint

### 11.3 评测结果

保存目录：

- `dcase2024-SedHead/results_audio/`
- `dcase2024-SedHead/results_audio_visual/`

每次评测结果目录中最重要的文件是：

- `metrics_summary.json`
- `classwise_metrics.csv`

`metrics_summary.json` 是最重要的总分摘要。

---

## 12. 训练后如何评测

### 12.1 训练结束后自动评测

当前训练脚本会在训练结束后自动对 `test split` 做一次最终评测。

也就是说：

- `STARSS23-only` 实验最终会在 `STARSS23 test` 上给分
- `STARSS23 + HF` 实验最终也会在 `STARSS23 test` 上给分

### 12.2 手动重新评测

也可以使用：

- [`dcase2024-SedHead/evaluate_dev_checkpoint.py`](dcase2024-SedHead/evaluate_dev_checkpoint.py)

例如：

```bash
python3 evaluate_dev_checkpoint.py 247 starss23_only_sedhead_audio_run01 --ckpt best_full
```

这会：

1. 读取对应 checkpoint
2. 在当前 task 的 `test split` 上重新跑评测
3. 输出 `metrics_summary.json` 和 `classwise_metrics.csv`

---

## 13. 评测指标解释

这一部分非常重要。下面按当前代码实际输出的指标来解释。

### 13.1 Test loss

这是测试阶段按训练损失函数计算得到的平均 loss。

它反映的是：

- 模型输出与训练目标之间的数值差异

但它不是最终 benchmark 指标。

原因：

- loss 受训练目标形式影响很大
- 不能直接代表最终检测/定位表现

因此：

- `test loss` 用来辅助看模型是否数值稳定
- 最终对比主要看 `SELD` 及其分解指标

### 13.2 ER：Error Rate

`ER` 是检测错误率，越小越好。

在代码中的定义：

\[
ER = \frac{S + D + I}{N_{ref} + \epsilon}
\]

其中：

- `S`：substitution，替换错误
- `D`：deletion，漏检
- `I`：insertion，误检
- `N_ref`：参考事件总数

直观理解：

- 如果模型经常漏掉事件、报错事件、把一个事件识别成另一个，就会导致 ER 变大

### 13.3 F：F-score

`F` 是检测的 F-score，越大越好。

当前实现中是 location-sensitive detection 的 F-score。代码中使用：

\[
F = \frac{TP}{TP + FP_{spatial} + 0.5(FP + FN) + \epsilon}
\]

这里：

- `TP`：检测到且空间上也匹配的事件
- `FP_spatial`：类别对了，但空间不够接近
- `FP`：多报
- `FN`：漏报

直观理解：

- 它不是纯分类 F1
- 它要求“事件检测”和“空间匹配”同时满足

### 13.4 LR：Localization Recall

`LR` 是 localization recall，越大越好。

代码中：

\[
LR = \frac{DE\_TP}{DE\_TP + DE\_FN + \epsilon}
\]

含义：

- 在所有应该被定位到的事件中，有多少真正被成功定位了

直观理解：

- 这个指标更强调“定位召回率”
- 如果模型经常找不到该定位的声源，LR 会下降

### 13.5 AngE_deg：Angular Error

`AngE_deg` 是平均角度误差，单位是度，越小越好。

代码中：

\[
AngE = \frac{\sum \text{angular errors}}{DE\_TP + \epsilon}
\]

直观理解：

- 只在成功匹配上的参考-预测对之间计算
- 衡量方位/空间方向到底偏了多少度

如果没有有效定位匹配，可能会得到 `NaN`。

### 13.6 AngAcc

日志里还会打印：

- `AngAcc`

它不是单独直接评测得到的，而是在输出时由 `AngE_deg` 换算：

\[
AngAcc = \max(0, 1 - \frac{AngE}{180})
\]

含义：

- 一个便于直观理解的角度准确度近似
- `AngE` 越小，`AngAcc` 越高

它更像是辅助显示量，不是主要 benchmark 指标。

### 13.7 DistE：Distance Error

`DistE` 是绝对距离误差，越小越好。

代码中：

\[
DistE = \frac{\sum \text{absolute distance errors}}{DE\_TP\_dist + \epsilon}
\]

只在带有距离监督、且成功匹配上的样本对之间计算。

### 13.8 RelDistE：Relative Distance Error

`RelDistE` 是相对距离误差，越小越好。

代码中：

\[
RelDistE = \frac{\sum \text{relative distance errors}}{DE\_TP\_dist + \epsilon}
\]

与 `DistE` 的区别：

- `DistE` 更像绝对误差
- `RelDistE` 更像归一化后的相对误差

在当前这套 DCASE2024 距离评测逻辑中，`RelDistE` 非常重要，因为它直接进入最终 `SELD` 早停指标。

### 13.9 SELD：总体早停/排序指标

这是最重要的综合指标。

在当前 `eval_dist=True` 的实现中，代码使用：

\[
SELD = \text{mean}\left(1-F,\ \frac{AngE}{180},\ RelDistE\right)
\]

也就是说，在当前 2024 距离评测模式下：

- 不直接用 `ER`
- 不直接用 `LR`
- 而是综合：
  - 检测缺陷：`1 - F`
  - 方向误差：`AngE / 180`
  - 相对距离误差：`RelDistE`

因此：

- `SELD` 越小越好

这也是训练时选最优 checkpoint 的核心排序指标。

### 13.10 UpDown Accuracy

这是专门为 HFData 中 `up/down` 粗定位标签增加的辅助指标。

它不是官方 STARSS23 标准指标。

定义：

\[
Accuracy = \frac{correct}{total\_gt}
\]

其中：

- `correct`：上下方向判断正确的样本数
- `total_gt`：所有粗定位 `up/down` 参考样本数

只回答一个问题：

- “这个粗粒度垂直方向判断，有多少比例答对了？”

### 13.11 UpDown Precision

定义：

\[
Precision = \frac{TP}{TP + FP + \epsilon}
\]

含义：

- 预测成某个上下方向的事件里，有多少是真的

### 13.12 UpDown Recall

定义：

\[
Recall = \frac{TP}{TP + FN + \epsilon}
\]

含义：

- 真实存在的粗垂直方向事件里，有多少被模型正确判断出来

### 13.13 UpDown F1

定义：

\[
F1 = \frac{TP}{TP + 0.5(FP + FN) + \epsilon}
\]

这等价于：

\[
F1 = \frac{2TP}{2TP + FP + FN}
\]

它综合考虑了：

- precision
- recall

比单独看 accuracy 更能避免类别不平衡带来的虚高。

### 13.14 为什么需要单独的 UpDown 指标

因为 HFData 中一部分标注只有：

- 在上面
- 在下面

但没有精确仰角。

如果直接把它们也塞进标准 DOA 角度误差评测，会产生非法数值或不合理比较。

因此采用双轨评测：

1. 标准 SELD 指标：
   - 只对精确定位样本起作用
2. `UpDown` 辅助指标：
   - 只评估粗垂直方向样本

这样：

- 保住了标准 benchmark 的干净性
- 也不浪费 HFData 的粗定位信息

---

## 14. 当前已经完成的结果

截至本文档编写时，以下 6 个实验已经跑完并有 test 分数：

| Task | 实验 | Test loss | Best epoch | ER | F | LR | AngE | Dist | RelDist | SELD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 247 | SedHead audio-only, STARSS23-only | 2.0600 | 249 | 0.308 | 0.587 | 0.622 | 9.03 | 0.22 | 0.12 | 0.24 |
| 248 | SedHead audio-only, STARSS23+HF | 0.9510 | 209 | 1.042 | 0.325 | 0.488 | 21.22 | 0.37 | 0.17 | 0.42 |
| 249 | SedHead audio-visual, STARSS23-only | 1.6634 | 219 | 0.274 | 0.628 | 0.644 | 9.76 | 0.25 | 0.11 | 0.23 |
| 251 | baseline audio-only, STARSS23-only | 0.1120 | 229 | 0.216 | 0.615 | 0.651 | 10.74 | 0.13 | 0.07 | 0.22 |
| 252 | baseline audio-only, STARSS23+HF | 0.0529 | 229 | 0.577 | 0.356 | 0.376 | 11.43 | 0.17 | 0.07 | 0.44 |
| 253 | baseline audio-visual, STARSS23-only | 0.1052 | 239 | 0.220 | 0.579 | 0.565 | 7.10 | 0.12 | 0.06 | 0.27 |

尚在运行中的实验：

- `250`：SedHead audio-visual, STARSS23 + HF
- `254`：baseline audio-visual, STARSS23 + HF

---

## 15. 当前结果怎么解读

### 15.1 只看 STARSS23-only audio-only

- `251` baseline：`SELD = 0.22`
- `247` SedHead：`SELD = 0.24`

当前是 baseline 略优。

### 15.2 只看 STARSS23-only audio-visual

- `253` baseline：`SELD = 0.27`
- `249` SedHead：`SELD = 0.23`

当前是 SedHead 略优。

### 15.3 只看 STARSS23+HF audio-only

- `252` baseline：`SELD = 0.44`
- `248` SedHead：`SELD = 0.42`

当前是 SedHead 略优。

如果使用新的 `STARSS23 val-internal` 口径，最终应该看的是新 task：

- `255` vs `256`：`STARSS23 + HF` 的 `audio-only`
- `257` vs `258`：`STARSS23 + HF` 的 `audio-visual`

---

## 16. 常见坑与排错建议

### 16.1 混合采样率不能直接训练

如果 STARSS23 和 HFData 采样率不同，训练代码不会自动纠正，必须先统一到 `16kHz`。

### 16.2 merged AV 不能直接用普通 manifest

如果 `audio-visual` 直接使用普通 merged `split_manifest.json`，会把无视频 STARSS23 train 样本也放进训练集，最终在 dataloader 中越界。

必须用：

- `split_manifest_starss_internal_val_av.json`

### 16.3 `inf/-inf` 不能直接用于标准 DOA 评测

它们只能：

- 训练时参与 SED
- 评测时单独进入 `UpDown` 指标

不能直接当成精确仰角参考。

### 16.4 baseline 不要重复提特征

旧的 `251-254` 复用 `247-250` 的特征目录；新的 `255-258` 也继续复用同一套特征目录，不要再额外提一次。

### 16.5 看日志时优先看 `*_auto.log`

如果同一个 task 同时存在：

- `*_train.log`
- `*_train_manual.log`
- `*_auto.log`

优先看：

- `*_auto.log`

因为它通常对应当前由自动调度/守护进程实际托管的任务。

---

## 17. 一条最简复现路线

如果一个新人只想按最短路径复现，顺序如下：

1. 准备原始 `STARSS23`
2. 用 [`code/segment_starss23_fixed20s.py`](code/segment_starss23_fixed20s.py) 切成 `20s`
3. 准备原始 `HFData`
4. 用 [`code/normalize_hf_json_headings.py`](code/normalize_hf_json_headings.py) 统一角度
5. 填写 [`code/spatialqa_event_mapping_template.csv`](code/spatialqa_event_mapping_template.csv)
6. 用 [`code/segment_spatialqa_event_based.py`](code/segment_spatialqa_event_based.py) 做事件驱动切段
7. 用 [`code/convert_spatialqa_json_to_dcase_csv.py`](code/convert_spatialqa_json_to_dcase_csv.py) 转成 CSV
8. 构建 merged 训练根目录
9. 统一到 `16kHz`
10. 跑 `batch_feature_extraction.py`
11. 跑 `train_seldnet.py`
12. 到 `results_audio/` 或 `results_audio_visual/` 下查看 `metrics_summary.json`

---

## 18. 最后说明

这个项目现在已经不是“原始公开 baseline 直接跑一下”的状态，而是一套**带有实际工程修正**的训练与评测流程，主要修正包括：

- HFData 类别映射与清理
- HFData 角度归一化
- HFData 事件驱动切段
- `up/down` 粗定位标签保留但不参与精确 DOA 回归
- 增加 `UpDown` 辅助评测
- merged AV 专用 manifest
- AV dataloader 越界修复
- baseline 特征复用

因此，**要复现当前结果，必须复现的不只是训练命令本身，更是前面的数据组织和评测口径**。

如果后续还有新的最终分数（尤其是 `250/254`），建议直接把本文件第 14 节的结果表继续补全。
