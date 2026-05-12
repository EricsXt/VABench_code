#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/zhuzhiyuan/starss23/dcase2024-SedHead"
TASK_ID="248"
JOB_ID="starss23_plus_hf_4gpu_run01"
GPU_IDS="4,5,6,7"
FEAT_DIR="/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task248"
EXPECTED_COUNT=3978
LOG_DIR="$REPO_DIR/run_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_ID}.log"

exec >> "$LOG_FILE" 2>&1

echo "[$(date '+%F %T')] task $TASK_ID launcher started"

feature_ready() {
python3 - <<'PY'
from pathlib import Path
feat_dir = Path("/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task248")
expected = 3978
checks = [
    feat_dir / "foa_dev",
    feat_dir / "foa_dev_norm",
    feat_dir / "foa_dev_adpit_label",
]
if not all(p.exists() and p.is_dir() for p in checks):
    raise SystemExit(1)
feature_count = sum(1 for _ in (feat_dir / "foa_dev").glob("*.npy"))
norm_count = sum(1 for _ in (feat_dir / "foa_dev_norm").glob("*.npy"))
label_count = sum(1 for _ in (feat_dir / "foa_dev_adpit_label").glob("*.npy"))
stats_ok = (feat_dir / "foa_dev_dataset_stats.json").exists()
wts_ok = (feat_dir / "foa_wts").exists()
ok = (
    feature_count == expected and
    norm_count == expected and
    label_count == expected and
    stats_ok and
    wts_ok
)
raise SystemExit(0 if ok else 1)
PY
}

wait_for_extractor() {
  while pgrep -af "python3 batch_feature_extraction.py ${TASK_ID}" >/dev/null; do
    echo "[$(date '+%F %T')] task $TASK_ID feature extraction still running; waiting 60s"
    sleep 60
  done
}

wait_for_gpus() {
  while python3 - <<'PY'
import subprocess

gpu_ids = {4, 5, 6, 7}
gpu_info = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,uuid",
    "--format=csv,noheader,nounits",
], text=True)
uuid_to_idx = {}
for line in gpu_info.strip().splitlines():
    idx, uuid = [part.strip() for part in line.split(",", 1)]
    uuid_to_idx[uuid] = int(idx)

busy = set()
try:
    app_info = subprocess.check_output([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ], text=True, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    app_info = ""

for line in app_info.strip().splitlines():
    uuid, _pid = [part.strip() for part in line.split(",", 1)]
    idx = uuid_to_idx.get(uuid)
    if idx in gpu_ids:
        busy.add(idx)

raise SystemExit(0 if busy else 1)
PY
  do
    echo "[$(date '+%F %T')] GPUs ${GPU_IDS} busy; waiting 120s"
    sleep 120
  done
}

wait_for_cpu() {
  while python3 - <<'PY'
import subprocess

heavy_patterns = (
    "scripts/pretrain.py",
    "lbm_eval.evaluate",
)

out = subprocess.check_output(
    ["ps", "-eo", "pcpu,args", "--no-headers"],
    text=True,
)

heavy = []
for line in out.splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split(None, 1)
    if len(parts) != 2:
        continue
    pcpu, args = parts
    if any(pat in args for pat in heavy_patterns):
        try:
            cpu = float(pcpu)
        except ValueError:
            continue
        if cpu >= 20.0:
            heavy.append((cpu, args))

raise SystemExit(0 if heavy else 1)
PY
  do
    echo "[$(date '+%F %T')] heavy CPU jobs detected; waiting 120s"
    sleep 120
  done
}

cd "$REPO_DIR"

wait_for_extractor
if ! feature_ready; then
  echo "[$(date '+%F %T')] features incomplete; running batch_feature_extraction.py $TASK_ID"
  python3 batch_feature_extraction.py "$TASK_ID"
else
  echo "[$(date '+%F %T')] features ready"
fi

wait_for_cpu
wait_for_gpus
echo "[$(date '+%F %T')] starting training on GPUs ${GPU_IDS}"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 train_seldnet.py "$TASK_ID" "$JOB_ID"
