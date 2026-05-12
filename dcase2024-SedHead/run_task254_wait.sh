#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/zhuzhiyuan/starss23/dcase2024-SedHead"
TASK_ID="254"
JOB_ID="starss23_plus_hf_4gpu_av_baseline_run01"
GPU_IDS="4,5,6,7"
LOG_DIR="$REPO_DIR/run_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${JOB_ID}.log"

exec >> "$LOG_FILE" 2>&1

echo "[$(date '+%F %T')] task $TASK_ID launcher started"

feature_ready() {
python3 - <<'PY'
from pathlib import Path
feat_dir = Path("/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task250_av")
checks = [
    feat_dir / "foa_dev",
    feat_dir / "foa_dev_norm",
    feat_dir / "foa_dev_adpit_label",
    feat_dir / "video_dev",
    feat_dir / "foa_dev_dataset_stats.json",
    feat_dir / "foa_wts",
]
if not all(p.exists() for p in checks):
    raise SystemExit(1)
video_count = sum(1 for _ in (feat_dir / "video_dev").glob("*.npy"))
raise SystemExit(0 if video_count == 3918 else 1)
PY
}

wait_for_cpu() {
  while python3 - <<'PY'
import subprocess
patterns = ("scripts/pretrain.py", "lbm_eval.evaluate")
out = subprocess.check_output(["ps", "-eo", "pcpu,args", "--no-headers"], text=True)
heavy = False
for line in out.splitlines():
    parts = line.strip().split(None, 1)
    if len(parts) != 2:
        continue
    pcpu, args = parts
    if any(p in args for p in patterns):
        try:
            cpu = float(pcpu)
        except ValueError:
            continue
        if cpu >= 20.0:
            heavy = True
            break
raise SystemExit(0 if heavy else 1)
PY
  do
    echo "[$(date '+%F %T')] heavy CPU jobs detected; waiting 120s"
    sleep 120
  done
}

wait_for_gpus() {
  while python3 - <<'PY'
import subprocess
gpu_ids = {4,5,6,7}
gpu_info = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"], text=True)
uuid_to_idx = {}
for line in gpu_info.strip().splitlines():
    idx, uuid = [part.strip() for part in line.split(",", 1)]
    uuid_to_idx[uuid] = int(idx)
try:
    app_info = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError:
    app_info = ""
busy = set()
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

cd "$REPO_DIR"
if ! feature_ready; then
  echo "[$(date '+%F %T')] shared AV features from task250 not ready; waiting 120s"
  until feature_ready; do sleep 120; done
fi
echo "[$(date '+%F %T')] shared AV features ready"
wait_for_cpu
wait_for_gpus
echo "[$(date '+%F %T')] starting training on GPUs ${GPU_IDS}"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 train_seldnet.py "$TASK_ID" "$JOB_ID"
