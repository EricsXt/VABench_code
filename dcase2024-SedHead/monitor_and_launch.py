#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


HEAVY_PATTERNS = (
    "scripts/pretrain.py",
    "lbm_eval.evaluate",
)


def log(msg: str, fh):
    timestamp = time.strftime("%F %T")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def feature_ready(feat_dir: Path, expected_count: int) -> bool:
    checks = [
        feat_dir / "foa_dev",
        feat_dir / "foa_dev_norm",
        feat_dir / "foa_dev_adpit_label",
        feat_dir / "foa_dev_dataset_stats.json",
        feat_dir / "foa_wts",
    ]
    if not all(p.exists() for p in checks):
        return False
    feature_count = sum(1 for _ in (feat_dir / "foa_dev").glob("*.npy"))
    norm_count = sum(1 for _ in (feat_dir / "foa_dev_norm").glob("*.npy"))
    label_count = sum(1 for _ in (feat_dir / "foa_dev_adpit_label").glob("*.npy"))
    return (
        feature_count == expected_count
        and norm_count == expected_count
        and label_count == expected_count
    )


def heavy_cpu_jobs_present() -> bool:
    out = subprocess.check_output(
        ["ps", "-eo", "pcpu,args", "--no-headers"],
        text=True,
    )
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pcpu, args = parts
        if any(pat in args for pat in HEAVY_PATTERNS):
            try:
                cpu = float(pcpu)
            except ValueError:
                continue
            if cpu >= 20.0:
                return True
    return False


def gpus_busy(gpu_ids: list[int]) -> bool:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    uuid_to_idx = {}
    for line in gpu_info.strip().splitlines():
        idx, uuid = [part.strip() for part in line.split(",", 1)]
        uuid_to_idx[uuid] = int(idx)

    try:
        app_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        app_info = ""

    target = set(gpu_ids)
    for line in app_info.strip().splitlines():
        uuid, _pid = [part.strip() for part in line.split(",", 1)]
        idx = uuid_to_idx.get(uuid)
        if idx in target:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--feat-dir", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    feat_dir = Path(args.feat_dir)
    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]

    with log_file.open("a", encoding="utf-8") as fh:
        log(
            f"monitor started task={args.task_id} job={args.job_id} gpus={args.gpu_ids}",
            fh,
        )

        while not feature_ready(feat_dir, args.expected_count):
            log("features not ready; waiting 60s", fh)
            time.sleep(60)

        log("features ready", fh)

        while heavy_cpu_jobs_present():
            log("heavy CPU jobs detected; waiting 120s", fh)
            time.sleep(120)

        log("CPU is clear enough", fh)

        while gpus_busy(gpu_ids):
            log(f"GPUs {args.gpu_ids} busy; waiting 120s", fh)
            time.sleep(120)

        log(f"starting training on GPUs {args.gpu_ids}", fh)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        cmd = ["python3", "train_seldnet.py", args.task_id, args.job_id]
        proc = subprocess.Popen(cmd, cwd=repo_dir, env=env)
        log(f"training process pid={proc.pid}", fh)
        return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
