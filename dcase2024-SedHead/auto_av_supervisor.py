#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


REPO_DIR = Path("/data/zhuzhiyuan/starss23/dcase2024-SedHead")
RUN_LOG_DIR = REPO_DIR / "run_logs"
STATUS_LOG = RUN_LOG_DIR / "auto_av_supervisor.log"
STATE_FILE = RUN_LOG_DIR / "auto_av_supervisor_state.json"
POLL_SECONDS = 300
HOURLY_SECONDS = 3600

TASKS = {
    247: {
        "job_id": "starss23_only_sedhead_audio_run01",
        "gpus": "0,1",
    },
    248: {
        "job_id": "starss23_plus_hf_sedhead_audio_run01",
        "gpus": "2,3",
    },
    251: {
        "job_id": "starss23_only_baseline_audio_run01",
        "gpus": "4,5",
    },
    252: {
        "job_id": "starss23_plus_hf_baseline_audio_run01",
        "gpus": "6,7",
    },
    249: {
        "job_id": "starss23_only_sedhead_av_run01",
        "gpus": "0,1",
        "prereq": 247,
        "feat_dir": "/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task249_av/video_dev",
        "expected_features": 1348,
    },
    250: {
        "job_id": "starss23_plus_hf_sedhead_av_run01",
        "gpus": "2,3",
        "prereq": 248,
        "feat_dir": "/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task250_av/video_dev",
        "expected_features": 3918,
    },
    253: {
        "job_id": "starss23_only_baseline_av_run01",
        "gpus": "4,5",
        "prereq": 251,
        "feat_dir": "/data/zhuzhiyuan/starss23/seld_feat_label/starss23_20s_16k_task249_av/video_dev",
        "expected_features": 1348,
    },
    254: {
        "job_id": "starss23_plus_hf_baseline_av_run01",
        "gpus": "6,7",
        "prereq": 252,
        "feat_dir": "/data/zhuzhiyuan/starss23/seld_feat_label/merged_starss23_spatialqa_16k_task250_av/video_dev",
        "expected_features": 3918,
    },
}


ROOT_TASKS = (247, 248, 251, 252)
AV_TASKS = (249, 250, 253, 254)


def log(msg: str) -> None:
    timestamp = time.strftime("%F %T")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with STATUS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"launched": {}, "last_hourly": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def find_task_pid(task_id: int, job_id: str) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", f"train_seldnet.py {task_id} {job_id}"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            return int(parts[0])
        except ValueError:
            continue
    return None


def process_elapsed(pid: int) -> str:
    try:
        out = subprocess.check_output(["ps", "-o", "etimes=", "-p", str(pid)], text=True).strip()
        secs = int(out)
    except Exception:
        return "unknown"
    hours, rem = divmod(secs, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def feature_count(path: Path) -> int:
    return sum(1 for _ in path.glob("*.npy"))


def feature_ready(task_id: int) -> Tuple[bool, int, int]:
    conf = TASKS[task_id]
    path = Path(conf["feat_dir"])
    have = feature_count(path) if path.exists() else 0
    need = conf["expected_features"]
    return have >= need, have, need


def gpus_busy(gpu_ids: List[int]) -> bool:
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return True
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


def launch_task(task_id: int, state: dict) -> None:
    conf = TASKS[task_id]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = conf["gpus"]
    log_path = RUN_LOG_DIR / f"{task_id}_auto.log"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{time.strftime('%F %T')}] auto-launch task {task_id}\n")
        fh.flush()
        proc = subprocess.Popen(
            ["python3", "-u", "train_seldnet.py", str(task_id), conf["job_id"]],
            cwd=REPO_DIR,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["launched"][str(task_id)] = {
        "pid": proc.pid,
        "time": time.strftime("%F %T"),
    }
    save_state(state)
    log(f"launched task {task_id} pid={proc.pid} gpus={conf['gpus']}")


def hourly_report(state: dict) -> None:
    lines = []
    for task_id in (247, 248, 251, 252, 249, 250, 253, 254):
        conf = TASKS[task_id]
        pid = find_task_pid(task_id, conf["job_id"])
        if pid is not None:
            lines.append(f"task {task_id}: running pid={pid} elapsed={process_elapsed(pid)}")
        elif "feat_dir" in conf:
            ready, have, need = feature_ready(task_id)
            launched = str(task_id) in state["launched"]
            lines.append(
                f"task {task_id}: idle launched={launched} features={have}/{need} ready={ready}"
            )
        else:
            lines.append(f"task {task_id}: not running")
    log("hourly status | " + " | ".join(lines))


def maybe_launch_av(task_id: int, state: dict) -> None:
    conf = TASKS[task_id]
    if find_task_pid(task_id, conf["job_id"]) is not None:
        return
    if str(task_id) in state["launched"]:
        return
    prereq = conf["prereq"]
    if str(prereq) not in state["launched"]:
        return
    prereq_conf = TASKS[prereq]
    if find_task_pid(prereq, prereq_conf["job_id"]) is not None:
        return
    ready, have, need = feature_ready(task_id)
    if not ready:
        log(f"task {task_id} waiting for features {have}/{need}")
        return
    gpu_ids = [int(x) for x in conf["gpus"].split(",")]
    if gpus_busy(gpu_ids):
        log(f"task {task_id} waiting for GPUs {conf['gpus']}")
        return
    launch_task(task_id, state)


def maybe_launch_root(task_id: int, state: dict) -> None:
    conf = TASKS[task_id]
    if find_task_pid(task_id, conf["job_id"]) is not None:
        return
    if str(task_id) in state["launched"]:
        return
    gpu_ids = [int(x) for x in conf["gpus"].split(",")]
    if gpus_busy(gpu_ids):
        log(f"task {task_id} waiting for GPUs {conf['gpus']}")
        return
    launch_task(task_id, state)


def main() -> int:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    log("auto AV supervisor started")
    while True:
        now = time.time()
        if now - state.get("last_hourly", 0) >= HOURLY_SECONDS:
            hourly_report(state)
            state["last_hourly"] = now
            save_state(state)
        for task_id in ROOT_TASKS:
            maybe_launch_root(task_id, state)
        for task_id in AV_TASKS:
            maybe_launch_av(task_id, state)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
