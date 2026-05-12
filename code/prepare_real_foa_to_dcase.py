#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import subprocess
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, unit=None, dynamic_ncols=None):
        if iterable is None:
            class _Dummy:
                def update(self, n=1):
                    return None

                def close(self):
                    return None
            return _Dummy()
        return iterable


LABEL_HOP_SECONDS = 0.1
TARGET_SAMPLE_RATE = 16000
SEGMENT_MIN_SECONDS = 5.0
SEGMENT_MAX_SECONDS = 20.0
SEGMENT_TARGET_SECONDS = 12.0
MERGE_GAP_SECONDS = 1.5
DEFAULT_DATASET_KEY = "realfoa"
MISSING_DISTANCE_CM = -1


RAW_TO_FSD_MAP = {
    "309": "home_sound",
    "alarm": "alarm",
    "appliance": "appliance",
    "applicance": "appliance",
    "ball": "percussion",
    "bicycle": "vehicle",
    "bird": "bird",
    "black": "home_sound",
    "board": "wood",
    "broadcasting": "speech",
    "cabinet": "drawer_cabinet",
    "cabinet door": "drawer_cabinet",
    "car": "car",
    "cart": "vehicle",
    "cat": "cat",
    "chair": "home_sound",
    "close_door": "door",
    "close_window": "door",
    "coffee": "cooking",
    "colorblue": "home_sound",
    "cooking": "cooking",
    "crash": "crack",
    "crush": "crushing",
    "cutting": "tool",
    "decoration": "home_sound",
    "door": "door",
    "door_opening": "door",
    "dustbin": "home_sound",
    "facemachine": "machine",
    "farblue": "home_sound",
    "female_speech": "female_speech",
    "fetch": "home_sound",
    "fetching": "home_sound",
    "footsteps": "footsteps",
    "glass": "glass",
    "hit": "percussion",
    "human_vocalization": "human_vocalization",
    "items": "home_sound",
    "kitchenware": "kitchenware",
    "larder": "drawer_cabinet",
    "laughter": "laughter",
    "machine": "machine",
    "male_speech": "male_speech",
    "man_speech": "male_speech",
    "medal": "metal_clink",
    "metal_clink": "metal_clink",
    "music": "musical_instrument",
    "operation": "machine",
    "packing": "zipper",
    "paging": "paper",
    "paper": "paper",
    "phone": "telephone_alarm",
    "placing": "home_sound",
    "plastic": "home_sound",
    "plasticbag": "crushing",
    "printer": "printer",
    "printing": "printer",
    "putting": "home_sound",
    "rangehood": "appliance",
    "ringtone": "telephone_alarm",
    "screen": "appliance",
    "shredder": "machine",
    "speech": "speech",
    "suitcase": "zipper",
    "sweeping": "home_sound",
    "talking": "speech",
    "tape": "tape",
    "tearing": "tearing",
    "tool": "tool",
    "train": "train",
    "typing": "typing",
    "umbrella": "home_sound",
    "unbrella": "home_sound",
    "vehicle": "vehicle",
    "walking_female_speech": "female_speech",
    "water_tap_and_faucet": "water",
    "waterdispenser": "water",
    "whistle": "human_vocalization",
    "wind": "wind",
}


RAW_TO_FSD_NOTE = {
    "ball": "impact-like, mapped to percussion",
    "bicycle": "not in FSD vocabulary, folded into vehicle",
    "black": "non-acoustic label, folded into home_sound",
    "board": "surface/object cue, folded into wood",
    "broadcasting": "speech content, folded into speech",
    "cabinet": "object-like event, folded into drawer_cabinet",
    "cabinet door": "door/cabinet interaction, folded into drawer_cabinet",
    "chair": "furniture manipulation, folded into home_sound",
    "close_door": "alias of door event",
    "close_window": "window interaction, folded into door",
    "coffee": "coffee-making scene, folded into cooking",
    "colorblue": "non-acoustic label, folded into home_sound",
    "crash": "closest available FSD label is crack",
    "crush": "alias of crushing",
    "cutting": "tool-use event, folded into tool",
    "decoration": "object-like event, folded into home_sound",
    "door_opening": "alias of door event",
    "dustbin": "object-like event, folded into home_sound",
    "facemachine": "machine-like event, folded into machine",
    "farblue": "non-acoustic label, folded into home_sound",
    "fetch": "generic handling event, folded into home_sound",
    "fetching": "generic handling event, folded into home_sound",
    "hit": "impact-like, folded into percussion",
    "items": "generic handling event, folded into home_sound",
    "larder": "cabinet-like source, folded into drawer_cabinet",
    "man_speech": "alias of male_speech",
    "music": "music not present in FSD subset, folded into musical_instrument",
    "operation": "machine operation, folded into machine",
    "packing": "bag/packing interaction, folded into zipper",
    "phone": "phone-related event, folded into telephone_alarm",
    "placing": "generic handling event, folded into home_sound",
    "plastic": "generic material sound, folded into home_sound",
    "plasticbag": "bag crumpling, folded into crushing",
    "putting": "generic handling event, folded into home_sound",
    "rangehood": "appliance source, folded into appliance",
    "ringtone": "alias of telephone_alarm",
    "screen": "device-like source, folded into appliance",
    "shredder": "machine source, folded into machine",
    "suitcase": "suitcase interaction, folded into zipper",
    "sweeping": "domestic activity, folded into home_sound",
    "talking": "alias of speech",
    "umbrella": "object handling, folded into home_sound",
    "unbrella": "typo alias of umbrella, folded into home_sound",
    "walking_female_speech": "compound label, folded into female_speech",
    "water_tap_and_faucet": "not in FSD subset, folded into water",
    "waterdispenser": "water/appliance source, folded into water",
    "whistle": "human-produced whistle, folded into human_vocalization",
}


@dataclass
class Event:
    recording_id: str
    annotation_index: int
    raw_event_name: str
    mapped_event_name: str
    start_time: float
    end_time: float
    moving: int
    location: Optional[dict]
    location_end: Optional[dict]


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real FOA clips and DCASE-style metadata from annotated recordings.")
    parser.add_argument("--audio-root", type=Path, default=Path("/apdcephfs_cq10/share_1603164/user/schmittzhu/data/RealFOA"))
    parser.add_argument("--anno-root", type=Path, default=Path("/apdcephfs_cq10/share_1603164/user/schmittzhu/data/metadata/real_anno"))
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/apdcephfs_cq10/share_1603164/user/schmittzhu/code/DCASE2024_seld_baseline/prepared_datasets/real_foa_fsd50k_16k"),
    )
    parser.add_argument(
        "--fsd-vocab-csv",
        type=Path,
        default=Path("/apdcephfs_cq12/share_302080740/user/schmittzhu/data/fsd50k/FSD50K.ground_truth/final_vocabulary.csv"),
    )
    parser.add_argument("--dataset-key", type=str, default=DEFAULT_DATASET_KEY)
    parser.add_argument("--target-sr", type=int, default=TARGET_SAMPLE_RATE)
    parser.add_argument("--label-hop-seconds", type=float, default=LABEL_HOP_SECONDS)
    parser.add_argument("--min-seconds", type=float, default=SEGMENT_MIN_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=SEGMENT_MAX_SECONDS)
    parser.add_argument("--target-seconds", type=float, default=SEGMENT_TARGET_SECONDS)
    parser.add_argument("--merge-gap-seconds", type=float, default=MERGE_GAP_SECONDS)
    parser.add_argument("--split-strategy", choices=["hash", "all_train"], default="hash")
    parser.add_argument("--split-ratios", type=str, default="0.85,0.05,0.10")
    parser.add_argument("--split-seed", type=str, default="realfoa_v1")
    parser.add_argument("--position-rel-convention", choices=["x_front_y_left"], default="x_front_y_left")
    parser.set_defaults(skip_missing_location=True)
    parser.add_argument("--skip-missing-location", dest="skip_missing_location", action="store_true")
    parser.add_argument("--keep-missing-location", dest="skip_missing_location", action="store_false")
    parser.add_argument("--workers", type=int, default=max(1, min(8, cpu_count() or 1)))
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_signed_azimuth(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def updown_to_elevation(up_down_value) -> float:
    if up_down_value is None:
        return 0.0
    value = str(up_down_value).strip().lower()
    if value == "up":
        return float("inf")
    if value == "down":
        return float("-inf")
    return 0.0


def interpolate_angle_deg(start_deg: float, end_deg: float, alpha: float) -> float:
    delta = ((end_deg - start_deg + 180.0) % 360.0) - 180.0
    return normalize_signed_azimuth(start_deg + alpha * delta)


def position_rel_to_listener_frame(position_rel: Sequence[float], convention: str) -> Tuple[float, float, float]:
    if convention != "x_front_y_left":
        raise ValueError(f"Unsupported convention: {convention}")
    x_front = float(position_rel[0])
    y_left = float(position_rel[1])
    z_up = float(position_rel[2])
    return x_front, y_left, z_up


def listener_frame_to_dcase_pose(x_front: float, y_left: float, z_up: float) -> Tuple[float, float, int]:
    distance_m = math.sqrt(x_front * x_front + y_left * y_left + z_up * z_up)
    azimuth = normalize_signed_azimuth(math.degrees(math.atan2(y_left, x_front)))
    elevation = math.degrees(math.atan2(z_up, math.hypot(x_front, y_left))) if distance_m > 0 else 0.0
    distance_cm = int(round(distance_m * 100.0))
    return azimuth, elevation, distance_cm


def infer_pose_from_location(location: Optional[dict], convention: str) -> Optional[Tuple[float, float, int]]:
    if not isinstance(location, dict):
        return None
    position_rel = location.get("position_rel")
    if isinstance(position_rel, list) and len(position_rel) >= 3:
        x_front, y_left, z_up = position_rel_to_listener_frame(position_rel, convention)
        return listener_frame_to_dcase_pose(x_front, y_left, z_up)
    heading = safe_float(location.get("heading"))
    if heading is None:
        heading = safe_float(location.get("headmining"))
    if heading is not None:
        azimuth = normalize_signed_azimuth(heading)
        elevation = updown_to_elevation(location.get("up_down"))
        return azimuth, elevation, MISSING_DISTANCE_CM
    left_right = str(location.get("left_right")).strip().lower() if location.get("left_right") is not None else ""
    front_back = str(location.get("front_back")).strip().lower() if location.get("front_back") is not None else ""
    up_down = location.get("up_down")
    if left_right or front_back or up_down not in (None, "", "0", 0):
        if front_back == "front" and left_right == "left":
            azimuth = 45.0
        elif front_back == "front" and left_right == "right":
            azimuth = -45.0
        elif front_back == "back" and left_right == "left":
            azimuth = 135.0
        elif front_back == "back" and left_right == "right":
            azimuth = -135.0
        elif left_right == "left":
            azimuth = 90.0
        elif left_right == "right":
            azimuth = -90.0
        elif front_back == "back":
            azimuth = 180.0
        else:
            azimuth = 0.0
        elevation = updown_to_elevation(up_down)
        return normalize_signed_azimuth(azimuth), elevation, MISSING_DISTANCE_CM
    return None


def infer_pose_for_event(event: Event, t_global: float, convention: str) -> Optional[Tuple[float, float, int]]:
    duration = max(event.end_time - event.start_time, 1e-6)
    alpha = min(max((t_global - event.start_time) / duration, 0.0), 1.0)
    loc0 = event.location
    loc1 = event.location_end if isinstance(event.location_end, dict) else loc0
    if event.moving == 1 and isinstance(loc0, dict) and isinstance(loc1, dict):
        pos0 = loc0.get("position_rel")
        pos1 = loc1.get("position_rel")
        if isinstance(pos0, list) and len(pos0) >= 3 and isinstance(pos1, list) and len(pos1) >= 3:
            xyz0 = np.array(position_rel_to_listener_frame(pos0, convention), dtype=np.float64)
            xyz1 = np.array(position_rel_to_listener_frame(pos1, convention), dtype=np.float64)
            xyz = xyz0 + alpha * (xyz1 - xyz0)
            return listener_frame_to_dcase_pose(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        pose0 = infer_pose_from_location(loc0, convention)
        pose1 = infer_pose_from_location(loc1, convention)
        if pose0 is not None and pose1 is not None:
            azimuth = interpolate_angle_deg(pose0[0], pose1[0], alpha)
            elev0, elev1 = pose0[1], pose1[1]
            if np.isfinite(elev0) and np.isfinite(elev1):
                elevation = elev0 + (elev1 - elev0) * alpha
            else:
                elevation = elev0 if alpha < 0.5 else elev1
            dist0, dist1 = pose0[2], pose1[2]
            if dist0 >= 0 and dist1 >= 0:
                distance_cm = int(round(dist0 + (dist1 - dist0) * alpha))
            elif dist0 >= 0:
                distance_cm = dist0
            elif dist1 >= 0:
                distance_cm = dist1
            else:
                distance_cm = MISSING_DISTANCE_CM
            return normalize_signed_azimuth(azimuth), elevation, distance_cm
    return infer_pose_from_location(loc0, convention)


def iter_recording_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("20*/VID_*")):
        if path.is_dir():
            yield path


def find_single_json(recording_dir: Path) -> Optional[Path]:
    json_files = sorted(recording_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]
    return None


def find_single_video(recording_dir: Path) -> Optional[Path]:
    video_files = sorted(recording_dir.glob("*.mp4"))
    if len(video_files) == 1:
        return video_files[0]
    return None


def load_fsd_vocab(path: Path) -> Tuple[Dict[str, dict], List[dict]]:
    rows = []
    by_label = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row["clean_label"]
            rows.append(row)
            by_label[label] = row
    return by_label, rows


def parse_split_ratios(ratio_text: str) -> Tuple[float, float, float]:
    values = [float(item.strip()) for item in ratio_text.split(",")]
    if len(values) != 3:
        raise ValueError("split-ratios must have three comma-separated values.")
    total = sum(values)
    if total <= 0:
        raise ValueError("split-ratios must sum to a positive value.")
    return values[0] / total, values[1] / total, values[2] / total


def assign_split(recording_id: str, strategy: str, seed: str, ratios: Tuple[float, float, float]) -> str:
    if strategy == "all_train":
        return "train"
    digest = hashlib.md5(f"{seed}:{recording_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    train_ratio, valid_ratio, _ = ratios
    if value < train_ratio:
        return "train"
    if value < train_ratio + valid_ratio:
        return "valid"
    return "test"


def clip_annotation_times(start: Optional[float], end: Optional[float], duration: float) -> Optional[Tuple[float, float]]:
    if start is None or end is None:
        return None
    start = min(max(start, 0.0), duration)
    end = min(max(end, 0.0), duration)
    if end <= start:
        return None
    return start, end


def merge_intervals(intervals: List[Tuple[float, float]], merge_gap_seconds: float) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        prev = merged[-1]
        if start - prev[1] <= merge_gap_seconds:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def choose_cut(start: float, end: float, event_intervals: List[Tuple[float, float]], min_seconds: float, max_seconds: float, target_seconds: float) -> float:
    candidate_cuts: List[Tuple[float, float, float]] = []
    for (a_start, a_end), (b_start, b_end) in zip(event_intervals[:-1], event_intervals[1:]):
        gap = b_start - a_end
        if gap <= 0:
            continue
        cut = a_end + gap / 2.0
        left = cut - start
        right = end - cut
        if left >= min_seconds and left <= max_seconds and right >= min_seconds:
            candidate_cuts.append((-gap, abs(left - target_seconds), cut))
    if candidate_cuts:
        candidate_cuts.sort()
        return candidate_cuts[0][2]

    boundary_cuts: List[Tuple[float, float]] = []
    for boundary in sorted({x for interval in event_intervals for x in interval}):
        left = boundary - start
        right = end - boundary
        if left >= min_seconds and left <= max_seconds and right >= min_seconds:
            boundary_cuts.append((abs(left - target_seconds), boundary))
    if boundary_cuts:
        boundary_cuts.sort()
        return boundary_cuts[0][1]

    cut = min(start + target_seconds, start + max_seconds)
    cut = min(cut, end - min_seconds)
    return max(cut, start + min_seconds)


def segment_island(
    island_start: float,
    island_end: float,
    event_intervals: List[Tuple[float, float]],
    audio_duration: float,
    min_seconds: float,
    max_seconds: float,
    target_seconds: float,
) -> List[Segment]:
    island_duration = island_end - island_start
    if island_duration <= max_seconds:
        if island_duration >= min_seconds:
            return [Segment(island_start, island_end)]
        needed = min_seconds - island_duration
        pad_left = min(needed / 2.0, island_start)
        pad_right = min(needed - pad_left, audio_duration - island_end)
        seg_start = island_start - pad_left
        seg_end = island_end + pad_right
        if seg_end - seg_start < min_seconds:
            deficit = min_seconds - (seg_end - seg_start)
            extra_left = min(deficit, seg_start)
            seg_start -= extra_left
            seg_end = min(audio_duration, seg_end + (deficit - extra_left))
        return [Segment(max(0.0, seg_start), min(audio_duration, seg_end))]

    segments: List[Segment] = []
    current_start = island_start
    current_event_idx = 0
    while island_end - current_start > max_seconds:
        relevant = []
        while current_event_idx < len(event_intervals) and event_intervals[current_event_idx][1] <= current_start:
            current_event_idx += 1
        scan_idx = current_event_idx
        while scan_idx < len(event_intervals) and event_intervals[scan_idx][0] < island_end:
            relevant.append(event_intervals[scan_idx])
            scan_idx += 1
        cut = choose_cut(current_start, island_end, relevant, min_seconds, max_seconds, target_seconds)
        segments.append(Segment(current_start, cut))
        current_start = cut
    tail = Segment(current_start, island_end)
    if tail.duration < min_seconds and segments:
        previous = segments[-1]
        combined = tail.end - previous.start
        if combined <= max_seconds:
            segments[-1] = Segment(previous.start, tail.end)
        else:
            shift = min_seconds - tail.duration
            new_prev_end = previous.end - shift
            if new_prev_end - previous.start >= min_seconds:
                segments[-1] = Segment(previous.start, new_prev_end)
                tail = Segment(new_prev_end, tail.end)
                segments.append(tail)
            else:
                segments.append(tail)
    else:
        segments.append(tail)
    return segments


def build_segments(events: List[Event], audio_duration: float, min_seconds: float, max_seconds: float, target_seconds: float, merge_gap_seconds: float) -> List[Segment]:
    intervals = sorted((event.start_time, event.end_time) for event in events if event.end_time > event.start_time)
    islands = merge_intervals(intervals, merge_gap_seconds)
    segments: List[Segment] = []
    interval_index = 0
    for island_start, island_end in islands:
        island_events = []
        while interval_index < len(intervals) and intervals[interval_index][1] <= island_start:
            interval_index += 1
        scan_idx = interval_index
        while scan_idx < len(intervals) and intervals[scan_idx][0] < island_end:
            island_events.append(intervals[scan_idx])
            scan_idx += 1
        segments.extend(segment_island(island_start, island_end, island_events, audio_duration, min_seconds, max_seconds, target_seconds))
    return segments


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def resample_audio(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return audio
    gcd = math.gcd(src_sr, target_sr)
    up = target_sr // gcd
    down = src_sr // gcd
    return resample_poly(audio, up, down, axis=0).astype(np.float32)


def overlap_exists(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return min(a_end, b_end) > max(a_start, b_start)


def format_numeric(value) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.3f}"
    return str(value)


def create_manifest_entry(split: str, stem: str, source_recording_id: str, segment_index: int, foa_path: Path, csv_path: Path, duration: float, sample_rate: int, length_samples: int, dataset_key: str) -> dict:
    return {
        "split": split,
        "stem": stem,
        "source_stem": source_recording_id.replace("/", "__"),
        "segment_index": segment_index,
        "dataset": dataset_key,
        "foa_path": foa_path.as_posix(),
        "csv_path": csv_path.as_posix(),
        "length": round(duration, 6),
        "length_samples": int(length_samples),
        "sample_rate": int(sample_rate),
    }


def export_video_segment(ffmpeg_bin: str, input_path: Path, output_path: Path, start: float, duration: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0?",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def clip_json_annotations(metadata: dict, segment_start: float, segment_end: float) -> dict:
    clipped_payload = {key: value for key, value in metadata.items() if key != "annotations"}
    clipped_annotations = []
    for annotation in metadata.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        start = safe_float(annotation.get("start_time"))
        end = safe_float(annotation.get("end_time"))
        if start is None or end is None:
            continue
        clipped_start = max(start, segment_start)
        clipped_end = min(end, segment_end)
        if clipped_end <= clipped_start:
            continue
        clipped = dict(annotation)
        clipped["start_time"] = round(clipped_start - segment_start, 6)
        clipped["end_time"] = round(clipped_end - segment_start, 6)
        clipped_annotations.append(clipped)
    clipped_payload["annotations"] = clipped_annotations
    clipped_payload["segment_start_time"] = round(segment_start, 6)
    clipped_payload["segment_end_time"] = round(segment_end, 6)
    clipped_payload["segment_duration"] = round(segment_end - segment_start, 6)
    return clipped_payload


def merge_counter(target: Counter, payload: Dict[str, int]) -> None:
    for key, value in payload.items():
        target[key] += value


def discover_recording(audio_dir: Path, args: argparse.Namespace, fsd_by_label: Dict[str, dict]) -> dict:
    rel = audio_dir.relative_to(args.audio_root).as_posix()
    anno_dir = args.anno_root / rel
    wav_path = audio_dir / "aligned_foa_bformat.wav"
    json_path = find_single_json(anno_dir) if anno_dir.exists() else None
    video_path = None
    if args.video_root is not None:
        video_dir = args.video_root / rel
        if video_dir.exists():
            video_path = find_single_video(video_dir)
    prep_stats = Counter()
    raw_event_counter = Counter()

    if not wav_path.exists() or json_path is None:
        prep_stats["skipped_missing_pair"] += 1
        return {"item": None, "prep_stats": dict(prep_stats), "raw_event_counter": dict(raw_event_counter)}

    with sf.SoundFile(str(wav_path), "r") as handle:
        audio_duration = len(handle) / handle.samplerate
        src_sr = int(handle.samplerate)
        channels = int(handle.channels)

    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    events: List[Event] = []
    for annotation_index, ann in enumerate(metadata.get("annotations", [])):
        raw_name = ann.get("event_name")
        if not raw_name:
            prep_stats["skipped_missing_event_name"] += 1
            continue
        raw_event_counter[raw_name] += 1
        mapped_label = RAW_TO_FSD_MAP.get(raw_name, raw_name if raw_name in fsd_by_label else None)
        if mapped_label is None:
            prep_stats["skipped_unmapped_label"] += 1
            continue
        if mapped_label not in fsd_by_label:
            raise ValueError(f"Mapped label {mapped_label} for raw event {raw_name} not found in FSD vocabulary.")
        clipped = clip_annotation_times(safe_float(ann.get("start_time")), safe_float(ann.get("end_time")), audio_duration)
        if clipped is None:
            prep_stats["skipped_invalid_time"] += 1
            continue
        location = ann.get("location")
        location_end = ann.get("location_end")
        if args.skip_missing_location and location is None:
            prep_stats["skipped_missing_location"] += 1
            continue
        events.append(
            Event(
                recording_id=rel,
                annotation_index=annotation_index,
                raw_event_name=raw_name,
                mapped_event_name=mapped_label,
                start_time=clipped[0],
                end_time=clipped[1],
                moving=int(ann.get("moving", 0) or 0),
                location=location,
                location_end=location_end,
            )
        )

    if not events:
        prep_stats["recordings_with_no_usable_events"] += 1
        return {"item": None, "prep_stats": dict(prep_stats), "raw_event_counter": dict(raw_event_counter)}

    item = {
        "recording_id": rel,
        "wav_path": wav_path,
        "json_path": json_path,
        "video_path": video_path,
        "audio_duration": audio_duration,
        "src_sr": src_sr,
        "channels": channels,
        "events": events,
        "metadata": metadata,
    }
    return {"item": item, "prep_stats": dict(prep_stats), "raw_event_counter": dict(raw_event_counter)}


def process_recording_item(
    item: dict,
    args: argparse.Namespace,
    split_ratios: Tuple[float, float, float],
    class_name_to_id: Dict[str, int],
) -> dict:
    recording_id = item["recording_id"]
    split = assign_split(recording_id, args.split_strategy, args.split_seed, split_ratios)
    segments = build_segments(
        item["events"],
        item["audio_duration"],
        args.min_seconds,
        args.max_seconds,
        args.target_seconds,
        args.merge_gap_seconds,
    )
    result = {
        "recording_id": recording_id,
        "split": split,
        "segments_total": len(segments),
        "prep_stats": {},
        "split_stems": [],
        "split_manifest_entries": [],
        "per_segment_rows": [],
    }

    if args.dry_run:
        return result

    foa_dir = args.output_root / "foa_dev" / args.dataset_key
    meta_dir = args.output_root / "metadata_dev" / args.dataset_key
    json_dir = args.output_root / "json_dev" / args.dataset_key
    video_dir = args.output_root / "video_dev" / args.dataset_key

    local_stats = Counter()
    split_stems = []
    manifest_entries = []
    per_segment_rows = []

    with sf.SoundFile(str(item["wav_path"]), "r") as audio_handle:
        for segment_index, segment in enumerate(segments):
            segment_events = [
                event for event in item["events"]
                if overlap_exists(event.start_time, event.end_time, segment.start, segment.end)
            ]
            if not segment_events:
                continue
            start_frame = int(round(segment.start * item["src_sr"]))
            end_frame = int(round(segment.end * item["src_sr"]))
            audio_handle.seek(start_frame)
            audio = audio_handle.read(end_frame - start_frame, dtype="float32", always_2d=True)
            audio_16k = resample_audio(audio, item["src_sr"], args.target_sr)
            stem = f"{args.dataset_key}__{recording_id.replace('/', '__')}__seg{segment_index:04d}"
            out_wav = foa_dir / f"{stem}.wav"
            out_csv = meta_dir / f"{stem}.csv"
            out_json = json_dir / f"{stem}.json"
            sf.write(str(out_wav), audio_16k, args.target_sr)

            segment_duration = audio_16k.shape[0] / args.target_sr
            num_frames = max(1, int(math.ceil(segment_duration / args.label_hop_seconds)))
            rows = []
            for local_track_id, event in enumerate(segment_events, start=1):
                local_start = max(0.0, event.start_time - segment.start)
                local_end = min(segment_duration, event.end_time - segment.start)
                if local_end <= local_start:
                    continue
                frame_start = max(0, int(math.floor(local_start / args.label_hop_seconds)))
                frame_end = min(num_frames - 1, int(math.ceil(local_end / args.label_hop_seconds)) - 1)
                class_id = class_name_to_id[event.mapped_event_name]
                for frame_idx in range(frame_start, frame_end + 1):
                    frame_begin = frame_idx * args.label_hop_seconds
                    frame_end_time = min(segment_duration, (frame_idx + 1) * args.label_hop_seconds)
                    if not overlap_exists(local_start, local_end, frame_begin, frame_end_time):
                        continue
                    t_local = min(max((frame_begin + frame_end_time) / 2.0, local_start), local_end)
                    pose = infer_pose_for_event(event, segment.start + t_local, args.position_rel_convention)
                    if pose is None:
                        local_stats["rows_skipped_missing_pose"] += 1
                        continue
                    azimuth, elevation, distance_cm = pose
                    rows.append(
                        [
                            frame_idx,
                            class_id,
                            local_track_id,
                            format_numeric(float(azimuth)),
                            format_numeric(float(elevation)),
                            int(distance_cm),
                        ]
                    )

            with out_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)

            clipped_metadata = clip_json_annotations(item["metadata"], segment.start, segment.end)
            out_json.write_text(json.dumps(clipped_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

            if item.get("video_path") is not None:
                out_video = video_dir / f"{stem}.mp4"
                export_video_segment(args.ffmpeg_bin, item["video_path"], out_video, segment.start, segment.duration)

            split_stems.append(stem)
            manifest_entries.append(
                create_manifest_entry(
                    split=split,
                    stem=stem,
                    source_recording_id=recording_id,
                    segment_index=segment_index,
                    foa_path=out_wav,
                    csv_path=out_csv,
                    duration=segment_duration,
                    sample_rate=args.target_sr,
                    length_samples=audio_16k.shape[0],
                    dataset_key=args.dataset_key,
                )
            )
            per_segment_rows.append(
                {
                    "stem": stem,
                    "split": split,
                    "source_recording_id": recording_id,
                    "segment_index": segment_index,
                    "start_time_sec": round(segment.start, 6),
                    "end_time_sec": round(segment.end, 6),
                    "duration_sec": round(segment.duration, 6),
                    "num_events": len(segment_events),
                    "num_rows": len(rows),
                }
            )

    result["prep_stats"] = dict(local_stats)
    result["split_stems"] = split_stems
    result["split_manifest_entries"] = manifest_entries
    result["per_segment_rows"] = per_segment_rows
    return result


def main() -> None:
    args = parse_args()
    split_ratios = parse_split_ratios(args.split_ratios)
    fsd_by_label, fsd_rows = load_fsd_vocab(args.fsd_vocab_csv)

    all_recordings = []
    raw_event_counter = Counter()
    prep_stats = Counter()
    mapping_rows = []
    recording_dirs = list(iter_recording_dirs(args.audio_root))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(discover_recording, audio_dir, args, fsd_by_label) for audio_dir in recording_dirs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Discover recordings", unit="rec", dynamic_ncols=True):
            payload = future.result()
            merge_counter(prep_stats, payload["prep_stats"])
            merge_counter(raw_event_counter, payload["raw_event_counter"])
            if payload["item"] is not None:
                all_recordings.append(payload["item"])
    all_recordings.sort(key=lambda item: item["recording_id"])

    used_mapped_labels = sorted({event.mapped_event_name for item in all_recordings for event in item["events"]}, key=lambda label: int(fsd_by_label[label]["label_id"]))
    class_name_to_id = {label: index for index, label in enumerate(used_mapped_labels)}
    class_id_to_name = {str(index): label for label, index in class_name_to_id.items()}

    class_mapping = {
        "class_count": len(used_mapped_labels),
        "class_id_to_name": class_id_to_name,
        "class_name_to_id": class_name_to_id,
        "class_names": used_mapped_labels,
        "missing_distance_cm": MISSING_DISTANCE_CM,
        "target_sample_rate": args.target_sr,
        "label_hop_seconds": args.label_hop_seconds,
        "fsd_label_id_by_name": {label: int(fsd_by_label[label]["label_id"]) for label in used_mapped_labels},
        "raw_to_canonical": {raw: RAW_TO_FSD_MAP.get(raw, raw) for raw in sorted(raw_event_counter)},
    }

    for raw_name, count in sorted(raw_event_counter.items()):
        mapped = RAW_TO_FSD_MAP.get(raw_name, raw_name if raw_name in fsd_by_label else "")
        vocab_row = fsd_by_label.get(mapped) if mapped else None
        mapping_rows.append(
            {
                "raw_event_name": raw_name,
                "raw_count": count,
                "mapped_label": mapped,
                "mapping_type": "exact" if raw_name == mapped else ("manual" if mapped else "unmapped"),
                "relation_note": RAW_TO_FSD_NOTE.get(raw_name, ""),
                "fsd_label_id": vocab_row["label_id"] if vocab_row else "",
                "fsd_domain_major": vocab_row["domain_major"] if vocab_row else "",
            }
        )

    dataset_root = args.output_root
    foa_dir = dataset_root / "foa_dev" / args.dataset_key
    meta_dir = dataset_root / "metadata_dev" / args.dataset_key
    split_dir = dataset_root / "split_lists"
    ensure_output_root(dataset_root, args.overwrite)
    if not args.dry_run:
        foa_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        if args.video_root is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)

    split_lists = {"train": [], "valid": [], "test": []}
    split_manifest = {"datasets": {args.dataset_key: {"counts": {"train": 0, "valid": 0, "test": 0}, "splits": {"train": [], "valid": [], "test": []}}}}
    manifests = {"train": [], "valid": [], "test": []}
    per_segment_rows = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_recording_item, item, args, split_ratios, class_name_to_id) for item in all_recordings]
        desc = "Plan segments" if args.dry_run else "Export recordings"
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="rec", dynamic_ncols=True):
            result = future.result()
            split = result["split"]
            prep_stats["source_recordings"] += 1
            prep_stats[f"source_recordings_{split}"] += 1
            prep_stats["segments_total"] += result["segments_total"]
            merge_counter(prep_stats, result["prep_stats"])
            if args.dry_run:
                continue
            split_lists[split].extend(result["split_stems"])
            split_manifest["datasets"][args.dataset_key]["counts"][split] += len(result["split_stems"])
            split_manifest["datasets"][args.dataset_key]["splits"][split].extend(result["split_stems"])
            manifests[split].extend(result["split_manifest_entries"])
            per_segment_rows.extend(result["per_segment_rows"])

    if not args.dry_run:
        for split_name in split_lists:
            split_lists[split_name].sort()
            split_manifest["datasets"][args.dataset_key]["splits"][split_name].sort()
            manifests[split_name].sort(key=lambda entry: entry["stem"])
        per_segment_rows.sort(key=lambda row: row["stem"])
        with (dataset_root / "class_mapping.json").open("w", encoding="utf-8") as handle:
            json.dump(class_mapping, handle, ensure_ascii=False, indent=2)
        with (dataset_root / "split_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(split_manifest, handle, ensure_ascii=False, indent=2)
        with (dataset_root / "raw_to_fsd_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "raw_event_name",
                    "raw_count",
                    "mapped_label",
                    "mapping_type",
                    "relation_note",
                    "fsd_label_id",
                    "fsd_domain_major",
                ],
            )
            writer.writeheader()
            writer.writerows(mapping_rows)
        with (dataset_root / "segment_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["stem", "split", "source_recording_id", "segment_index", "start_time_sec", "end_time_sec", "duration_sec", "num_events", "num_rows"],
            )
            writer.writeheader()
            writer.writerows(per_segment_rows)
        for split_name, stems in split_lists.items():
            with (split_dir / f"{split_name}.txt").open("w", encoding="utf-8") as handle:
                for stem in stems:
                    handle.write(f"{stem}\n")
            with (dataset_root / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
                for entry in manifests[split_name]:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "dataset_key": args.dataset_key,
        "audio_root": str(args.audio_root),
        "anno_root": str(args.anno_root),
        "output_root": str(args.output_root),
        "target_sample_rate": args.target_sr,
        "label_hop_seconds": args.label_hop_seconds,
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "target_seconds": args.target_seconds,
        "merge_gap_seconds": args.merge_gap_seconds,
        "split_strategy": args.split_strategy,
        "split_ratios": split_ratios,
        "position_rel_convention": args.position_rel_convention,
        "source_recordings": prep_stats["source_recordings"],
        "segments_total": prep_stats["segments_total"],
        "used_class_count": len(used_mapped_labels),
        "used_classes": used_mapped_labels,
        "raw_class_count": len(raw_event_counter),
        "stats": dict(prep_stats),
        "fsd_vocab_size": len(fsd_rows),
    }
    if not args.dry_run:
        with (dataset_root / "prep_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("Preparation plan completed.")
    print(f"Source recordings with usable events: {prep_stats['source_recordings']}")
    print(f"Segments planned: {prep_stats['segments_total']}")
    print(f"Used mapped classes: {len(used_mapped_labels)}")
    print(f"Skipped missing-location events: {prep_stats['skipped_missing_location']}")
    print(f"Skipped unmapped-label events: {prep_stats['skipped_unmapped_label']}")
    if not args.dry_run:
        print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
