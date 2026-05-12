#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


DEFAULT_DATASET_ROOT = Path(
    "/apdcephfs_cq10/share_1603164/user/schmittzhu/code/DCASE2024_seld_baseline/prepared_datasets/real_foa_fsd50k_16k"
)


@dataclass
class CsvAnalysisResult:
    clip_duration_bin: str
    clip_duration_sec: float
    class_duration_frames: Counter
    azimuth_bin_counter: Counter
    elevation_bin_counter: Counter
    distance_bin_counter: Counter
    unknown_distance_count: int
    unknown_elevation_count: int
    infinite_elevation_count: int
    row_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the distribution of a prepared DCASE-style real FOA dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_real_foa_profile"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, cpu_count() or 1)))
    return parser.parse_args()


def duration_bin_label(duration_sec: float) -> str:
    lower = int(math.floor(duration_sec))
    upper = lower + 1
    return f"[{lower},{upper})"


def azimuth_bin_label(azimuth_deg: float, bin_size: int = 20) -> str:
    clamped = min(max(azimuth_deg, -180.0), 180.0 - 1e-9)
    lower = int(math.floor((clamped + 180.0) / bin_size)) * bin_size - 180
    upper = lower + bin_size
    return f"[{lower},{upper})"


def elevation_bin_label(elevation_deg: float, bin_size: int = 10) -> str:
    clamped = min(max(elevation_deg, -90.0), 90.0 - 1e-9)
    lower = int(math.floor((clamped + 90.0) / bin_size)) * bin_size - 90
    upper = lower + bin_size
    return f"[{lower},{upper})"


def distance_bin_label(distance_m: float, bin_size: float = 0.5) -> str:
    lower = math.floor(distance_m / bin_size) * bin_size
    upper = lower + bin_size
    return f"[{lower:.1f},{upper:.1f})"


def safe_float(text: str) -> float:
    lowered = text.strip().lower()
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    return float(lowered)


def load_class_mapping(path: Path) -> Dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in payload["class_id_to_name"].items()}


def load_segment_manifest(path: Path) -> Dict[str, dict]:
    by_stem = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            by_stem[row["stem"]] = row
    return by_stem


def iter_csv_paths(metadata_root: Path) -> Iterable[Path]:
    yield from sorted(metadata_root.rglob("*.csv"))


def analyze_csv(csv_path: Path, segment_manifest_by_stem: Dict[str, dict], class_id_to_name: Dict[int, str], label_hop_seconds: float) -> CsvAnalysisResult:
    stem = csv_path.stem
    if stem not in segment_manifest_by_stem:
        raise KeyError(f"Missing segment manifest entry for stem: {stem}")

    manifest_row = segment_manifest_by_stem[stem]
    clip_duration_sec = float(manifest_row["duration_sec"])
    clip_bin = duration_bin_label(clip_duration_sec)

    azimuth_bin_counter = Counter()
    elevation_bin_counter = Counter()
    distance_bin_counter = Counter()
    class_duration_frames = Counter()
    unknown_distance_count = 0
    unknown_elevation_count = 0
    infinite_elevation_count = 0
    row_count = 0

    by_track = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            frame_idx = int(row[0])
            class_id = int(row[1])
            track_id = int(row[2])
            azimuth_deg = safe_float(row[3])
            elevation_deg = safe_float(row[4])
            distance_cm = int(row[5])
            row_count += 1

            azimuth_bin_counter[azimuth_bin_label(azimuth_deg)] += 1

            if math.isinf(elevation_deg):
                infinite_elevation_count += 1
                elevation_bin_counter["infinite"] += 1
            elif math.isnan(elevation_deg):
                unknown_elevation_count += 1
                elevation_bin_counter["unknown"] += 1
            else:
                elevation_bin_counter[elevation_bin_label(elevation_deg)] += 1

            if distance_cm < 0:
                unknown_distance_count += 1
                distance_bin_counter["unknown"] += 1
            else:
                distance_m = distance_cm / 100.0
                distance_bin_counter[distance_bin_label(distance_m)] += 1

            by_track[(class_id, track_id)].append(frame_idx)

    for (class_id, _track_id), frames in by_track.items():
        frames = sorted(set(frames))
        run_start = frames[0]
        run_end = frames[0]
        total_frames = 0
        for frame_idx in frames[1:]:
            if frame_idx == run_end + 1:
                run_end = frame_idx
            else:
                total_frames += run_end - run_start + 1
                run_start = frame_idx
                run_end = frame_idx
        total_frames += run_end - run_start + 1
        class_name = class_id_to_name[class_id]
        class_duration_frames[class_name] += total_frames

    return CsvAnalysisResult(
        clip_duration_bin=clip_bin,
        clip_duration_sec=clip_duration_sec,
        class_duration_frames=class_duration_frames,
        azimuth_bin_counter=azimuth_bin_counter,
        elevation_bin_counter=elevation_bin_counter,
        distance_bin_counter=distance_bin_counter,
        unknown_distance_count=unknown_distance_count,
        unknown_elevation_count=unknown_elevation_count,
        infinite_elevation_count=infinite_elevation_count,
        row_count=row_count,
    )


def write_counter_csv(path: Path, first_col: str, counter: Counter, value_col: str = "count") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([first_col, value_col])
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
            writer.writerow([key, value])


def write_class_duration_csv(path: Path, class_duration_sec: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mapped_fsd_class", "total_duration_sec"])
        for class_name, duration_sec in sorted(class_duration_sec.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([class_name, f"{duration_sec:.6f}"])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    class_mapping_path = dataset_root / "class_mapping.json"
    segment_manifest_path = dataset_root / "segment_manifest.csv"
    metadata_root = dataset_root / "metadata_dev"

    class_id_to_name = load_class_mapping(class_mapping_path)
    segment_manifest_by_stem = load_segment_manifest(segment_manifest_path)
    label_hop_seconds = json.loads(class_mapping_path.read_text(encoding="utf-8"))["label_hop_seconds"]

    csv_paths = list(iter_csv_paths(metadata_root))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {metadata_root}")

    duration_bin_counter = Counter()
    class_duration_frames = Counter()
    azimuth_bin_counter = Counter()
    elevation_bin_counter = Counter()
    distance_bin_counter = Counter()
    total_clip_duration_sec = 0.0
    total_rows = 0
    total_unknown_distance_rows = 0
    total_unknown_elevation_rows = 0
    total_infinite_elevation_rows = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(analyze_csv, csv_path, segment_manifest_by_stem, class_id_to_name, label_hop_seconds)
            for csv_path in csv_paths
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyze clips", unit="clip", dynamic_ncols=True):
            result = future.result()
            duration_bin_counter[result.clip_duration_bin] += 1
            class_duration_frames.update(result.class_duration_frames)
            azimuth_bin_counter.update(result.azimuth_bin_counter)
            elevation_bin_counter.update(result.elevation_bin_counter)
            distance_bin_counter.update(result.distance_bin_counter)
            total_clip_duration_sec += result.clip_duration_sec
            total_rows += result.row_count
            total_unknown_distance_rows += result.unknown_distance_count
            total_unknown_elevation_rows += result.unknown_elevation_count
            total_infinite_elevation_rows += result.infinite_elevation_count

    class_duration_sec = {
        class_name: frames * label_hop_seconds
        for class_name, frames in class_duration_frames.items()
    }

    summary = {
        "dataset_root": str(dataset_root),
        "num_clips": len(csv_paths),
        "total_clip_duration_sec": round(total_clip_duration_sec, 6),
        "label_hop_seconds": label_hop_seconds,
        "num_mapped_classes": len(class_duration_sec),
        "total_metadata_rows": total_rows,
        "unknown_distance_rows": total_unknown_distance_rows,
        "unknown_elevation_rows": total_unknown_elevation_rows,
        "infinite_elevation_rows": total_infinite_elevation_rows,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_counter_csv(output_dir / "clip_duration_bins_1s.csv", "duration_bin_sec", duration_bin_counter)
    write_class_duration_csv(output_dir / "class_total_duration_sec.csv", class_duration_sec)
    write_counter_csv(output_dir / "azimuth_bins_20deg.csv", "azimuth_bin_deg", azimuth_bin_counter)
    write_counter_csv(output_dir / "elevation_bins_10deg.csv", "elevation_bin_deg", elevation_bin_counter)
    write_counter_csv(output_dir / "distance_bins_0p5m.csv", "distance_bin_m", distance_bin_counter)

    print("Dataset profile completed.")
    print(f"Output dir: {output_dir}")
    print(f"Clips analyzed: {len(csv_paths)}")
    print(f"Mapped classes: {len(class_duration_sec)}")
    print(f"Unknown distance rows: {total_unknown_distance_rows}")
    print(f"Infinite elevation rows: {total_infinite_elevation_rows}")


if __name__ == "__main__":
    main()
