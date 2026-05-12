#!/usr/bin/env python3
import argparse
import csv
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TIME_TOLERANCE = 1e-3
ANGLE_BIN_SIZE = 30
ELEVATION_BIN_SIZE = 15
DISTANCE_BIN_EDGES = [0, 0.5, 1, 2, 3, 5, 10]


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate in {path}")
        return frames / sample_rate


def iter_recording_dirs(root: Path) -> Iterable[Path]:
    for recording_dir in sorted(root.glob("20*/VID_*")):
        if recording_dir.is_dir():
            yield recording_dir


def find_single_json(recording_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    json_files = sorted(recording_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0], json_files
    return None, json_files


def interval_union_length(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    cleaned = sorted(intervals)
    total = 0.0
    current_start, current_end = cleaned[0]
    for start, end in cleaned[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return total


def heading_bin_label(angle: float, bin_size: int = ANGLE_BIN_SIZE) -> str:
    normalized = angle % 360.0
    start = int(normalized // bin_size) * bin_size
    end = start + bin_size
    return f"[{start},{end})"


def centered_elevation_bin_label(angle: float, bin_size: int = ELEVATION_BIN_SIZE) -> str:
    start = math.floor(angle / bin_size) * bin_size
    end = start + bin_size
    return f"[{int(start)},{int(end)})"


def distance_bin_label(distance: float) -> str:
    for idx in range(len(DISTANCE_BIN_EDGES) - 1):
        left = DISTANCE_BIN_EDGES[idx]
        right = DISTANCE_BIN_EDGES[idx + 1]
        if left <= distance < right:
            return f"[{left},{right})"
    last = DISTANCE_BIN_EDGES[-1]
    return f"[{last},inf)"


def infer_spatial_features(location: Optional[Dict]) -> Dict[str, Optional[float]]:
    features = {
        "heading_deg": None,
        "elevation_deg": None,
        "distance": None,
        "location_type": None,
    }
    if not isinstance(location, dict):
        return features

    heading = safe_float(location.get("heading"))
    if heading is not None:
        features["heading_deg"] = heading % 360.0
        features["location_type"] = "heading"

    if any(key in location for key in ("front_back", "left_right", "up_down")) and features["location_type"] is None:
        features["location_type"] = "categorical_direction"

    position_rel = location.get("position_rel")
    if isinstance(position_rel, list) and len(position_rel) >= 3:
        x = safe_float(position_rel[0])
        y = safe_float(position_rel[1])
        z = safe_float(position_rel[2])
        if x is not None and y is not None and z is not None:
            xy_norm = math.hypot(x, y)
            distance = math.sqrt(x * x + y * y + z * z)
            if features["heading_deg"] is None:
                features["heading_deg"] = math.degrees(math.atan2(y, x)) % 360.0
            features["elevation_deg"] = math.degrees(math.atan2(z, xy_norm)) if distance > 0 else 0.0
            features["distance"] = distance
            if features["location_type"] is None:
                features["location_type"] = "position_rel"
    elif location.get("xyz_name") is not None and features["location_type"] is None:
        features["location_type"] = "named_location"
    return features


def write_dict_rows(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_counter_rows(counter: Counter, name_field: str, value_field: str = "count") -> List[Dict]:
    rows = []
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
        rows.append({name_field: key, value_field: value})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real FOA annotations and summarize dataset statistics.")
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("/apdcephfs_cq10/share_1603164/user/schmittzhu/data/RealFOA"),
    )
    parser.add_argument(
        "--anno-root",
        type=Path,
        default=Path("/apdcephfs_cq10/share_1603164/user/schmittzhu/data/metadata/real_anno"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("check_reports"))
    parser.add_argument("--verbose", action="store_true", help="Print per-recording progress.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("Scanning audio directories...", flush=True)
    audio_dirs = {path.relative_to(args.audio_root).as_posix(): path for path in iter_recording_dirs(args.audio_root)}
    if args.verbose:
        print("Scanning annotation directories...", flush=True)
    anno_dirs = {path.relative_to(args.anno_root).as_posix(): path for path in iter_recording_dirs(args.anno_root)}

    missing_annotation_dirs = sorted(set(audio_dirs) - set(anno_dirs))
    missing_audio_dirs = sorted(set(anno_dirs) - set(audio_dirs))
    common_keys = sorted(set(audio_dirs) & set(anno_dirs))

    class_counter = Counter()
    class_duration_counter = Counter()
    heading_counter = Counter()
    heading_bin_counter = Counter()
    elevation_counter = Counter()
    elevation_bin_counter = Counter()
    distance_counter = Counter()
    distance_bin_counter = Counter()
    location_type_counter = Counter()

    invalid_events = []
    parse_errors = []
    missing_annotation_details = []
    per_audio_rows = []

    total_audio_duration = 0.0
    total_covered_duration = 0.0
    total_annotations = 0

    for index, rel_key in enumerate(common_keys, start=1):
        if args.verbose:
            print(f"[{index}/{len(common_keys)}] processing {rel_key}", flush=True)
        audio_dir = audio_dirs[rel_key]
        anno_dir = anno_dirs[rel_key]
        wav_path = audio_dir / "aligned_foa_bformat.wav"

        json_path, json_candidates = find_single_json(anno_dir)
        if not wav_path.exists():
            missing_annotation_details.append(
                {"recording": rel_key, "issue": "missing_audio_file", "detail": str(wav_path)}
            )
            continue
        if len(json_candidates) != 1:
            missing_annotation_details.append(
                {
                    "recording": rel_key,
                    "issue": "invalid_json_file_count",
                    "detail": ",".join(path.name for path in json_candidates) or "no_json",
                }
            )
            continue

        try:
            duration = wav_duration_seconds(wav_path)
        except Exception as exc:
            parse_errors.append({"recording": rel_key, "path": str(wav_path), "error": str(exc)})
            continue

        try:
            with json_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as exc:
            parse_errors.append({"recording": rel_key, "path": str(json_path), "error": str(exc)})
            continue

        annotations = metadata.get("annotations")
        if annotations is None:
            missing_annotation_details.append(
                {"recording": rel_key, "issue": "missing_annotations_field", "detail": str(json_path.name)}
            )
            annotations = []
        elif not isinstance(annotations, list):
            missing_annotation_details.append(
                {"recording": rel_key, "issue": "annotations_not_list", "detail": str(type(annotations).__name__)}
            )
            annotations = []

        total_audio_duration += duration
        audio_intervals = []
        invalid_count_this_audio = 0

        for idx, ann in enumerate(annotations):
            total_annotations += 1
            if not isinstance(ann, dict):
                invalid_events.append(
                    {
                        "recording": rel_key,
                        "annotation_index": idx,
                        "event_name": None,
                        "issue": "annotation_not_dict",
                        "start_time": None,
                        "end_time": None,
                        "audio_duration": duration,
                    }
                )
                invalid_count_this_audio += 1
                continue

            event_name = ann.get("event_name")
            start = safe_float(ann.get("start_time"))
            end = safe_float(ann.get("end_time"))
            location = ann.get("location")
            issues = []

            if not event_name:
                issues.append("missing_event_name")
            if start is None:
                issues.append("missing_or_invalid_start_time")
            if end is None:
                issues.append("missing_or_invalid_end_time")

            if start is not None and start < -TIME_TOLERANCE:
                issues.append("start_time_negative")
            if end is not None and end < -TIME_TOLERANCE:
                issues.append("end_time_negative")
            if start is not None and end is not None and end <= start + TIME_TOLERANCE:
                issues.append("non_positive_duration")
            if start is not None and start > duration + TIME_TOLERANCE:
                issues.append("start_time_exceeds_audio_duration")
            if end is not None and end > duration + TIME_TOLERANCE:
                issues.append("end_time_exceeds_audio_duration")
            if location is None:
                issues.append("missing_location")

            spatial = infer_spatial_features(location)
            if location is not None and spatial["location_type"] is None:
                issues.append("unrecognized_location_format")

            if issues:
                for issue in issues:
                    invalid_events.append(
                        {
                            "recording": rel_key,
                            "annotation_index": idx,
                            "event_name": event_name,
                            "issue": issue,
                            "start_time": start,
                            "end_time": end,
                            "audio_duration": duration,
                        }
                    )
                invalid_count_this_audio += 1

            if event_name:
                class_counter[event_name] += 1
                if start is not None and end is not None and end > start:
                    class_duration_counter[event_name] += end - start

            if start is not None and end is not None:
                clipped_start = min(max(start, 0.0), duration)
                clipped_end = min(max(end, 0.0), duration)
                if clipped_end > clipped_start:
                    audio_intervals.append((clipped_start, clipped_end))

            heading = spatial["heading_deg"]
            elevation = spatial["elevation_deg"]
            distance = spatial["distance"]
            location_type = spatial["location_type"]

            if location_type:
                location_type_counter[location_type] += 1
            if heading is not None:
                heading_counter[round(heading, 3)] += 1
                heading_bin_counter[heading_bin_label(heading)] += 1
            if elevation is not None:
                elevation_counter[round(elevation, 3)] += 1
                elevation_bin_counter[centered_elevation_bin_label(elevation)] += 1
            if distance is not None:
                distance_counter[round(distance, 3)] += 1
                distance_bin_counter[distance_bin_label(distance)] += 1

        covered_duration = interval_union_length(audio_intervals)
        total_covered_duration += covered_duration
        coverage_ratio = covered_duration / duration if duration > 0 else 0.0
        if len(annotations) == 0:
            missing_annotation_details.append(
                {"recording": rel_key, "issue": "empty_annotations", "detail": str(json_path.name)}
            )

        per_audio_rows.append(
            {
                "recording": rel_key,
                "audio_duration_sec": round(duration, 6),
                "num_annotations": len(annotations),
                "invalid_event_count": invalid_count_this_audio,
                "covered_duration_sec": round(covered_duration, 6),
                "coverage_ratio": round(coverage_ratio, 6),
                "json_file": json_path.name,
            }
        )

    for rel_key in missing_annotation_dirs:
        missing_annotation_details.append({"recording": rel_key, "issue": "missing_annotation_dir", "detail": ""})
    for rel_key in missing_audio_dirs:
        missing_annotation_details.append({"recording": rel_key, "issue": "missing_audio_dir", "detail": ""})

    summary = {
        "audio_root": str(args.audio_root),
        "anno_root": str(args.anno_root),
        "num_audio_recordings": len(audio_dirs),
        "num_annotation_recordings": len(anno_dirs),
        "num_paired_recordings": len(common_keys),
        "missing_annotation_dir_count": len(missing_annotation_dirs),
        "missing_audio_dir_count": len(missing_audio_dirs),
        "missing_annotation_issue_count": len(missing_annotation_details),
        "parse_error_count": len(parse_errors),
        "invalid_event_issue_count": len(invalid_events),
        "total_annotations": total_annotations,
        "total_audio_duration_sec": round(total_audio_duration, 6),
        "total_covered_duration_sec": round(total_covered_duration, 6),
        "overall_coverage_ratio": round(total_covered_duration / total_audio_duration, 6)
        if total_audio_duration > 0
        else 0.0,
        "num_classes": len(class_counter),
        "num_recordings_with_empty_annotations": sum(
            1 for item in missing_annotation_details if item["issue"] == "empty_annotations"
        ),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_dict_rows(
        output_dir / "per_audio_summary.csv",
        [
            "recording",
            "audio_duration_sec",
            "num_annotations",
            "invalid_event_count",
            "covered_duration_sec",
            "coverage_ratio",
            "json_file",
        ],
        per_audio_rows,
    )
    write_dict_rows(
        output_dir / "invalid_events.csv",
        ["recording", "annotation_index", "event_name", "issue", "start_time", "end_time", "audio_duration"],
        invalid_events,
    )
    write_dict_rows(
        output_dir / "missing_annotation_details.csv",
        ["recording", "issue", "detail"],
        missing_annotation_details,
    )
    write_dict_rows(output_dir / "parse_errors.csv", ["recording", "path", "error"], parse_errors)

    class_rows = []
    for event_name, count in sorted(class_counter.items(), key=lambda item: (-item[1], item[0])):
        class_rows.append(
            {
                "event_name": event_name,
                "count": count,
                "total_duration_sec": round(class_duration_counter[event_name], 6),
            }
        )
    write_dict_rows(output_dir / "class_distribution.csv", ["event_name", "count", "total_duration_sec"], class_rows)
    write_dict_rows(output_dir / "location_type_distribution.csv", ["location_type", "count"], build_counter_rows(location_type_counter, "location_type"))
    write_dict_rows(output_dir / "heading_distribution.csv", ["heading_deg", "count"], build_counter_rows(heading_counter, "heading_deg"))
    write_dict_rows(output_dir / "heading_bins.csv", ["heading_bin", "count"], build_counter_rows(heading_bin_counter, "heading_bin"))
    write_dict_rows(output_dir / "elevation_distribution.csv", ["elevation_deg", "count"], build_counter_rows(elevation_counter, "elevation_deg"))
    write_dict_rows(output_dir / "elevation_bins.csv", ["elevation_bin", "count"], build_counter_rows(elevation_bin_counter, "elevation_bin"))
    write_dict_rows(output_dir / "distance_distribution.csv", ["distance", "count"], build_counter_rows(distance_counter, "distance"))
    write_dict_rows(output_dir / "distance_bins.csv", ["distance_bin", "count"], build_counter_rows(distance_bin_counter, "distance_bin"))

    print("Annotation check completed.")
    print(f"Summary saved to: {summary_path}")
    print(f"Paired recordings: {summary['num_paired_recordings']}")
    print(f"Total annotations: {summary['total_annotations']}")
    print(f"Invalid event issues: {summary['invalid_event_issue_count']}")
    print(f"Missing annotation issues: {summary['missing_annotation_issue_count']}")
    print(f"Overall coverage ratio: {summary['overall_coverage_ratio']:.4f}")
    print(f"Number of classes: {summary['num_classes']}")


if __name__ == "__main__":
    main()
