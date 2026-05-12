#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


LABEL_HOP_SECONDS = 0.1
MISSING_DISTANCE_CM = -1


@dataclass
class Event:
    annotation_index: int
    raw_event_name: str
    mapped_event_name: str
    start_time: float
    end_time: float
    moving: int
    location: Optional[dict]
    location_end: Optional[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SpatialQA-style JSON annotations into STARSS23/DCASE-style CSV metadata."
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/SpatialQA_hf/json"),
        help="Directory containing flat JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/SpatialQA_hf_csv"),
        help="Directory where CSV files and class_mapping.json will be written.",
    )
    parser.add_argument(
        "--class-mapping-path",
        type=Path,
        default=None,
        help="Optional existing class_mapping.json with class_name_to_id. If omitted, a new mapping is generated.",
    )
    parser.add_argument(
        "--event-mapping-csv",
        type=Path,
        default=None,
        help="Optional CSV with raw_event_name,target_class. Empty/0 targets are skipped.",
    )
    parser.add_argument(
        "--label-hop-seconds",
        type=float,
        default=LABEL_HOP_SECONDS,
    )
    parser.add_argument(
        "--require-precise-pose",
        action="store_true",
        help="Keep only annotations with precise geometry from position_rel. Moving events require precise start and end positions.",
    )
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


def position_rel_to_listener_frame(position_rel: Sequence[float]) -> Tuple[float, float, float]:
    x_front = float(position_rel[0])
    y_left = float(position_rel[1])
    z_up = float(position_rel[2])
    return x_front, y_left, z_up


def has_precise_location(location: Optional[dict]) -> bool:
    return isinstance(location, dict) and isinstance(location.get("position_rel"), list) and len(location.get("position_rel")) >= 3


def event_has_precise_pose(event: Event) -> bool:
    if not has_precise_location(event.location):
        return False
    if event.moving == 1:
        loc1 = event.location_end if isinstance(event.location_end, dict) else event.location
        return has_precise_location(loc1)
    return True


def listener_frame_to_dcase_pose(x_front: float, y_left: float, z_up: float) -> Tuple[float, float, int]:
    distance_m = math.sqrt(x_front * x_front + y_left * y_left + z_up * z_up)
    azimuth = normalize_signed_azimuth(math.degrees(math.atan2(y_left, x_front)))
    elevation = math.degrees(math.atan2(z_up, math.hypot(x_front, y_left))) if distance_m > 0 else 0.0
    distance_cm = int(round(distance_m * 100.0))
    return azimuth, elevation, distance_cm


def infer_pose_from_location(location: Optional[dict]) -> Optional[Tuple[float, float, int]]:
    if not isinstance(location, dict):
        return None

    position_rel = location.get("position_rel")
    if isinstance(position_rel, list) and len(position_rel) >= 3:
        x_front, y_left, z_up = position_rel_to_listener_frame(position_rel)
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


def infer_pose_for_event(event: Event, t_global: float) -> Optional[Tuple[float, float, int]]:
    duration = max(event.end_time - event.start_time, 1e-6)
    alpha = min(max((t_global - event.start_time) / duration, 0.0), 1.0)
    loc0 = event.location
    loc1 = event.location_end if isinstance(event.location_end, dict) else loc0

    if event.moving == 1 and isinstance(loc0, dict) and isinstance(loc1, dict):
        pos0 = loc0.get("position_rel")
        pos1 = loc1.get("position_rel")
        if isinstance(pos0, list) and len(pos0) >= 3 and isinstance(pos1, list) and len(pos1) >= 3:
            x0, y0, z0 = position_rel_to_listener_frame(pos0)
            x1, y1, z1 = position_rel_to_listener_frame(pos1)
            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)
            z = z0 + alpha * (z1 - z0)
            return listener_frame_to_dcase_pose(x, y, z)

        pose0 = infer_pose_from_location(loc0)
        pose1 = infer_pose_from_location(loc1)
        if pose0 is not None and pose1 is not None:
            azimuth = interpolate_angle_deg(pose0[0], pose1[0], alpha)
            elev0, elev1 = pose0[1], pose1[1]
            if math.isfinite(elev0) and math.isfinite(elev1):
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

    return infer_pose_from_location(loc0)


def load_class_mapping(path: Optional[Path], mapped_event_names: List[str]) -> Dict[str, int]:
    if path is not None and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = payload.get("class_name_to_id")
        if not isinstance(mapping, dict):
            raise ValueError(f"class_name_to_id missing in {path}")
        return {str(key): int(value) for key, value in mapping.items()}

    labels = sorted(set(mapped_event_names))
    return {label: idx for idx, label in enumerate(labels)}


def load_event_mapping(path: Optional[Path]) -> Optional[Dict[str, str]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Event mapping CSV not found: {path}")

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "raw_event_name" not in (reader.fieldnames or []) or "target_class" not in (reader.fieldnames or []):
            raise ValueError("Event mapping CSV must contain raw_event_name and target_class columns")
        for row in reader:
            raw_name = str(row.get("raw_event_name", "")).strip()
            target_name = str(row.get("target_class", "")).strip()
            if not raw_name:
                continue
            if not target_name or target_name == "0":
                continue
            mapping[raw_name] = target_name
    return mapping


def write_class_mapping(output_root: Path, class_name_to_id: Dict[str, int]) -> None:
    class_id_to_name = {str(class_id): label for label, class_id in sorted(class_name_to_id.items(), key=lambda x: x[1])}
    payload = {
        "class_count": len(class_name_to_id),
        "class_names": [label for label, _ in sorted(class_name_to_id.items(), key=lambda x: x[1])],
        "class_name_to_id": class_name_to_id,
        "class_id_to_name": class_id_to_name,
        "missing_distance_cm": MISSING_DISTANCE_CM,
    }
    (output_root / "class_mapping.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clip_annotation_times(start: Optional[float], end: Optional[float], duration: float) -> Optional[Tuple[float, float]]:
    if start is None or end is None:
        return None
    start = min(max(start, 0.0), duration)
    end = min(max(end, 0.0), duration)
    if end <= start:
        return None
    return start, end


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}"


def convert_file(
    json_path: Path,
    output_csv_path: Path,
    class_name_to_id: Dict[str, int],
    label_hop_seconds: float,
    event_name_mapping: Optional[Dict[str, str]],
    require_precise_pose: bool,
) -> Dict[str, int]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    audio_duration = safe_float(payload.get("aligned_duration")) or safe_float(payload.get("audio_time")) or 0.0
    events: List[Event] = []
    stats = {"annotations_total": 0, "annotations_kept": 0, "annotations_skipped_imprecise": 0}

    for annotation_index, ann in enumerate(payload.get("annotations", [])):
        if not isinstance(ann, dict):
            continue
        stats["annotations_total"] += 1
        raw_name = str(ann.get("event_name", "")).strip()
        if not raw_name:
            continue
        mapped_name = event_name_mapping.get(raw_name, raw_name) if event_name_mapping is not None else raw_name
        if not mapped_name:
            continue
        clipped = clip_annotation_times(safe_float(ann.get("start_time")), safe_float(ann.get("end_time")), audio_duration)
        if clipped is None:
            continue
        if mapped_name not in class_name_to_id:
            continue
        event = Event(
            annotation_index=annotation_index,
            raw_event_name=raw_name,
            mapped_event_name=mapped_name,
            start_time=clipped[0],
            end_time=clipped[1],
            moving=int(ann.get("moving", 0) or 0),
            location=ann.get("location"),
            location_end=ann.get("location_end"),
        )
        if require_precise_pose and not event_has_precise_pose(event):
            stats["annotations_skipped_imprecise"] += 1
            continue
        events.append(event)
        stats["annotations_kept"] += 1

    rows: List[List[str]] = []
    for event in events:
        frame_start = max(0, int(math.floor(event.start_time / label_hop_seconds)))
        frame_end = max(frame_start, int(math.ceil(event.end_time / label_hop_seconds)) - 1)
        class_id = class_name_to_id[event.mapped_event_name]
        track_id = event.annotation_index
        for frame_idx in range(frame_start, frame_end + 1):
            frame_begin = frame_idx * label_hop_seconds
            frame_end_time = frame_begin + label_hop_seconds
            if min(event.end_time, frame_end_time) <= max(event.start_time, frame_begin):
                continue
            t_global = min(max((frame_begin + frame_end_time) / 2.0, event.start_time), event.end_time)
            pose = infer_pose_for_event(event, t_global)
            if pose is None:
                continue
            azimuth, elevation, distance_cm = pose
            rows.append([
                str(frame_idx),
                str(class_id),
                str(track_id),
                format_float(azimuth),
                format_float(elevation),
                str(int(distance_cm)),
            ])

    rows.sort(key=lambda row: (int(row[0]), int(row[1]), int(row[2])))
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return stats


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    event_name_mapping = load_event_mapping(args.event_mapping_csv)

    json_files = sorted(args.json_root.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {args.json_root}")

    mapped_event_names: List[str] = []
    for json_path in json_files:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for ann in payload.get("annotations", []):
            if isinstance(ann, dict):
                raw_name = str(ann.get("event_name", "")).strip()
                if raw_name:
                    mapped_name = event_name_mapping.get(raw_name, raw_name) if event_name_mapping is not None else raw_name
                    if mapped_name:
                        mapped_event_names.append(mapped_name)

    class_name_to_id = load_class_mapping(args.class_mapping_path, mapped_event_names)
    write_class_mapping(args.output_root, class_name_to_id)

    total_stats = {"annotations_total": 0, "annotations_kept": 0, "annotations_skipped_imprecise": 0}
    for json_path in json_files:
        output_csv_path = args.output_root / f"{json_path.stem}.csv"
        file_stats = convert_file(
            json_path,
            output_csv_path,
            class_name_to_id,
            args.label_hop_seconds,
            event_name_mapping,
            args.require_precise_pose,
        )
        for key, value in file_stats.items():
            total_stats[key] += value

    print(f"Converted {len(json_files)} JSON files to CSV under {args.output_root}")
    print(f"Class count: {len(class_name_to_id)}")
    print(
        "Annotations total={annotations_total} kept={annotations_kept} skipped_imprecise={annotations_skipped_imprecise}".format(
            **total_stats
        )
    )


if __name__ == "__main__":
    main()
