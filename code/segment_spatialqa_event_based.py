#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf


SEGMENT_MIN_SECONDS = 5.0
SEGMENT_MAX_SECONDS = 20.0
SEGMENT_TARGET_SECONDS = 12.0
MERGE_GAP_SECONDS = 1.5


@dataclass
class Event:
    start_time: float
    end_time: float


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split flat SpatialQA_hf audio/json/video files using the same event-driven logic as prepare_real_foa_to_dcase.py."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/SpatialQA_hf"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/SpatialQA_hf_segmented"),
    )
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
    parser.add_argument("--min-seconds", type=float, default=SEGMENT_MIN_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=SEGMENT_MAX_SECONDS)
    parser.add_argument("--target-seconds", type=float, default=SEGMENT_TARGET_SECONDS)
    parser.add_argument("--merge-gap-seconds", type=float, default=MERGE_GAP_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Output root already exists: {path}. Re-run with --overwrite.")
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)
    for sub in ("audio", "json", "visual"):
        (path / sub).mkdir(parents=True, exist_ok=True)


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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


def clip_json_annotations(payload: dict, segment_start: float, segment_end: float) -> dict:
    clipped_payload = {key: value for key, value in payload.items() if key != "annotations"}
    clipped_annotations = []
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        start = safe_float(annotation.get("start_time"))
        end = safe_float(annotation.get("end_time"))
        clipped = clip_annotation_times(start, end, segment_end)
        if clipped is None:
            continue
        clip_start, clip_end = clipped
        if clip_end <= segment_start or clip_start >= segment_end:
            continue
        seg_start = max(clip_start, segment_start)
        seg_end = min(clip_end, segment_end)
        if seg_end <= seg_start:
            continue
        ann_copy = dict(annotation)
        ann_copy["start_time"] = round(seg_start - segment_start, 6)
        ann_copy["end_time"] = round(seg_end - segment_start, 6)
        clipped_annotations.append(ann_copy)
    clipped_payload["annotations"] = clipped_annotations
    clipped_payload["segment_start_time"] = round(segment_start, 6)
    clipped_payload["segment_end_time"] = round(segment_end, 6)
    clipped_payload["segment_duration"] = round(segment_end - segment_start, 6)
    return clipped_payload


def export_video_segment(ffmpeg_bin: str, input_path: Path, output_path: Path, start: float, duration: float) -> None:
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


def main() -> None:
    args = parse_args()
    ensure_output_root(args.output_root, args.overwrite)

    audio_root = args.input_root / "audio"
    json_root = args.input_root / "json"
    visual_root = args.input_root / "visual"

    audio_files = sorted(audio_root.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No wav files found under {audio_root}")

    for audio_path in audio_files:
        stem = audio_path.stem
        json_path = json_root / f"{stem}.json"
        video_path = visual_root / f"{stem}.mp4"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing json file for {stem}: {json_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"Missing mp4 file for {stem}: {video_path}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio_duration = len(audio) / float(sr)

        events: List[Event] = []
        for annotation in payload.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            clipped = clip_annotation_times(
                safe_float(annotation.get("start_time")),
                safe_float(annotation.get("end_time")),
                audio_duration,
            )
            if clipped is None:
                continue
            events.append(Event(start_time=clipped[0], end_time=clipped[1]))

        if not events:
            continue

        segments = build_segments(
            events,
            audio_duration,
            args.min_seconds,
            args.max_seconds,
            args.target_seconds,
            args.merge_gap_seconds,
        )

        for segment_index, segment in enumerate(segments):
            out_stem = f"{stem}_seg{segment_index:04d}"
            start_sample = int(round(segment.start * sr))
            end_sample = int(round(segment.end * sr))

            out_audio = args.output_root / "audio" / f"{out_stem}.wav"
            out_json = args.output_root / "json" / f"{out_stem}.json"
            out_video = args.output_root / "visual" / f"{out_stem}.mp4"

            sf.write(str(out_audio), audio[start_sample:end_sample], sr)
            out_json.write_text(
                json.dumps(clip_json_annotations(payload, segment.start, segment.end), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            export_video_segment(args.ffmpeg_bin, video_path, out_video, segment.start, segment.duration)

    print(f"Segmented dataset written to: {args.output_root}")


if __name__ == "__main__":
    main()
