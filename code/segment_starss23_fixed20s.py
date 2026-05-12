#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
from pathlib import Path

import soundfile as sf


LABEL_HOP_SECONDS = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a STARSS23-style dataset into fixed 20-second segments for audio, video, and CSV metadata."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/STARSS23"),
        help="Dataset root containing foa_dev/, video_dev/, and metadata_dev/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/STARSS23_20s"),
        help="Output dataset root.",
    )
    parser.add_argument("--segment-seconds", type=float, default=20.0)
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg")
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
    (path / "foa_dev").mkdir(parents=True, exist_ok=True)
    (path / "video_dev").mkdir(parents=True, exist_ok=True)
    (path / "metadata_dev").mkdir(parents=True, exist_ok=True)


def segment_stem(stem: str, index: int) -> str:
    return f"{stem}_seg{index:03d}"


def write_audio_segments(input_wav: Path, output_dir: Path, segment_seconds: float) -> list:
    audio, sr = sf.read(str(input_wav), dtype="float32", always_2d=True)
    total_seconds = len(audio) / float(sr)
    num_segments = max(1, int(math.ceil(total_seconds / segment_seconds)))
    stems = []
    for idx in range(num_segments):
        start_sec = idx * segment_seconds
        end_sec = min(total_seconds, (idx + 1) * segment_seconds)
        start_sample = int(round(start_sec * sr))
        end_sample = int(round(end_sec * sr))
        out_stem = segment_stem(input_wav.stem, idx)
        out_path = output_dir / f"{out_stem}.wav"
        sf.write(str(out_path), audio[start_sample:end_sample], sr)
        stems.append((out_stem, start_sec, end_sec))
    return stems


def write_csv_segments(input_csv: Path, output_dir: Path, segments: list, segment_seconds: float) -> None:
    rows = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                rows.append(row)

    for idx, (out_stem, start_sec, end_sec) in enumerate(segments):
        start_frame = int(round(start_sec / LABEL_HOP_SECONDS))
        end_frame = int(round(end_sec / LABEL_HOP_SECONDS))
        out_path = output_dir / f"{out_stem}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for row in rows:
                frame_idx = int(row[0])
                if start_frame <= frame_idx < end_frame:
                    new_row = list(row)
                    new_row[0] = str(frame_idx - start_frame)
                    writer.writerow(new_row)


def write_video_segments(input_mp4: Path, output_dir: Path, segments: list, ffmpeg_bin: str) -> None:
    for out_stem, start_sec, end_sec in segments:
        out_path = output_dir / f"{out_stem}.mp4"
        duration = end_sec - start_sec
        command = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{start_sec:.6f}",
            "-i",
            str(input_mp4),
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
            str(out_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> None:
    args = parse_args()
    ensure_output_root(args.output_root, args.overwrite)

    input_foa = args.input_root / "foa_dev"
    input_video = args.input_root / "video_dev"
    input_meta = args.input_root / "metadata_dev"

    audio_files = sorted(input_foa.rglob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No wav files found under {input_foa}")

    for audio_path in audio_files:
        rel_parent = audio_path.parent.relative_to(input_foa)
        out_audio_dir = args.output_root / "foa_dev" / rel_parent
        out_video_dir = args.output_root / "video_dev" / rel_parent
        out_meta_dir = args.output_root / "metadata_dev" / rel_parent
        out_audio_dir.mkdir(parents=True, exist_ok=True)
        out_video_dir.mkdir(parents=True, exist_ok=True)
        out_meta_dir.mkdir(parents=True, exist_ok=True)

        segments = write_audio_segments(audio_path, out_audio_dir, args.segment_seconds)

        csv_path = input_meta / rel_parent / f"{audio_path.stem}.csv"
        if csv_path.exists():
            write_csv_segments(csv_path, out_meta_dir, segments, args.segment_seconds)

        mp4_path = input_video / rel_parent / f"{audio_path.stem}.mp4"
        if mp4_path.exists():
            write_video_segments(mp4_path, out_video_dir, segments, args.ffmpeg_bin)

    print(f"Segmented dataset written to: {args.output_root}")


if __name__ == "__main__":
    main()
