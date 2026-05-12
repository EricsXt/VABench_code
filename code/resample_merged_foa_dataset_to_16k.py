#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resample a merged FOA SELD dataset to 16 kHz while preserving metadata/video/manifests."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/merged_seld_foa_starss23_spatialqa_20s_16k"),
    )
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        raise FileExistsError(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def resample_one(src: Path, dst: Path, target_sr: int):
    audio, sr = sf.read(str(src), always_2d=True)
    subtype = sf.info(str(src)).subtype
    if sr != target_sr:
        audio = resample_poly(audio, up=target_sr, down=sr, axis=0)
    sf.write(str(dst), audio, target_sr, subtype=subtype)
    return src.name, sr, target_sr, audio.shape[0], audio.shape[1]


def main():
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")

    input_audio = args.input_root / "foa_dev"
    input_meta = args.input_root / "metadata_dev"
    input_video = args.input_root / "video_dev"
    output_audio = args.output_root / "foa_dev"
    output_meta = args.output_root / "metadata_dev"
    output_video = args.output_root / "video_dev"

    ensure_dir(output_audio)
    ensure_dir(output_meta)
    ensure_dir(output_video)

    for src in sorted(input_meta.glob("*.csv")):
        link_or_copy(src, output_meta / src.name)
    for src in sorted(input_video.glob("*.mp4")):
        link_or_copy(src, output_video / src.name)
    for name in ["class_mapping.json", "split_manifest.json", "source_manifest.json", "README.md", "merge_summary.json"]:
        src = args.input_root / name
        if src.exists():
            link_or_copy(src, args.output_root / name)

    wavs = sorted(input_audio.glob("*.wav"))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(resample_one, src, output_audio / src.name, args.target_sr): src
            for src in wavs
        }
        for future in as_completed(futures):
            results.append(future.result())

    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "target_sr": args.target_sr,
        "audio_files": len(wavs),
        "metadata_files": len(list(input_meta.glob("*.csv"))),
        "video_files": len(list(input_video.glob("*.mp4"))),
    }
    (args.output_root / "resample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
