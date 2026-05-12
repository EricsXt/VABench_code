#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any
from typing import Tuple


def normalize_signed_azimuth(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def maybe_normalize_heading(location: Any) -> bool:
    if not isinstance(location, dict) or "heading" not in location:
        return False
    try:
        original = float(location["heading"])
    except (TypeError, ValueError):
        return False

    normalized = normalize_signed_azimuth(original)
    if abs(normalized - original) < 1e-9:
        return False

    # Keep integer-looking headings as ints, otherwise preserve float precision.
    if abs(normalized - round(normalized)) < 1e-9:
        location["heading"] = int(round(normalized))
    else:
        location["heading"] = round(normalized, 6)
    return True


def process_file(path: Path, dry_run: bool = False) -> Tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed_count = 0
    seen_count = 0

    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        for key in ("location", "location_end"):
            location = annotation.get(key)
            if isinstance(location, dict) and "heading" in location:
                seen_count += 1
                if maybe_normalize_heading(location):
                    changed_count += 1

    if changed_count and not dry_run:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return seen_count, changed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize HF JSON heading values from any degree range into [-180, 180)."
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path("/data/zhuzhiyuan/starss23/SpatialQA_hf/json"),
        help="Directory containing flat JSON files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_files = sorted(args.json_root.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {args.json_root}")

    total_files_changed = 0
    total_headings_seen = 0
    total_headings_changed = 0

    for path in json_files:
        seen_count, changed_count = process_file(path, dry_run=args.dry_run)
        total_headings_seen += seen_count
        total_headings_changed += changed_count
        if changed_count:
            total_files_changed += 1
            print(f"{path.name}: changed {changed_count} heading values")

    mode = "Dry run" if args.dry_run else "Done"
    print(
        f"{mode}. files_changed={total_files_changed}, "
        f"headings_seen={total_headings_seen}, headings_changed={total_headings_changed}"
    )


if __name__ == "__main__":
    main()
