#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a manifest only for the public release subset:
train_3000.csv UNION test_main.csv.

Example:
  python tools/generate_video_manifest.py ^
      --video-dir D:/public_videos ^
      --output-csv D:/public_videos/video_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_splits_dir = repo_root / "splits" / "split_v3"

    parser = argparse.ArgumentParser(
        description="Generate video_manifest.csv for train_3000 + test_main."
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Root directory that contains the public videos.",
    )
    parser.add_argument(
        "--splits-dir",
        default=str(default_splits_dir),
        help="Directory that contains train_3000.csv and test_main.csv.",
    )
    parser.add_argument(
        "--train-csv",
        default="train_3000.csv",
        help="Training split filename. Default: train_3000.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="test_main.csv",
        help="Test split filename. Default: test_main.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="video_manifest.csv",
        help="Output CSV path. Default: ./video_manifest.csv",
    )
    parser.add_argument(
        "--extensions",
        default=".mp4,.avi,.mov,.mkv,.webm,.m4v",
        help="Comma-separated video extensions to scan.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Return non-zero if any required video_id is missing.",
    )
    return parser.parse_args()


def load_split_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "video_id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV missing video_id column: {csv_path}")
        return {
            str(row["video_id"]).strip()
            for row in reader
            if str(row.get("video_id", "")).strip()
        }


def build_video_index(video_dir: Path, extensions: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}

    for path in sorted(video_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue

        video_id = path.stem
        rel_path = path.relative_to(video_dir).as_posix()

        if video_id in index:
            duplicates.setdefault(video_id, [index[video_id].relative_to(video_dir).as_posix()])
            duplicates[video_id].append(rel_path)
            continue

        index[video_id] = path

    if duplicates:
        duplicate_msgs = [f"{video_id}: {paths}" for video_id, paths in sorted(duplicates.items())]
        raise ValueError("Duplicated video_id files found:\n" + "\n".join(duplicate_msgs))

    return index


def write_manifest(
    output_csv: Path,
    video_dir: Path,
    public_ids: set[str],
    train_ids: set[str],
    test_ids: set[str],
    video_index: dict[str, Path],
) -> list[str]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    missing_ids: list[str] = []
    rows: list[dict[str, object]] = []

    for video_id in sorted(public_ids):
        path = video_index.get(video_id)
        if path is None:
            missing_ids.append(video_id)
            continue

        stat = path.stat()
        rows.append(
            {
                "video_id": video_id,
                "filename": path.name,
                "rel_path": path.relative_to(video_dir).as_posix(),
                "size_bytes": stat.st_size,
                "in_train_3000": int(video_id in train_ids),
                "in_test_main": int(video_id in test_ids),
            }
        )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "filename",
                "rel_path",
                "size_bytes",
                "in_train_3000",
                "in_test_main",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return missing_ids


def main() -> int:
    args = parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    splits_dir = Path(args.splits_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not video_dir.exists() or not video_dir.is_dir():
        print(f"[ERROR] Invalid video-dir: {video_dir}")
        return 1
    if not splits_dir.exists() or not splits_dir.is_dir():
        print(f"[ERROR] Invalid splits-dir: {splits_dir}")
        return 1

    extensions = {
        ext.strip().lower()
        for ext in args.extensions.split(",")
        if ext.strip()
    }

    train_ids = load_split_ids(splits_dir / args.train_csv)
    test_ids = load_split_ids(splits_dir / args.test_csv)
    public_ids = train_ids | test_ids
    video_index = build_video_index(video_dir, extensions)
    missing_ids = write_manifest(
        output_csv=output_csv,
        video_dir=video_dir,
        public_ids=public_ids,
        train_ids=train_ids,
        test_ids=test_ids,
        video_index=video_index,
    )

    print(f"[DONE] Output: {output_csv}")
    print(f"[DONE] train_3000 ids: {len(train_ids)}")
    print(f"[DONE] test_main ids: {len(test_ids)}")
    print(f"[DONE] public union ids: {len(public_ids)}")
    print(f"[DONE] indexed videos written: {len(public_ids) - len(missing_ids)}")

    if missing_ids:
        print(f"[WARN] Missing video count: {len(missing_ids)}")
        for video_id in missing_ids[:20]:
            print(f"[WARN] missing: {video_id}")
        if len(missing_ids) > 20:
            print(f"[WARN] ... and {len(missing_ids) - 20} more")
        if args.fail_on_missing:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

