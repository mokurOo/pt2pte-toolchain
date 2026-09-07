"""Calibration sample discovery."""

from __future__ import annotations

import json
import random
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sample_yolo_ndjson(
    ndjson: str | Path,
    images_root: str | Path,
    *,
    split: str,
    count: int,
    seed: int,
) -> list[Path]:
    ndjson_path = Path(ndjson).expanduser().resolve()
    split_dir = Path(images_root).expanduser().resolve() / split
    if not ndjson_path.is_file():
        raise FileNotFoundError(f"NDJSON file does not exist: {ndjson_path}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"calibration directory for split {split!r} does not exist: {split_dir}")
    if count <= 0:
        raise ValueError("calibration count must be greater than zero")

    candidates: list[Path] = []
    with ndjson_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid NDJSON at line {line_number}: {error}") from error
            if row.get("type") == "image" and row.get("split") == split:
                candidates.append(split_dir / str(row["file"]))

    if len(candidates) < count:
        raise ValueError(
            f"split {split!r} contains {len(candidates)} records, fewer than requested {count}"
        )
    missing = [path for path in candidates if not path.is_file() or path.stat().st_size == 0]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"NDJSON references missing or empty images: {preview}")

    rng = random.Random(seed)
    return rng.sample(candidates, count)


def sample_image_directory(
    images_root: str | Path,
    *,
    count: int,
    seed: int,
) -> list[Path]:
    root = Path(images_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"calibration image directory does not exist: {root}")
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stat().st_size > 0
    )
    if len(candidates) < count:
        raise ValueError(
            f"calibration directory contains {len(candidates)} images, fewer than requested {count}"
        )
    return random.Random(seed).sample(candidates, count)
