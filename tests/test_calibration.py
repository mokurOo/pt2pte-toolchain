import json
from pathlib import Path

import pt2pte.calibration as calibration_module


def _write_ndjson(path: Path) -> None:
    rows = [
        {"type": "dataset", "name": "fixture"},
        {"type": "image", "split": "train", "file": "a.jpg"},
        {"type": "image", "split": "val", "file": "v.jpg"},
        {"type": "image", "split": "train", "file": "b.jpg"},
        {"type": "image", "split": "train", "file": "c.jpg"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_sample_yolo_ndjson_uses_only_requested_split_and_is_deterministic(tmp_path):
    ndjson = tmp_path / "data.ndjson"
    images_root = tmp_path / "images"
    (images_root / "train").mkdir(parents=True)
    (images_root / "val").mkdir(parents=True)
    _write_ndjson(ndjson)
    for relative in ("train/a.jpg", "train/b.jpg", "train/c.jpg", "val/v.jpg"):
        (images_root / relative).write_bytes(b"image")

    sample_paths = getattr(calibration_module, "sample_yolo_ndjson", None)
    assert callable(sample_paths)
    first = sample_paths(ndjson, images_root, split="train", count=2, seed=7)
    second = sample_paths(ndjson, images_root, split="train", count=2, seed=7)

    assert first == second
    assert len(first) == 2
    assert all(path.parent.name == "train" for path in first)


def test_sample_yolo_ndjson_rejects_missing_calibration_directory(tmp_path):
    ndjson = tmp_path / "data.ndjson"
    _write_ndjson(ndjson)

    sample_paths = getattr(calibration_module, "sample_yolo_ndjson", None)
    assert callable(sample_paths)
    try:
        sample_paths(ndjson, tmp_path / "missing", split="train", count=1, seed=1)
    except FileNotFoundError as error:
        assert "train" in str(error)
    else:
        raise AssertionError("missing calibration directory must be rejected")


def test_sample_yolo_ndjson_rejects_missing_referenced_image(tmp_path):
    ndjson = tmp_path / "data.ndjson"
    images_root = tmp_path / "images"
    (images_root / "train").mkdir(parents=True)
    _write_ndjson(ndjson)
    (images_root / "train/a.jpg").write_bytes(b"image")

    sample_paths = getattr(calibration_module, "sample_yolo_ndjson", None)
    assert callable(sample_paths)
    try:
        sample_paths(ndjson, images_root, split="train", count=3, seed=1)
    except FileNotFoundError as error:
        assert "b.jpg" in str(error) or "c.jpg" in str(error)
    else:
        raise AssertionError("missing NDJSON image must be rejected")
