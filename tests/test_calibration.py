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


def test_load_image_list_resolves_relative_paths_and_preserves_order(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("b.jpg", "a.jpg"):
        (images / name).write_bytes(b"image")
    image_list = tmp_path / "calibration.txt"
    image_list.write_text("./images/b.jpg\n./images/a.jpg\n", encoding="utf-8")

    load_image_list = getattr(calibration_module, "load_image_list", None)
    assert callable(load_image_list)
    loaded = load_image_list(image_list, count=2)

    assert loaded == [(images / "b.jpg").resolve(), (images / "a.jpg").resolve()]


def test_load_image_list_requires_exact_count(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    image_list = tmp_path / "calibration.txt"
    image_list.write_text("./image.jpg\n", encoding="utf-8")

    try:
        calibration_module.load_image_list(image_list, count=2)
    except ValueError as error:
        assert "expected 2" in str(error)
    else:
        raise AssertionError("image-list count mismatch must be rejected")
