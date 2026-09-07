from pathlib import Path

import yaml

import pt2pte.config as config_module


def test_load_config_resolves_paths_relative_to_config_and_keeps_yolo_size_optional(tmp_path):
    config_path = tmp_path / "configs" / "yolo.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "adapter": "ultralytics_yolo",
                    "weights": "../models/best.pt",
                    "input_size": None,
                },
                "calibration": {
                    "provider": "yolo_ndjson",
                    "ndjson": "../dataset/data.ndjson",
                    "images_root": "../dataset/images",
                    "split": "train",
                    "count": 200,
                    "seed": 42,
                },
                "target": {
                    "accelerator": "ethos-u85-256",
                    "system_config": "Ethos_U85_SYS_DRAM_Mid",
                    "memory_mode": "Sram_Only",
                },
                "output": {"directory": "../artifacts", "name": "hand_pose"},
                "diagnostics": {"debug": True},
            }
        ),
        encoding="utf-8",
    )

    load_config = getattr(config_module, "load_config", None)
    assert callable(load_config)
    loaded = load_config(config_path)

    assert loaded.model.input_size is None
    assert loaded.model.weights == (tmp_path / "models/best.pt").resolve()
    assert loaded.calibration.ndjson == (tmp_path / "dataset/data.ndjson").resolve()
    assert loaded.target.system_config == "Ethos_U85_SYS_DRAM_Mid"
    assert loaded.diagnostics.debug is True


def test_load_config_rejects_non_positive_calibration_count(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
model:
  adapter: ultralytics_yolo
  weights: best.pt
calibration:
  provider: yolo_ndjson
  ndjson: data.ndjson
  images_root: images
  split: train
  count: 0
target:
  accelerator: ethos-u85-256
  system_config: Ethos_U85_SYS_DRAM_Mid
  memory_mode: Sram_Only
output:
  directory: artifacts
  name: model
""",
        encoding="utf-8",
    )

    load_config = getattr(config_module, "load_config", None)
    assert callable(load_config)
    try:
        load_config(config_path)
    except ValueError as error:
        assert "count" in str(error)
    else:
        raise AssertionError("zero calibration count must be rejected")
