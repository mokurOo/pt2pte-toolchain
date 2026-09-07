"""Typed configuration loading for the PT-to-PTE pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    adapter: str
    weights: Path
    input_size: int | tuple[int, int] | None
    architecture: str | None = None
    num_classes: int | None = None


@dataclass(frozen=True)
class CalibrationConfig:
    provider: str
    ndjson: Path | None
    images_root: Path
    split: str
    count: int
    seed: int


@dataclass(frozen=True)
class TargetConfig:
    accelerator: str
    system_config: str
    memory_mode: str
    vela_ini: Path | None
    vela_extra_flags: tuple[str, ...]


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    name: str
    save_intermediates: bool


@dataclass(frozen=True)
class DiagnosticsConfig:
    debug: bool


@dataclass(frozen=True)
class ToolkitConfig:
    model: ModelConfig
    calibration: CalibrationConfig
    target: TargetConfig
    output: OutputConfig
    diagnostics: DiagnosticsConfig


def _path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _input_size(value: Any) -> int | tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("model.input_size must be null, an integer, or [height, width]")


def load_config(path: str | Path) -> ToolkitConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    missing = [name for name in ("model", "calibration", "target", "output") if name not in raw]
    if missing:
        raise ValueError(f"configuration is missing sections: {', '.join(missing)}")

    base = config_path.parent
    model = raw["model"]
    calibration = raw["calibration"]
    target = raw["target"]
    output = raw["output"]
    diagnostics = raw.get("diagnostics", {})
    count = int(calibration.get("count", 200))
    if count <= 0:
        raise ValueError("calibration.count must be greater than zero")

    return ToolkitConfig(
        model=ModelConfig(
            adapter=str(model["adapter"]),
            weights=_path(base, model["weights"]),  # type: ignore[arg-type]
            input_size=_input_size(model.get("input_size")),
            architecture=model.get("architecture"),
            num_classes=(int(model["num_classes"]) if model.get("num_classes") is not None else None),
        ),
        calibration=CalibrationConfig(
            provider=str(calibration["provider"]),
            ndjson=_path(base, calibration.get("ndjson")),
            images_root=_path(base, calibration["images_root"]),  # type: ignore[arg-type]
            split=str(calibration.get("split", "train")),
            count=count,
            seed=int(calibration.get("seed", 42)),
        ),
        target=TargetConfig(
            accelerator=str(target["accelerator"]),
            system_config=str(target["system_config"]),
            memory_mode=str(target.get("memory_mode", "Sram_Only")),
            vela_ini=_path(base, target.get("vela_ini")),
            vela_extra_flags=tuple(str(flag) for flag in target.get("vela_extra_flags", [])),
        ),
        output=OutputConfig(
            directory=_path(base, output["directory"]),  # type: ignore[arg-type]
            name=str(output["name"]),
            save_intermediates=bool(output.get("save_intermediates", True)),
        ),
        diagnostics=DiagnosticsConfig(debug=bool(diagnostics.get("debug", False))),
    )
