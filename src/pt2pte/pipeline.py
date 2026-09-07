"""PT2E quantization and Ethos-U/Vela lowering to a PTE file."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .adapters import LoadedAdapter, load_adapter
from .artifacts import write_manifest
from .calibration import sample_image_directory, sample_yolo_ndjson
from .config import ToolkitConfig


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    pte: Path
    report: Path
    calibration_manifest: Path
    delegation_info: Path
    lowered_graph: Path
    lowering_log: Path
    intermediate: Path


def normalize_outputs(output: Any) -> tuple[torch.Tensor, ...]:
    """Normalize a model result to a tuple of tensors."""
    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, (tuple, list)) and all(
        isinstance(value, torch.Tensor) for value in output
    ):
        return tuple(output)
    raise TypeError("model output must be a Tensor or a sequence of Tensors")


def compare_output_tensors(
    expected: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    *,
    names: Sequence[str],
) -> dict[str, Any]:
    """Compare matching eager and quantized outputs, preserving field names."""
    if len(expected) != len(actual) or len(expected) != len(names):
        raise ValueError("expected, actual, and names must have equal lengths")
    expected_shapes = [list(value.shape) for value in expected]
    actual_shapes = [list(value.shape) for value in actual]
    if expected_shapes != actual_shapes:
        raise ValueError(
            f"output shapes changed during quantization: "
            f"{expected_shapes} != {actual_shapes}"
        )
    errors = {
        name: float((expected_value - actual_value).abs().max().item())
        for name, expected_value, actual_value in zip(
            names, expected, actual, strict=True
        )
    }
    return {
        "names": list(names),
        "expected_shapes": expected_shapes,
        "actual_shapes": actual_shapes,
        "per_output_max_abs_error": errors,
        "max_abs_error": max(errors.values(), default=0.0),
    }


def _output_names(metadata: dict[str, Any], count: int) -> tuple[str, ...]:
    configured = metadata.get("output_names")
    if isinstance(configured, (list, tuple)) and len(configured) == count:
        return tuple(str(name) for name in configured)
    return tuple(f"output_{index}" for index in range(count))


def output_paths(directory: str | Path, name: str, accelerator: str) -> OutputPaths:
    root = Path(directory).expanduser().resolve()
    target_name = accelerator.replace("-", "_")
    return OutputPaths(
        root=root,
        pte=root / f"{name}_{target_name}.pte",
        report=root / "export_report.json",
        calibration_manifest=root / "calibration_manifest.json",
        delegation_info=root / "delegation_info.txt",
        lowered_graph=root / "lowered_graph.txt",
        lowering_log=root / "lowering.log",
        intermediate=root / "intermediate",
    )


def build_compile_spec_arguments(
    *,
    accelerator: str,
    system_config: str,
    memory_mode: str,
    vela_ini: Path | None,
    extra_flags: tuple[str, ...],
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "target": accelerator,
        "system_config": system_config,
        "memory_mode": memory_mode,
        "extra_flags": list(extra_flags),
    }
    if vela_ini is not None:
        arguments["config_ini"] = str(vela_ini)
    return arguments


def _calibration_paths(config: ToolkitConfig) -> list[Path]:
    calibration = config.calibration
    if calibration.provider == "yolo_ndjson":
        if calibration.ndjson is None:
            raise ValueError("calibration.ndjson is required for yolo_ndjson")
        return sample_yolo_ndjson(
            calibration.ndjson,
            calibration.images_root,
            split=calibration.split,
            count=calibration.count,
            seed=calibration.seed,
        )
    if calibration.provider == "image_directory":
        return sample_image_directory(
            calibration.images_root,
            count=calibration.count,
            seed=calibration.seed,
        )
    raise ValueError(f"unsupported calibration provider: {calibration.provider}")


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _save_program(program: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        program.write_to_file(handle)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"ExecuTorch program was not written: {path}")


def _write_delegation_info(graph_module: Any, path: Path) -> dict[str, int]:
    from executorch.devtools.backend_debug import get_delegation_info

    delegation = get_delegation_info(graph_module)
    table = delegation.get_operator_delegation_dataframe()
    table_text = table.to_string(index=False) if hasattr(table, "to_string") else str(table)
    path.write_text(
        "Delegation info:\n"
        f"{delegation.get_summary()}\n\n"
        "Delegation table:\n"
        f"{table_text}\n",
        encoding="utf-8",
    )
    return {
        "delegated_nodes": int(delegation.num_delegated_nodes),
        "non_delegated_nodes": int(delegation.num_non_delegated_nodes),
        "delegated_subgraphs": int(delegation.num_delegated_subgraphs),
    }


def _load_model(config: ToolkitConfig) -> LoadedAdapter:
    model = config.model
    return load_adapter(
        model.adapter,
        model.weights,
        input_size=model.input_size,
        num_classes=model.num_classes,
    )


def export_pte(config: ToolkitConfig) -> Path:
    try:
        from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
        from executorch.backends.arm.quantizer import (
            EthosUQuantizer,
            get_symmetric_quantization_config,
        )
        from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
            CortexMPassManager,
        )
        from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
        from executorch.exir import (
            EdgeCompileConfig,
            ExecutorchBackendConfig,
            to_edge_transform_and_lower,
        )
        from torchao.quantization.pt2e import allow_exported_model_train_eval
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    except ImportError as error:
        raise RuntimeError(
            "ExecuTorch Arm dependencies are unavailable; activate the toolkit "
            "environment and set EXECUTORCH_ROOT as described in README.md"
        ) from error

    if config.diagnostics.debug:
        logging.basicConfig(level=logging.DEBUG, force=True)

    loaded = _load_model(config)
    calibration_paths = _calibration_paths(config)
    paths = output_paths(
        config.output.directory, config.output.name, config.target.accelerator
    )
    paths.root.mkdir(parents=True, exist_ok=True)
    _save_json(
        paths.calibration_manifest,
        {
            "provider": config.calibration.provider,
            "split": config.calibration.split,
            "seed": config.calibration.seed,
            "count": len(calibration_paths),
            "images": [str(path) for path in calibration_paths],
        },
    )

    compile_arguments = build_compile_spec_arguments(
        accelerator=config.target.accelerator,
        system_config=config.target.system_config,
        memory_mode=config.target.memory_mode,
        vela_ini=config.target.vela_ini,
        extra_flags=config.target.vela_extra_flags,
    )
    compile_spec = EthosUCompileSpec(**compile_arguments)
    if config.output.save_intermediates:
        paths.intermediate.mkdir(parents=True, exist_ok=True)
        compile_spec.dump_intermediate_artifacts_to(str(paths.intermediate))
    if config.diagnostics.debug:
        compile_spec.dump_debug_info(EthosUCompileSpec.DebugMode.TOSA)

    example = torch.zeros(
        (1, 3, *loaded.input_size), dtype=torch.float32
    ).to(memory_format=torch.channels_last)
    exported = torch.export.export(loaded.model, (example,), strict=True)
    eager_outputs = tuple(
        value.detach().cpu() for value in normalize_outputs(loaded.model(example))
    )

    quantizer = EthosUQuantizer(compile_spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    # Arm's PT2E passes do not accept a call_module node for ExportedProgram's
    # runtime input guards. Disable those guards explicitly instead of relying
    # on PyTorch's call-stack/path heuristic, which makes behavior depend on
    # the virtual-environment directory name.
    prepared = prepare_pt2e(exported.module(check_guards=False), quantizer)
    with torch.inference_mode():
        for image_path in calibration_paths:
            prepared(loaded.preprocess(image_path))
    converted = convert_pt2e(prepared)
    allow_exported_model_train_eval(converted)
    with torch.inference_mode():
        quantized_outputs = tuple(
            value.detach().cpu() for value in normalize_outputs(converted(example))
        )
    names = _output_names(loaded.metadata, len(eager_outputs))
    output_comparison = compare_output_tensors(
        eager_outputs, quantized_outputs, names=names
    )
    quantized_exported = torch.export.export(converted, (example,), strict=True)

    paths.lowering_log.parent.mkdir(parents=True, exist_ok=True)
    with paths.lowering_log.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            edge = to_edge_transform_and_lower(
                programs=quantized_exported,
                partitioner=[EthosUPartitioner(compile_spec)],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )

    delegation = _write_delegation_info(
        edge.exported_program().graph_module, paths.delegation_info
    )

    cpu = CortexM.M85 if "u85" in config.target.accelerator.lower() else CortexM.M55
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program(), target_config=CortexMTargetConfig(cpu=cpu)
    ).transform()
    lowered_graph = edge.exported_program().graph_module.graph
    paths.lowered_graph.write_text(str(lowered_graph), encoding="utf-8")
    program = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    _save_program(program, paths.pte)

    call_targets = [
        str(node.target)
        for node in lowered_graph.nodes
        if node.op == "call_function"
    ]
    metadata = {
        "model": {
            "adapter": loaded.name,
            "weights": str(config.model.weights),
            "input_size": list(loaded.input_size),
            **loaded.metadata,
        },
        "target": {
            "accelerator": config.target.accelerator,
            "system_config": config.target.system_config,
            "memory_mode": config.target.memory_mode,
            "vela_extra_flags": list(config.target.vela_extra_flags),
        },
        "calibration": {
            "manifest": str(paths.calibration_manifest),
            "count": len(calibration_paths),
        },
        "outputs": {
            **output_comparison,
            "eager_shape": output_comparison["expected_shapes"][0]
            if len(eager_outputs) == 1
            else None,
            "quantized_shape": output_comparison["actual_shapes"][0]
            if len(quantized_outputs) == 1
            else None,
        },
        "lowering": {
            "delegate_calls": sum("executorch_call_delegate" in target for target in call_targets),
            "delegation_info": str(paths.delegation_info),
            **delegation,
            "graph": str(paths.lowered_graph),
            "log": str(paths.lowering_log),
            "intermediate": str(paths.intermediate) if config.output.save_intermediates else None,
        },
    }
    write_manifest(paths.report, pte=paths.pte, metadata=metadata)
    return paths.pte
