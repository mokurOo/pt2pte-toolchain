from pathlib import Path

import torch

import pt2pte.pipeline as pipeline_module


def test_compile_spec_arguments_preserve_configured_u85_system_profile():
    build_arguments = getattr(pipeline_module, "build_compile_spec_arguments", None)
    assert callable(build_arguments)
    arguments = build_arguments(
        accelerator="ethos-u85-256",
        system_config="Ethos_U85_SYS_DRAM_Mid",
        memory_mode="Sram_Only",
        vela_ini=None,
        extra_flags=("--arena-cache-size=4194304",),
    )

    assert arguments == {
        "target": "ethos-u85-256",
        "system_config": "Ethos_U85_SYS_DRAM_Mid",
        "memory_mode": "Sram_Only",
        "extra_flags": ["--arena-cache-size=4194304"],
    }


def test_output_paths_create_only_pte_pipeline_artifacts(tmp_path):
    output_paths = getattr(pipeline_module, "output_paths", None)
    assert callable(output_paths)
    paths = output_paths(tmp_path, "hand_pose", "ethos-u85-256")

    assert paths.pte == tmp_path / "hand_pose_ethos_u85_256.pte"
    assert paths.report == tmp_path / "export_report.json"
    assert paths.calibration_manifest == tmp_path / "calibration_manifest.json"
    assert paths.delegation_info == tmp_path / "delegation_info.txt"
    assert not hasattr(paths, "fvp")
    assert not hasattr(paths, "runner")


def test_compare_output_tensors_reports_each_named_output():
    compare = getattr(pipeline_module, "compare_output_tensors", None)
    assert callable(compare)
    expected = (
        torch.tensor([[[1.0, 2.0]]]),
        torch.tensor([[[0.5, 0.75]]]),
    )
    actual = (
        torch.tensor([[[1.25, 2.0]]]),
        torch.tensor([[[0.5, 0.5]]]),
    )

    report = compare(expected, actual, names=("boxes", "confidence"))

    assert report == {
        "names": ["boxes", "confidence"],
        "expected_shapes": [[1, 1, 2], [1, 1, 2]],
        "actual_shapes": [[1, 1, 2], [1, 1, 2]],
        "per_output_max_abs_error": {
            "boxes": 0.25,
            "confidence": 0.25,
        },
        "max_abs_error": 0.25,
    }
