from pathlib import Path

import torch
from PIL import Image

import pt2pte.adapters as adapters_module


def test_yolo_input_size_comes_from_checkpoint_when_config_omits_it():
    resolve = getattr(adapters_module, "resolve_yolo_input_size", None)
    assert callable(resolve)
    assert resolve(None, {"imgsz": 224}, stride=32) == (224, 224)


def test_yolo_input_size_accepts_configured_rectangular_shape():
    resolve = getattr(adapters_module, "resolve_yolo_input_size", None)
    assert callable(resolve)
    assert resolve([256, 320], {"imgsz": 224}, stride=32) == (256, 320)


def test_yolo_input_size_rejects_shape_not_aligned_to_stride():
    resolve = getattr(adapters_module, "resolve_yolo_input_size", None)
    assert callable(resolve)
    try:
        resolve(230, {"imgsz": 224}, stride=32)
    except ValueError as error:
        assert "stride" in str(error)
    else:
        raise AssertionError("unaligned input size must be rejected")


def test_yolo_input_size_requires_checkpoint_metadata_or_config():
    resolve = getattr(adapters_module, "resolve_yolo_input_size", None)
    assert callable(resolve)
    try:
        resolve(None, {}, stride=32)
    except ValueError as error:
        assert "input_size" in str(error)
    else:
        raise AssertionError("missing input size must be rejected")


def test_preprocess_yolo_image_letterboxes_to_configured_rectangular_shape(tmp_path):
    image_path = tmp_path / "red.png"
    Image.new("RGB", (40, 20), color=(255, 0, 0)).save(image_path)

    preprocess = getattr(adapters_module, "preprocess_yolo_image", None)
    assert callable(preprocess)
    value = preprocess(image_path, (64, 96))

    assert tuple(value.shape) == (1, 3, 64, 96)
    assert value.dtype == torch.float32
    assert float(value.min()) >= 0.0
    assert float(value.max()) <= 1.0
    assert torch.allclose(value[0, :, 32, 48], torch.tensor([1.0, 0.0, 0.0]))


def test_load_yolo_adapter_derives_size_and_splits_pose_output_for_quantization():
    weights = Path(
        "/home/mokuroo/documents/python/yolo_hand/"
        "results/runs/yolo_hand_pose_224/checkpoints/best.pt"
    )
    load_adapter = getattr(adapters_module, "load_yolo_adapter", None)
    assert callable(load_adapter)
    loaded = load_adapter(weights, input_size=None)

    assert loaded.input_size == (224, 224)
    with torch.inference_mode():
        output = loaded.model(torch.zeros(1, 3, 224, 224))
    assert isinstance(output, tuple)
    assert [tuple(value.shape) for value in output] == [
        (1, 4, 1029),
        (1, 1, 1029),
        (1, 42, 1029),
        (1, 21, 1029),
    ]
    assert float(output[1].min()) >= 0.0
    assert float(output[1].max()) <= 1.0
    assert float(output[3].min()) >= 0.0
    assert float(output[3].max()) <= 1.0
    assert loaded.metadata["output_names"] == [
        "boxes",
        "confidence",
        "keypoint_xy",
        "keypoint_scores",
    ]


def test_split_yolo_pose_output_preserves_attribute_order():
    split = getattr(adapters_module, "split_yolo_pose_output", None)
    assert callable(split)
    output = torch.arange(68 * 3, dtype=torch.float32).reshape(1, 68, 3)

    fields = split(output, keypoint_count=21)

    assert len(fields) == 4
    assert torch.equal(fields[0], output[:, 0:4, :])
    assert torch.equal(fields[1], output[:, 4:5, :])
    assert torch.equal(fields[2], output[:, 5:47, :])
    assert torch.equal(fields[3], output[:, 47:68, :])
    assert torch.equal(torch.cat(fields, dim=1), output)


def test_load_mobilenet_v2_adapter_restores_state_dict(tmp_path):
    from torchvision.models import mobilenet_v2

    source = mobilenet_v2(weights=None)
    source.classifier[1] = torch.nn.Linear(source.last_channel, 3)
    checkpoint = tmp_path / "mobilenet.pt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    load_adapter = getattr(adapters_module, "load_mobilenet_v2_adapter", None)
    assert callable(load_adapter)
    loaded = load_adapter(checkpoint, input_size=[160, 192], num_classes=3)

    assert loaded.input_size == (160, 192)
    with torch.inference_mode():
        output = loaded.model(torch.zeros(1, 3, 160, 192))
    assert tuple(output.shape) == (1, 3)
