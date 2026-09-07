"""Model adapter utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class LoadedAdapter:
    name: str
    model: torch.nn.Module
    input_size: tuple[int, int]
    metadata: dict[str, Any]

    def preprocess(self, path: str | Path) -> torch.Tensor:
        if self.name == "ultralytics_yolo":
            return preprocess_yolo_image(path, self.input_size)
        if self.name == "torchvision_mobilenet_v2":
            return preprocess_mobilenet_image(path, self.input_size)
        raise ValueError(f"unsupported adapter: {self.name}")


class _TensorOutputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.model(value)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"model output must be a Tensor, got {type(output)!r}")
        return output


def _unwrap_tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"model output must be a Tensor, got {type(output)!r}")
    return output


def split_yolo_pose_output(
    output: torch.Tensor,
    *,
    keypoint_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed Ultralytics pose tensor for inspection or compatibility.

    Ultralytics exports pose predictions as ``[N, 5 + 3*K, candidates]``:
    ``[cx, cy, w, h, confidence, x1, y1, score1, ...]``.  Coordinates and
    scores have very different ranges.  The actual adapter does not use this
    post-hoc split for quantization, because Arm's shared-qspec pass propagates
    one scale through ``slice``/``split`` nodes.  It is retained as a small
    layout-validation utility.
    """
    if output.dim() != 3:
        raise ValueError(f"YOLO pose output must be rank 3, got rank {output.dim()}")
    if keypoint_count <= 0:
        raise ValueError("keypoint_count must be greater than zero")
    expected_attributes = 5 + 3 * keypoint_count
    if output.shape[1] != expected_attributes:
        raise ValueError(
            "YOLO pose output attribute count does not match keypoint_count: "
            f"expected {expected_attributes}, got {output.shape[1]}"
        )

    keypoint_start = 5
    keypoint_xy_end = keypoint_start + 2 * keypoint_count
    return (
        output[:, 0:4, :],
        output[:, 4:5, :],
        output[:, keypoint_start:keypoint_xy_end, :],
        output[:, keypoint_xy_end:expected_attributes, :],
    )


class _YoloPoseFieldHead(torch.nn.Module):
    """Expose the Ultralytics pose head fields before its final concatenation.

    Slicing the already-concatenated ``[boxes, score, keypoints]`` tensor is
    too late for the Arm quantizer: its shared-qspec pass deliberately shares
    quantization parameters across ``slice``/``split``/``getitem`` nodes.  The
    head therefore returns the independently-produced fields directly, while
    retaining Ultralytics' normal decode math.
    """

    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.head = head
        self.f = head.f
        self.i = head.i

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        predictions = self.head.forward_head(features, **self.head.one2many)
        boxes = self.head._get_decode_boxes(predictions)
        confidence = predictions["scores"].sigmoid()
        raw_keypoints = predictions["kpts"]
        batch_size = raw_keypoints.shape[0]
        keypoint_count, keypoint_dims = self.head.kpt_shape
        keypoints = raw_keypoints.view(batch_size, keypoint_count, keypoint_dims, -1)
        keypoint_xy = (
            keypoints[:, :, :2] * 2.0
            + (self.head.anchors - 0.5)
        ) * self.head.strides
        keypoint_xy = keypoint_xy.reshape(batch_size, keypoint_count * 2, -1)
        if keypoint_dims == 3:
            keypoint_scores = keypoints[:, :, 2:3].sigmoid().reshape(
                batch_size, keypoint_count, -1
            )
            return boxes, confidence, keypoint_xy, keypoint_scores
        return boxes, confidence, keypoint_xy


def resolve_yolo_input_size(
    configured: int | Sequence[int] | None,
    checkpoint_args: Mapping[str, Any],
    *,
    stride: int,
) -> tuple[int, int]:
    value = configured if configured is not None else checkpoint_args.get("imgsz")
    if value is None:
        raise ValueError(
            "model.input_size is required when checkpoint metadata has no imgsz"
        )
    if isinstance(value, int):
        height = width = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise ValueError("YOLO input_size must be an integer or [height, width]")
    if height <= 0 or width <= 0:
        raise ValueError("YOLO input_size dimensions must be positive")
    if height % stride or width % stride:
        raise ValueError(
            f"YOLO input_size {(height, width)} must be aligned to stride {stride}"
        )
    return height, width


def _normalize_size(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        result = int(value[0]), int(value[1])
    else:
        raise ValueError("input_size must be an integer or [height, width]")
    if min(result) <= 0:
        raise ValueError("input_size dimensions must be positive")
    return result


def _letterbox_rgb(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    target_height, target_width = size
    ratio = min(target_width / image.width, target_height / image.height)
    resized_width = max(1, round(image.width * ratio))
    resized_height = max(1, round(image.height * ratio))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), color=(114, 114, 114))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return np.asarray(canvas, dtype=np.float32)


def preprocess_yolo_image(path: str | Path, size: tuple[int, int]) -> torch.Tensor:
    array = _letterbox_rgb(path, size) / 255.0
    value = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return value.to(memory_format=torch.channels_last)


def preprocess_mobilenet_image(path: str | Path, size: tuple[int, int]) -> torch.Tensor:
    array = _letterbox_rgb(path, size) / 255.0
    value = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return ((value - mean) / std).to(memory_format=torch.channels_last)


def load_yolo_adapter(
    weights: str | Path,
    *,
    input_size: int | Sequence[int] | None,
) -> LoadedAdapter:
    from ultralytics import YOLO

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint does not exist: {weights_path}")
    model = YOLO(str(weights_path)).model.float().eval()
    head = model.model[-1]
    stride = int(model.stride.max().item())
    resolved_size = resolve_yolo_input_size(input_size, model.args, stride=stride)
    head.export = True
    head.dynamic = False
    head.format = "torchscript"
    task = str(model.args.get("task", "unknown"))
    keypoint_shape = list(getattr(head, "kpt_shape", []))
    keypoint_count = int(keypoint_shape[0]) if len(keypoint_shape) == 2 else 0
    keypoint_dims = int(keypoint_shape[1]) if len(keypoint_shape) == 2 else 0
    pose_output_names = ["boxes", "confidence", "keypoint_xy"]
    if keypoint_dims == 3:
        pose_output_names.append("keypoint_scores")
    example = torch.zeros((1, 3, *resolved_size), dtype=torch.float32).to(
        memory_format=torch.channels_last
    )
    with torch.inference_mode():
        source_output = _unwrap_tensor_output(model(example))
    if task == "pose" and keypoint_count > 0:
        model.model[-1] = _YoloPoseFieldHead(head).eval()
        wrapped = model.eval().to(memory_format=torch.channels_last)
    else:
        wrapped = _TensorOutputWrapper(model).eval().to(
            memory_format=torch.channels_last
        )
    with torch.inference_mode():
        output = wrapped(example)
    output_tensors = (
        list(output) if isinstance(output, (tuple, list)) else [output]
    )
    if not all(isinstance(value, torch.Tensor) for value in output_tensors):
        raise TypeError("YOLO adapter outputs must all be tensors")
    return LoadedAdapter(
        name="ultralytics_yolo",
        model=wrapped,
        input_size=resolved_size,
        metadata={
            "task": task,
            "stride": stride,
            "num_classes": int(getattr(head, "nc", 0)),
            "keypoint_shape": keypoint_shape,
            "source_output_shape": list(source_output.shape),
            "output_names": (
                pose_output_names
                if task == "pose" and keypoint_count > 0
                else ["output"]
            ),
            "output_shapes": [list(value.shape) for value in output_tensors],
        },
    )


def load_mobilenet_v2_adapter(
    weights: str | Path,
    *,
    input_size: int | Sequence[int] | None,
    num_classes: int | None,
) -> LoadedAdapter:
    from torchvision.models import mobilenet_v2

    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"MobileNetV2 checkpoint does not exist: {weights_path}")
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("MobileNetV2 checkpoint must be a state_dict or checkpoint mapping")
    state_dict = payload.get("state_dict", payload)
    model_config = payload.get("model_config", {})
    classes = num_classes if num_classes is not None else model_config.get("num_classes")
    if classes is None:
        raise ValueError("model.num_classes is required for MobileNetV2")
    size_value = input_size if input_size is not None else model_config.get("input_size", 224)
    resolved_size = _normalize_size(size_value)
    model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, int(classes))
    model.load_state_dict(state_dict)
    model.eval().to(memory_format=torch.channels_last)
    return LoadedAdapter(
        name="torchvision_mobilenet_v2",
        model=model,
        input_size=resolved_size,
        metadata={
            "task": "classification",
            "num_classes": int(classes),
            "output_names": ["logits"],
        },
    )


def load_adapter(
    adapter: str,
    weights: str | Path,
    *,
    input_size: int | Sequence[int] | None,
    num_classes: int | None = None,
) -> LoadedAdapter:
    if adapter == "ultralytics_yolo":
        return load_yolo_adapter(weights, input_size=input_size)
    if adapter == "torchvision_mobilenet_v2":
        return load_mobilenet_v2_adapter(
            weights, input_size=input_size, num_classes=num_classes
        )
    raise ValueError(f"unsupported model adapter: {adapter}")
