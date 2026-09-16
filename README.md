# PT2PTE Toolchain

独立的配置驱动工具链：恢复 PyTorch checkpoint，执行 PT2E INT8 校准量化，经过 ExecuTorch Arm 后端和 Vela，生成 Ethos-U `.pte`。

当前内置两个 adapter：

- `ultralytics_yolo`：支持 Ultralytics YOLO checkpoint；`input_size: null` 时从 checkpoint 的 `train_args.imgsz` 推导，也可配置整数或 `[height, width]`。
- `torchvision_mobilenet_v2`：从 `state_dict` 或包含 `state_dict` 的 checkpoint 重建 MobileNetV2。

YOLO 的 LiteRT 与 PTE 转换共用显式校准清单：
`../yolo_hand/datasets/hand_pose/calibration/train_seed42_500.txt`。先在
`yolo_hand` 运行 `python calibration_manifest.py` 生成或校验清单，pt2pte 的
`image_list` provider 会按同一排序读取，不再独立随机抽样。

工具链只生成 `.pte` 及导出诊断文件，不构建 runner，也不执行 FVP。

## 快速使用

首先在配置文件中更改模型/目标/训练集路径等信息：

```bash
./scripts/export_pte.sh configs/mobilenet_v2_example.yaml
```

运行环境配置脚本：
```bash
cd /_path_to_pt2pte_toolchain
./scripts/bootstrap.sh
```

运行转换脚本并指定配置文件位置：
```bash
./scripts/export_pte.sh configs/yolo_hand_pose.yaml
```

最终文件位置由配置的 `output.directory` 和 `output.name` 决定。当前 YOLO 默认输出：

```text
artifacts/yolo_hand_pose/yolo_hand_pose_ethos_u85_256.pte
```

YOLO pose adapter 的 `model.pose_output` 支持 `split`（默认）和 `packed`。
`split` 在最终 concat 前输出四个张量：

```text
outputs[0]  boxes          [1, 4, N]
outputs[1]  confidence     [1, 1, N]
outputs[2]  keypoint_xy    [1, 42, N]
outputs[3]  keypoint_scores [1, 21, N]
```

其中 `confidence` 和 `keypoint_scores` 已经包含 sigmoid。这样量化器可以为
坐标和分数分别选择量化范围，避免分数被坐标范围压成 0。部署侧必须按四个
输出读取；旧的按单一 `[1, 68, N]` 输出读取的 parser 不适用于新 PTE。

U85-1024 对照配置为：

```text
configs/yolo_hand_pose_1024_split.yaml
configs/yolo_hand_pose_1024_packed.yaml
```

两者使用同一校准清单和 Vela 参数，并写入不同产物目录。
实验结果与输出量化语义见
[docs/yolo_pose_packed_split.md](docs/yolo_pose_packed_split.md)。
TFLite_B 的导出入口为 `cd ../yolo_hand && .venv/bin/python deploy.py
--tflite-pixel-split`，产物使用 `model_*_pixel_split.tflite` 独立命名。

环境和复现细节见 [docs/environment.md](docs/environment.md) 与 [docs/reproduction.md](docs/reproduction.md)。
