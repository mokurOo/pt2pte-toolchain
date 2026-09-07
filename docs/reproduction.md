# 从 best.pt 复现 YOLO Hand Pose PTE

## 输入

- checkpoint：`../yolo_hand/results/runs/yolo_hand_pose_224/checkpoints/best.pt`
- NDJSON：`../yolo_hand/hand-keypoints.ndjson`
- 图片根目录：`../yolo_hand/datasets/hand_pose/images`
- 校准 split：`train`
- 校准数量：200，seed 42

当前数据核验结果：NDJSON 有 18,724 条 train 和 7,953 条 val 图像记录；
`images/train` 存在，NDJSON 引用的 26,677 张图片均存在且非空。运行时会再次
检查 split 目录、记录数量以及抽样前所有引用图片是否存在且非空。

## 执行

```bash
cd /home/mokuroo/documents/python/pt2pte_toolchain
./scripts/bootstrap.sh --device cpu
./scripts/reproduce_yolo_hand.sh
```

复现脚本支持直接覆盖 checkpoint 和目标型号，不需要修改 YAML：

```bash
./scripts/reproduce_yolo_hand.sh \
  --weights /absolute/path/to/best.pt \
  --target ethos-u85-256
```

也可以指定另一份配置：

```bash
./scripts/reproduce_yolo_hand.sh \
  --config configs/yolo_hand_pose_debug.yaml \
  --weights /absolute/path/to/best.pt \
  --target ethos-u85-512
```

底层通用入口 `export_pte.sh` 的用法相同，配置文件放在第一个参数，后面传入
覆盖项：

```bash
./scripts/export_pte.sh configs/yolo_hand_pose.yaml \
  --weights /absolute/path/to/best.pt \
  --target ethos-u85-256
```

`--weights` 覆盖 `model.weights`，`--target` 覆盖 `target.accelerator`，其余
校准、输出和 Vela 配置继续从 YAML 读取。`model.input_size: null` 仍然表示从
YOLO checkpoint 元数据读取尺寸，不会由复现脚本固定 `imgsz`。

也可以在已有兼容环境中运行：

```bash
export EXECUTORCH_ROOT=/home/mokuroo/documents/python/cat_dog_torch/third_party/executorch
export PT2PTE_CMSIS_NN_PATH=/home/mokuroo/documents/python/cat_dog_torch/cmake-out-cmsis/backends/cortex_m/cmsis_nn-build
export PT2PTE_PYTHON=/home/mokuroo/documents/python/cat_dog_torch/.venv-executorch/bin/python
export PYTHONPATH=/home/mokuroo/documents/python/yolo_hand/.venv/lib/python3.12/site-packages
./scripts/export_pte.sh configs/yolo_hand_pose.yaml
```

## 配置语义

`model.input_size: null` 会从 YOLO checkpoint 元数据读取尺寸，不在 adapter 中
固定 224。本 checkpoint 的元数据解析结果是 `224x224`。显式传入尺寸时支持：

```yaml
input_size: 320
```

或：

```yaml
input_size: [256, 320]
```

尺寸必须为模型最大 stride 的整数倍；尺寸改变时，候选数 `N` 和四路输出的
最后一维也会随特征图尺寸改变，不应在部署侧硬编码 `1029`。

当前目标参数：

```yaml
accelerator: ethos-u85-256
system_config: Ethos_U85_SYS_DRAM_Mid
memory_mode: Shared_Sram
vela_extra_flags:
  - --optimise=Performance
  - --arena-cache-size=4194304
```

## 输出契约

Pose adapter 会在 Ultralytics head 的最终拼接之前拆出独立分支，PTE 的
`forward` 返回四个输出：

```text
outputs[0]  boxes          [1, 4, N]
outputs[1]  confidence     [1, 1, N]
outputs[2]  keypoint_xy    [1, 42, N]
outputs[3]  keypoint_scores [1, 21, N]
```

当前 checkpoint 在 `224x224` 输入下 `N=1029`。坐标和分数的量化参数由
各自分支独立统计；`confidence` 与 `keypoint_scores` 已经是 sigmoid 后的
值。部署侧不能继续把第 4 个属性当作单一 `[1,68,N]` 输出中的 confidence，
而应读取 `outputs[1]`。

## Debug 导出

需要查看 delegation 和 Vela 中间结果时运行：

```bash
./scripts/export_pte.sh configs/yolo_hand_pose_debug.yaml
```

该配置只增加导出诊断参数，不执行 FVP smoketest；产物写入
`output/yolo_intermediates/`。普通复现的交付产物写入 `artifacts/yolo_hand_pose/`。

导出开始前会打印 `torch` 和 `torchao` 的版本及文件路径。如果看到
`call_module _guards_fn`，优先检查 `PT2PTE_PYTHON` 与 `PYTHONPATH`：不能让
`yolo_hand/.venv` 或其他环境中的 torch/torchao 覆盖工具链环境；外部
`PYTHONPATH` 只用于补充 Ultralytics。重新使用 bootstrap 创建的 CPU 环境，或按
上面的“已有兼容环境”示例设置三个路径变量。

## 产物

```text
artifacts/yolo_hand_pose/
├── yolo_hand_pose_ethos_u85_256.pte
├── export_report.json
├── calibration_manifest.json
├── delegation_info.txt
├── lowered_graph.txt
├── lowering.log
└── intermediate/
```

报告记录路径、文件大小、输入输出形状、量化误差和 delegate 数量，不计算文件摘要。
指标解释见 [docs/pt2pte_report.md](pt2pte_report.md)。
