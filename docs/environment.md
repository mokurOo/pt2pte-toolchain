# 环境说明

## 工具链环境

工具链要求 Python 3.12，当前验证组合为：

- ExecuTorch v1.4.1
- Ethos-U Vela 5.1.0
- TOSA tools 2026.5.0
- Ultralytics 8.4.129
- PyTorch、torchvision、torchao、NumPy、Pillow、PyYAML

`pyproject.toml` 声明了模型 adapter 所需的 Python 依赖；ExecuTorch、
torchao、Vela、TOSA 和 CMSIS-NN Python binding 由 `scripts/bootstrap.sh`
准备。bootstrap 创建工具链自己的 `.venv` 和 `third_party/executorch`，
不会修改 `yolo_hand/.venv`。

## 安装与检查

```bash
cd /home/mokuroo/documents/python/pt2pte_toolchain
./scripts/bootstrap.sh --device cpu

.venv/bin/python -c 'import torch, torchao, torchvision, ultralytics; print(torch.__version__)'
PYTHONPATH=build/cmsis/backends/cortex_m/cmsis_nn-build \
  .venv/bin/python -c 'import cmsis_nn; print(cmsis_nn.__file__)'
.venv/bin/vela --version
```

bootstrap 默认就是 CPU 模式，并安装 `torch==2.13.0+cpu`。也可以使用环境变量：

```bash
PT2PTE_DEVICE=cpu ./scripts/bootstrap.sh
```

GPU 模式必须显式选择，且要求 CUDA toolkit 的 `nvcc` 已在 `PATH` 中：

```bash
./scripts/bootstrap.sh --device gpu
# 或：PT2PTE_DEVICE=gpu ./scripts/bootstrap.sh
```

CPU torch 版本和下载源可通过 `PT2PTE_TORCH_VERSION`、
`PT2PTE_TORCH_INDEX_URL` 覆盖；默认值分别为 `2.13.0` 和
`https://download.pytorch.org/whl/test/cpu`。CPU 模式会阻断 bootstrap 阶段的
`nvcc` 自动探测，并显式关闭 ExecuTorch CUDA 构建，因此不会因为机器上存在 CUDA
而切换到 CUDA torch 源或构建 CUDA 后端。GPU 模式只在显式选择后保留 CUDA 探测；
本工具链的目标仍是生成 Ethos-U PTE，不要求部署侧带 CUDA。

只需要主机侧 AOT 工具和 C/C++ 编译器；不需要 Arm bare-metal 编译器，
也不安装或运行 FVP。工具链边界是生成 PTE 和导出诊断文件。

## 使用已有环境

如果 ExecuTorch 和 CMSIS-NN 已在其他环境中准备好，可以跳过 bootstrap：

```bash
export PT2PTE_PYTHON=/absolute/path/to/python
export EXECUTORCH_ROOT=/absolute/path/to/executorch
export PT2PTE_CMSIS_NN_PATH=/absolute/path/to/cmsis_nn-build
./scripts/export_pte.sh configs/yolo_hand_pose.yaml
```

`PT2PTE_PYTHON` 必须能导入 `torch`、`torchao`、`torchvision` 和 Vela；
`EXECUTORCH_ROOT` 应包含 `src/executorch` 或顶层 `executorch`；
`PT2PTE_CMSIS_NN_PATH` 应包含 `cmsis_nn*.so`。

如果模型 adapter 的依赖安装在另一个 Python 环境中，需要把它的
`site-packages` 加到 `PYTHONPATH`。例如当前复用 `yolo_hand` 的 Ultralytics：

```bash
export PYTHONPATH=/home/mokuroo/documents/python/yolo_hand/.venv/lib/python3.12/site-packages
./scripts/export_pte.sh configs/yolo_hand_pose.yaml
```

脚本会自动把工具链源码、CMSIS-NN、ExecuTorch 和 `PT2PTE_PYTHON` 的
site-packages 加入 Python 搜索路径，并把上述外部 `PYTHONPATH` 放在它们之后。
外部路径只应提供 Ultralytics 等 adapter 依赖，不要把另一个环境的 torch 或
torchao 放在前面。脚本会打印并检查这两个包的实际来源；如果发现跨环境混用会在
导出前停止，从而避免 `call_module _guards_fn` 这类误导性错误。固定工具链如果确实
要使用其他 torch 主版本，可显式设置 `PT2PTE_EXPECTED_TORCH_PREFIX=`，但需要自行
确认该版本与 ExecuTorch v1.4.1 兼容。

## 依赖边界

- `ultralytics_yolo` 需要可恢复的 Ultralytics `.pt` 和 Ultralytics 包。
- `torchvision_mobilenet_v2` 需要可恢复的 MobileNetV2 `state_dict` checkpoint。
- 量化校准不需要额外的数据格式转换工具；YOLO 使用 NDJSON 加图片目录，
  MobileNetV2 使用图片目录。
- 不生成 SHA256 摘要，不生成 FVP 脚本，也不构建部署 runner。
