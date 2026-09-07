# 环境说明

## 固定版本

- Python 3.12
- ExecuTorch v1.4.1
- Ethos-U Vela 5.1.0
- TOSA tools 2026.5.0
- Ultralytics 8.4.129

运行 `scripts/bootstrap.sh` 会创建工具链自己的 `.venv` 和 `third_party/executorch`。它不会修改 `yolo_hand/.venv`。

只需要主机侧 AOT 工具：PyTorch、torchao、ExecuTorch Python 源码、TOSA、Vela，以及用于委托边界转换的 CMSIS-NN Python binding。bootstrap 会在 `build/cmsis` 构建该 binding；需要主机 C/C++ 编译器，但不需要 Arm bare-metal 编译器，也不安装 FVP。

如已有兼容的 ExecuTorch 源码，可跳过 bootstrap，并在执行时指定：

```bash
export EXECUTORCH_ROOT=/absolute/path/to/executorch
export PT2PTE_CMSIS_NN_PATH=/absolute/path/to/cmsis_nn-build
./scripts/export_pte.sh configs/yolo_hand_pose.yaml
```

环境至少应通过：

```bash
.venv/bin/python -c 'import torch, torchao, ultralytics'
PYTHONPATH=build/cmsis/backends/cortex_m/cmsis_nn-build .venv/bin/python -c 'import cmsis_nn'
.venv/bin/vela --version
```
