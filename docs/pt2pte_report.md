# PT2PTE 转换报告

## 1. 核心流程

```text
配置解析
  → adapter 恢复 PyTorch Module
  → torch.export
  → PT2E INT8 校准量化
  → ExecuTorch Arm/U85 lowering
  → Vela 优化
  → Cortex-M pass
  → 输出 .pte 与诊断报告
```

YOLO pose adapter 会在 Ultralytics Pose head 的最终拼接之前输出独立字段，
避免框坐标范围和 score 范围共享同一个量化 scale。MobileNetV2 则保持单一
分类 logits 输出；两者共用后续量化、lowering 和 PTE 生成流程。

运行入口：

```bash
./scripts/export_pte.sh configs/yolo_hand_pose.yaml \
  --weights /absolute/path/to/best.pt \
  --target ethos-u85-256
```

YOLO 快速复现也支持同样的覆盖项：

```bash
./scripts/reproduce_yolo_hand.sh \
  --weights /absolute/path/to/best.pt \
  --target ethos-u85-256
```

两项覆盖分别对应 `model.weights` 和 `target.accelerator`；输入尺寸仍由 YAML
或 checkpoint 元数据决定。

## 2. 当前转换输入

| 指标 | 当前值 |
|---|---|
| 模型 | `yolo_hand/.../best.pt` |
| adapter | `ultralytics_yolo` |
| 输入尺寸 | checkpoint 解析为 `224x224` |
| 最大 stride | `32` |
| 校准来源 | NDJSON `train` split + 图片目录 |
| 校准抽样 | `200` 张，seed `42` |
| 目标 | `ethos-u85-256` |
| system config | `Ethos_U85_SYS_DRAM_Mid` |
| memory mode | `Shared_Sram` |

`input_size: null` 不代表固定 224，而是读取 checkpoint 元数据；显式配置时
尺寸必须按 stride 对齐。YOLO 的候选数 `N` 随输入尺寸变化。

## 3. PTE 输出

当前 `224x224` 输入得到 `N=1029`，`forward` 返回四个输出：

| 输出 | Shape | 语义 |
|---|---:|---|
| `outputs[0]` | `[1, 4, 1029]` | 解码后的 box |
| `outputs[1]` | `[1, 1, 1029]` | hand confidence |
| `outputs[2]` | `[1, 42, 1029]` | 21 个关键点的 x/y |
| `outputs[3]` | `[1, 21, 1029]` | 21 个关键点 score |

`confidence` 和 `keypoint_scores` 已经 sigmoid；部署侧不应重复 sigmoid，
也不能再按旧的单一 `[1,68,1029]` tensor 读取。

普通 PTE：
`artifacts/yolo_hand_pose/yolo_hand_pose_ethos_u85_256.pte`，大小 `3,353,984` bytes。

## 4. 验证指标及解析

### 结构与数据验证

- 校准图片根目录存在，共 `26,677` 个非空图片文件。
- 导出前后四路 output shape 完全一致。
- 普通导出：`845` 个 delegated nodes、`9` 个 non-delegated nodes、`1` 个
  delegated subgraph。
- 已生成 `delegation_info.txt`、`lowered_graph.txt`、`lowering.log` 和
  `export_report.json`。

### 主机侧量化对比

当前报告中的最大绝对误差为：

| 输出 | max abs error |
|---|---:|
| boxes | `95.5675` |
| confidence | `0.0003348` |
| keypoint_xy | `78.9499` |
| keypoint_scores | `0.5508` |

这些误差是在固定全零 example input 上，对比 eager 输出和校准后的 PT2E
量化输出；它们是数值回归信号，不等同于 mAP、PCK 或最终检测准确率。框和
坐标的绝对误差受像素坐标范围影响，不能单独据此判断检测失败；score 是否
仍有合理范围、输出 shape 是否正确以及任务级指标更重要。

### 量化范围验证

debug lowered graph 中四个输出的 scale 为：

```text
boxes           1.6865493059
confidence      0.0037688587
keypoint_xy     3.0975365639
keypoint_scores 0.0039196536
```

分数分支使用约 `0.0039` 的 scale，而不是复用坐标的约 `1~3` scale，说明
原先 score 被量化为 0 的根因已经解除。任务级精度仍应在部署侧用独立的
验证集评估；本工具链不执行 FVP 或 runner。

完整机器可读结果见：

- [普通导出报告](../artifacts/yolo_hand_pose/export_report.json)
- [debug lowered graph](../output/yolo_intermediates/lowered_graph.txt)
- [debug 诊断说明](debug_pte.md)
