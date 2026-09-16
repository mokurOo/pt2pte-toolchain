# YOLO Hand Pose：PTE packed 与 split 对照

## 实验控制

- checkpoint：`yolo_hand_pose_224/checkpoints/best.pt`
- 输入：`1x3x224x224`
- 校准清单：`train_seed42_500.txt`，500 张，保持文件顺序
- 目标：`ethos-u85-1024`
- Vela：`Ethos_U85_SYS_DRAM_Mid`、`Shared_Sram`、Performance、4 MiB arena cache
- 唯一模型变量：最终输出为 packed `[1,68,1029]`，或 split 四路输出

## 结果

| 指标 | packed | split | split 相对 packed |
|---|---:|---:|---:|
| PTE 大小（bytes） | 3,358,256 | 3,359,440 | +0.04% |
| Vela passes | 133 | 131 | -2 |
| SRAM peak（KiB） | 687.59 | 597.84 | -13.05% |
| DRAM peak（KiB） | 3,251.17 | 3,251.17 | 0 |
| NPU cycles | 1,266,253 | 1,246,596 | -1.55% |
| 总 cycles | 3,037,551 | 3,015,407 | -0.73% |
| Vela 估算时间（ms） | 3.038 | 3.015 | -0.73% |
| delegated / non-delegated nodes | 854 / 3 | 845 / 9 | — |

两者都是单一 delegated subgraph。split 减少约 89.75 KiB SRAM，周期只减少
约 0.73%；因此拆输出能解释一部分内存收益，但不能单独解释此前 PTE 与 TFLite
之间更大的性能差距。

## 数值语义

packed 把像素坐标和 0 到 1 的分数放在同一个输出量化范围内。当前 packed 输出
scale 为约 `1.9718`，远大于置信度的有效变化范围，分数会严重离散化。split
分别量化坐标和分数：confidence scale 约 `0.00377`，keypoint score scale 约
`0.00392`，更适合后处理阈值和关键点可见性判断。

结论：当前部署应使用 split。packed 仅保留为图结构、内存和周期的因果对照，
不应因输出接口更简单而作为 INT8 精度方案。

这里的 Vela 时间是静态估算，不是 FVP 或实板端到端延迟；导出阶段的最大绝对
误差也只是零输入 smoke test，不替代完整验证集精度评估。

## TFLite_B 对标结果

额外导出的 TFLite_B 使用同一 pixel + split 契约。7,953 张验证集结果为：

| 模型 | box mAP50 | box mAP50-95 | pose mAP50 | pose mAP50-95 |
|---|---:|---:|---:|---:|
| TFLite_B FP32 | 0.99269 | 0.88841 | 0.90518 | 0.75559 |
| TFLite_B INT8 | 0.98079 | 0.81735 | 0.83819 | 0.57585 |

TFLite_B 的 U85-1024 Vela 静态结果为 207 passes、1,109.42 KiB SRAM、
3,253.95 KiB DRAM、2,109,710 total cycles。它与 PTE_B 的输出契约已经对齐，
但 lowering 仍不同；该结果不能直接宣称某条路线在目标硬件上更快。
