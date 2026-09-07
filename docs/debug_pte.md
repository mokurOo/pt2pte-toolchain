# YOLO pose PTE 部署诊断

## 历史问题

旧版 adapter 将 Ultralytics Pose head 输出拼成单一的 `[1, 68, N]` 张量。
框坐标范围约为 0 到 250，而 confidence 和关键点 score 范围为 0 到 1；
Arm 量化器在最终 `slice`/`split`/`getitem` 路径上共享 qspec，导致 score
被坐标范围的量化 scale 压成 0。

仅在最终张量后切片不能解决该问题。

## 当前输出契约

当前 adapter 在 Pose head 最终拼接前返回四个独立输出：

```text
outputs[0]  boxes          [1, 4, N]
outputs[1]  confidence     [1, 1, N]
outputs[2]  keypoint_xy    [1, 42, N]
outputs[3]  keypoint_scores [1, 21, N]
```

`confidence` 和 `keypoint_scores` 已经完成 sigmoid。部署侧不能继续按单一
`[1,68,N]` tensor 读取，也不应再次 sigmoid；hand confidence 应读取
`outputs[1]`。

当前 checkpoint 使用 `224x224` 输入时 `N=1029`，但 `N` 会随输入尺寸变化，
不能在通用部署代码中固定。

## 当前验证

debug lowered graph 的四路 output scale 已独立，约为：

```text
boxes           1.6865
confidence      0.003768
keypoint_xy     3.0975
keypoint_scores 0.003919
```

这说明 score 分支不再复用框坐标的量化范围。更完整的流程、指标和解析见
[pt2pte_report.md](pt2pte_report.md)。本工具链只生成 PTE 和诊断文件，不修改
FVP runner。
