已按优先方案修复 `pt2pte_toolchain`。

主要改动：

- 在 Ultralytics Pose head 最终拼接前拆分输出，避免共享量化 scale。
- PTE 输出改为四路：

```text
outputs[0] boxes          [1, 4, 1029]
outputs[1] confidence     [1, 1, 1029]
outputs[2] keypoint_xy    [1, 42, 1029]
outputs[3] keypoint_scores [1, 21, 1029]
```

- `confidence` 和 `keypoint_scores` 已包含 sigmoid。
- 校准目录确认存在，共 26677 张图片，实际使用 200 张。
- 重新生成普通版和 debug 版 U85 PTE。
- debug lowered graph 中四路量化 scale 已独立：

```text
boxes           1.6865
confidence      0.003768
keypoint_xy     3.0975
keypoint_scores 0.003919
```

验证结果：

```text
17 passed
delegate 子图: 1
PTE 已成功生成
```

产物：

- [yolo_hand_pose_ethos_u85_256.pte](/home/mokuroo/documents/python/pt2pte_toolchain/artifacts/yolo_hand_pose/yolo_hand_pose_ethos_u85_256.pte)
- [export_report.json](/home/mokuroo/documents/python/pt2pte_toolchain/artifacts/yolo_hand_pose/export_report.json)
- [debug_pte.md](/home/mokuroo/documents/python/pt2pte_toolchain/docs/debug_pte.md)
- [adapters.py](/home/mokuroo/documents/python/pt2pte_toolchain/src/pt2pte/adapters.py:87)

注意：部署侧必须从原来的单一 `[1,68,1029]` 输出解析，改为读取四个输出；旧 parser 不能直接使用新的 PTE。