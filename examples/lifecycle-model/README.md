# 生命周期合成场景 YOLO 检查点（v0.2）

本模型由项目自绘场景从零训练，用于离线推理与独立模拟器验证。实际通过的范围为 **YOLO 动态对象＋fixture 像素地形/HUD 的混合管线**；不代表纯 YOLO 地形检测或真实游戏识别已经通过。

| 项目 | 值 |
|---|---|
| 配置 | `configs/tiny-yolo.yaml` |
| 训练 | CPU 2 线程，320px，40 epoch，batch 8，seed 42，无预训练权重 |
| 数据 | lifecycle v2，18 会话、864 帧；train/val/test 为 480/192/192 |
| 输入 / 输出 | float32 RGB NCHW `[1,3,320,320]` / `raw_xywh` `[1,11,8000]` |
| ONNX SHA-256 | `7e9f61576f63be400909db6f9eb6cb89b5b935385d01a24155c7f765d2beb6e3` |
| 数据 manifest SHA-256 | `670f6dae181f0f4b3107ba6a47cb733f19cfbc216c4e8e8c23f96b756601b3ff` |
| 保留 test | 192 帧、1572 个目标；TP 1372、FP 3、FN 200 |
| test 截断 AP50 / micro P / micro R | 0.944822 / 0.997818 / 0.872774 |

AP50 使用 confidence≥0.25、IoU=0.5；不是完整 COCO mAP50–95。候选只依据 val 和独立合成任务选择，固定后一次评估 test。选模工程门禁在看过首候选 val 后记录，不冒称预注册实验。旧检查点保留在 `examples/fixture-model/`。

```bash
python -m pip install -e ".[inference]"
python -m maple_vision lab run --job assassin --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json --output artifacts/lifecycle-model-run
python -m maple_vision lab reinfer artifacts/lifecycle-model-run --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
python -m maple_vision lab suite --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json --output artifacts/lifecycle-model-suite --seeds 3
```

所有输出使用新路径。新运行包能实际重新感知 N+1 帧，并复验实际发生的监督与策略决策及终止证据；预算耗尽或监督停止等路径保留未执行决策的 `null`，不补造调用。旧版 `lab replay` 仍只验证动作重放。依赖安装后推理本地执行，无 LLM API；入口在导入 ONNX Runtime 前禁用遥测。

**已知限制：** 实际检测头为 `reg_max=8`、stride 4/8，单框最大宽度为 112 像素。320 像素全宽地板的单框 IoU 上限仅为 0.35，因此不会通过 IoU50；test 平台召回为 0.661458。混合任务明确使用像素地形，动态对象没有颜色回退。三职业两图任务和短时 soak 成功，不能证明未见地图、真实域或 24 小时可靠性。

完整训练、误差诊断、分组结果、重推理和资源记录见 [生命周期模型实验](../../docs/YOLO_LIFECYCLE_EXPERIMENT.md) 与 [v0.2 验收](../../docs/RELEASE_0_2_ACCEPTANCE.md)。
