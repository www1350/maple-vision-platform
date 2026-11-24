# 合成场景 YOLO 检查点

此模型由本项目原创几何场景从零训练，随仓库提供，方便在本地复现推理和闭环验证。它不是官方游戏识别模型。

- 结构：`configs/tiny-yolo.yaml`；Ultralytics 8.4.218，CPU 训练。
- 数据：18 个会话、288 帧；训练/验证/测试为 160/64/64 帧。
- 选模：只比较验证集，选择 320px、60 epoch 的候选。测试集只用于选定后评估。
- 输入：float32 RGB NCHW `[1,3,320,320]`；输出：`raw_xywh` `[1,11,8000]`。
- 文件 SHA256：`56cafce92c63d284744b9030d661486911e30c46b03000c605b530bc91d9ae04`。

```bash
python -m pip install -e ".[inference]"
python -m maple_vision lab run --job assassin --model examples/fixture-model/model.onnx --manifest examples/fixture-model/manifest.json --output artifacts/model-demo
python -m maple_vision lab replay artifacts/model-demo
```

闭环采用 YOLO 检测动态对象，并用合成场景专用的像素解析器读取地形、HP/MP、地图和对话标记。策略不读模拟器真值，也没有动态对象的调色板回退。模型在直接检测全部七类的测试中，平台召回率仍偏低；混合闭环通过不能被解释为纯 YOLO 平台定位已达标。

测试集 64 帧上的截断 AP50 约 0.92645；confidence=0.25、IoU=0.5，完整计数及限制在 [评估报告](../../reports/release-0.1.0/yolo-benchmark.json)。这不是完整 COCO mAP50–95。三职业混合闭环报告在 [任务报告](../../reports/release-0.1.0/yolo-missions.json)。

重现训练、来源和验证集比较见 [实验记录](../../docs/YOLO_EXPERIMENT.md)。推理依赖本地 ONNX Runtime；包入口在导入原生库前设置 `ORT_DISABLE_TELEMETRY=1`，并调用其关闭遥测 API。嵌入到其他程序时，应在宿主导入 ONNX Runtime 之前设置同一环境变量。
