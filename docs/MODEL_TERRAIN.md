# 模型地形模式：对象来自模型，UI 仍为合成像素

`terrain_source="model"` 是一个显式实验模式：七类对象 `player / monster / loot / portal / npc / platform / ladder` 全部来自传入检测器。HP、MP、地图标记与对话标记仍由自绘样例的固定像素规则读取。因此它是**模型对象与地形 + fixture UI**，不能称为整个管线都使用学习模型，也不是正式游戏或真实录像域验收。

实现位置：[感知组合](../src/maple_vision/agent.py)、[HUD-only 解析](../src/maple_vision/vision.py)、[运行与逐帧复验](../src/maple_vision/lab.py)。独立反例在 [test_model_terrain.py](../tests/test_model_terrain.py)。

## 模式与兼容性

| 参数 | 检测器 | 动态对象 | 平台、梯子 | HP/MP、地图、对话 |
|---|---|---|---|---|
| `fixture`，默认 | 无 | fixture 像素 | fixture 像素 | fixture 像素 |
| `fixture`，默认 | 有 | 仅检测器 | fixture 像素 | fixture 像素 |
| `model` | 必须提供 | 仅检测器 | 仅检测器 | fixture 像素 |

默认调用与原有 `HybridPerception(detector)` 保持原行为。既有运行包的感知身份、字段和来源标识保持兼容；历史像素包和混合包仍能按原方式实际重新推理。旧版仅有动作证据、没有终止感知证据的包仍只支持 `verify_run`，不会因新模式而升级为感知已复现。

CLI 中 `lab run`、`lab suite`、`lab soak`、`lab reinfer` 使用同名选择 `--terrain-source model`，并须同时提供 `--model`、`--manifest`；省略该选项仍为 `fixture`。`lab replay` 是动作重放，不使用模型或此选择。新模式包的 `run` 与 `reinfer` 两次命令都必须显式选择 `model`。

新模式不运行 `PixelPerception.detect`，只调用 `observe_hud`。模型返回什么对象框，就保留什么合法对象框，不根据可见调色板、模拟器地图几何或标签修补漏检，也不拼接模型碎框后冒充原始单框检测质量。运行感知的输入仍仅为 `ImageFrame`；模拟器状态只供渲染及独立验收使用。

## Python API

以下新增参数都是 keyword-only，允许的值严格为 `"fixture"`、`"model"`：

```python
HybridPerception(detector, *, terrain_source="fixture")
run_episode(output=None, config=None, detector=None, *,
            class_profile=None, sprite_pack=None, terrain_source="fixture")
run_suite(output, seeds=3, detector=None, *, terrain_source="fixture")
run_soak(output, seconds=60, detector=None, *,
         resource_interval_seconds=1.0, max_resource_samples=1024,
         terrain_source="fixture")
verify_perception(output, detector=None, *, terrain_source="fixture")
```

例如，给本地 ONNX 模型选择模型地形，并用相同模式实际重新推理：

```python
from pathlib import Path
from maple_vision.lab import RunConfig, run_episode, verify_perception
from maple_vision.onnx_detector import OnnxDetector

detector = OnnxDetector(Path("model.onnx"), Path("manifest.json"))
output = Path("artifacts/model-terrain-run")  # 必须是新路径
summary = run_episode(
    output,
    RunConfig(seed=0, job="assassin"),
    detector,
    terrain_source="model",
)
reproduced = verify_perception(output, detector, terrain_source="model")
```

运行失败也可以被正确复现。`perception_reproduced=true`、`policy_reproduced=true` 表示重新执行的结果与证据一致；任务完成仍须 `status="completed"` 且 `independent_goal_check=true`。这些不同结论不互相替代。

## 参数、类别与未知状态

缺少检测器、未知模式值或模型 manifest 未声明完整七类，都会在创建输出目录或第一次检测前明确失败。类别顺序由完整模型 manifest 决定；顺序不是固定类别编号的猜测。自定义检测器可以用于实验，但缺少真实模型哈希绑定时，仍不能通过 `verify_perception` 的模型身份门禁。

模型对象输出必须是不可变的 `tuple[Detection, ...]`，类别属于上述七类，框在原始输入图像内；`Detection` 已约束有限坐标、正面积和 `[0,1]` 置信度。错误输出抛出异常，不转为 fixture 地形。合法空结果、缺失玩家或多个玩家保持未知；没有平台/梯子框时，相应对象集合就是空，不补一个“已知地板”。

全局 `KNOWN` 只表示唯一玩家以及 HP/MP、地图标记可读，**不表示场景对象都检出，更不表示不存在未检出的地形**。任何资源条或地图标记无法读取时保留 `None`，新模式观察为 `UNKNOWN`，交由现有 Supervisor 的有界暂停/停止处理。对话仍是合成标记是否可见，并非 OCR 或真实聊天理解。

## 运行身份与重新推理

新模式的每帧 `source_id`、运行 `settings.pipeline` 均为：

```text
onnx-all-objects+fixture-hud-v1
```

`perception_evidence.settings` 另外显式保存 `terrain_source="model"`，并继续绑定实际加载字节的 SHA-256、完整 manifest、检测阈值与实现身份。默认 fixture 模式保持 v0.2 的 settings 原字段，不添加推测性迁移字段；其原 pipeline 身份已经明确区分像素和混合来源。

`verify_perception` 不从证据自动替调用者选择模式。传错模式、模型、manifest 或阈值，会在任何 `detect` 调用之前拒绝。必须显式带上 `terrain_source="model"` 才能复验新模式包；删除或改写模式字段不能获得默认兼容。动作重放 `verify_run` 仍不运行检测器，并明确 `perception_reproduced=false`。

匹配后，从原初始状态、嵌入职业参数/素材和故障配置重新生成帧，实际运行感知、Supervisor 与策略，比较完整轨迹以及终止观察。若有 N 个执行动作，会实际重新感知 N+1 帧；监督和策略只在原流程确实执行的位置比较。步数耗尽的终止帧没有虚构的额外策略调用。持续黑屏停止、短暂黑屏、重复帧、冻结输入等仍按原始终止证据处理，复验不修改运行包。

浮点观测允许 `abs_tol=rel_tol=1e-6`，对象顺序、身份、离散字段、动作和决策精确比较；环境身份差异单独报告。模式一致与输出复现不能保证不同硬件、库版本或新画面的预测逐字节相同。[完整复验合同](MODEL_VALIDATION.md)

## 验证与验收边界

新增回归覆盖隐藏地形颜色后仅使用模型框、明显可见的地形未被模型返回时不回填、全类别合同、非法框、缺失 UI、错误模式提前拒绝、真实 ONNX 终止帧和故障复验、原始失败保留，以及默认 v0.2 身份兼容。原生常量测试图只用于测试来源与证据，不是训练质量成绩；已随仓库保存的模型在新模式上的短运行复验，也不构成新模式任务成功率。

模式可选择不代表地形模型已达到验收标准。新模型需要独立记录同一数据上的 train/val 选择、选择后的保留 test、原始平台/梯子检测质量、三职业闭环、故障结果和资源报告。旧模型、历史报告、原始标签及评价 IoU 不因模式增加而更改。HUD 仍为合成特定规则；真实域、未见地图和 24 小时墙钟验证继续分别记录，不能由该开关或短时复验推导通过。
