# 地形模型结构预检

本工具回答一个有限问题：**在指定输入尺寸、预处理和本地 YOLO 实现下，数据集中的地形框是否有可用候选锚点，且该锚点到框四边的训练目标无需截断？** 它不读取训练结果、不计算识别精度，也不证明最终任务成功。运行时 Agent 仍只接收图像与感知结果；这里使用的标签属于训练和独立验证侧。

## 使用方式

需要已安装本项目的 `training` 可选依赖和一个通过 `dataset validate` 的本地数据集。所有模型从本地 YAML 创建随机权重；不获取预训练权重、不联网。第三方库只在 `training._offline_runtime` 中导入，Pillow loader、库设置和线程数在退出时恢复。建议在单独 CLI 进程运行，因为离线网络 guard 是进程级协作保护。

```bash
python -m maple_vision model audit-terrain artifacts/v03-experiment/dataset \
  --config configs/terrain-yolo.yaml \
  --output artifacts/v03-experiment/new-terrain-audit.json \
  --split val --image-size 320 --warmup 3 --iterations 20 --seed 42
```

当训练或其他计算正在占用 CPU 时，使用 `--iterations 0`。这仍会执行构建所需的内部探测和一次不计时前向，检查真实输出与网格；`cpu_forward.latency_ms` 为 `null`，不会伪造空闲机器延迟。已有输出路径一律拒绝，报告不能写入源数据集目录。

Python 接口：

```python
from pathlib import Path
from maple_vision.terrain_audit import audit_terrain_architecture

report = audit_terrain_architecture(
    Path("artifacts/v03-experiment/dataset"),
    Path("configs/terrain-yolo.yaml"),
    Path("artifacts/v03-experiment/new-geometry-only-audit.json"),
    split="val",
    image_size=320,
    warmup=3,
    iterations=0,
    seed=42,
)
```

`split` 只接受 `train` 或 `val`，避免用 test 做结构选择。完整数据集验证仍检查所有 split 的文件完整性；只有所选 split 的标签进入地形分析。报告同时记录 `test_targets_analyzed=false` 与 `complete_dataset_integrity_checked=true`，两者含义不同。

## 检查什么

1. **构建前预算。** 在导入原生模型库前检查原配置及实际使用的配置副本。当前仅支持串行 `Conv/C2f/SPPF` backbone 加单个普通 `Detect`；七类、最多 16 层、声明通道数不超过 128、width/depth 不超过 1、Conv stride 为 1 或 2，Conv/SPPF 只重复一次、C2f 最多重复两次。拒绝 Upsample、Concat、额外参数、端到端头和未知结构。输入尺寸须为 32 的倍数且在 64–640 之间。静态网格和 feature/head allocation 还有独立预算；小尺寸审计也按构建时的 256px 探测核算。该 guard 限制分配形状，不是操作系统内存隔离或硬实时保证。
2. **实际检测头。** 构建模型后核对 `Detect` 类型、实际 strides、DFL 权重是否为 `0..reg_max-1`、特征尺寸、输出尺寸及训练 loss/assigner 类型。执行真实 decode、`bbox2dist` 和 `DFLoss` clamp 探针；未知接口或行为直接失败。配置中的声明不能替代实际检查。
3. **实际候选规则。** 使用该模型 `init_criterion().assigner.select_candidates_in_gts`，不是按框宽猜测。相同输入框几何只计算一次，按有界批次调用，再将结果映射回每个 GT。当前最多 50,000 个地形 GT、4,096 种唯一几何、100,000 个 anchors，每批至多约一百万 candidate pairs。
4. **每边回归。** 对每个候选锚点，分别计算 left/top/right/bottom 距离并除以该锚点的 stride。四边全部满足范围才算覆盖，不能只检查总宽度。
5. **可追溯输入。** 报告绑定实际消费的模型配置字节、数据集 manifest、安装库版本及主要实现源码哈希。选中标签逐文件核对 manifest 哈希后进入内存；manifest 在验证前后、审计结束时核对，配置副本结束时核对。报告先完整写入同目录临时文件，再以独占链接发布，不覆盖已有报告。

预处理固定为居中方形 letterbox、`auto=false`、`scale_fill=false`、`scaleup=true`，使用安装版 `LetterBox` 参数和 `Instances` 坐标操作。这里不覆盖随机几何增强、矩形 batch、不同后处理或未来其他检测头。

## 为什么必须检查四条边

令 DFL 有 (R) 个 bins，某锚点 stride 为 (s)。本地实现的解码距离是 `0..R-1` 的 softmax 加权平均，因此每边的支持闭包是：

\[
0\le d_{\mathrm{left}},d_{\mathrm{top}},d_{\mathrm{right}},d_{\mathrm{bottom}}\le (R-1)s.
\]

总宽度上界 (2(R-1)s) 只是必要条件。训练实现还将每边的 stride 单位目标限制在 `[0, R-1-0.01]`。报告分别记录候选数、支持闭包数、训练无需截断数、严格内部支持数、负距离及超界数量，并给出一个可核对的锚点与 LTRB witness。

| 独立反例 | 仅看总宽度为何会误判 |
|---|---|
| 320×4 的框位于输入 y=232..236；stride32 最近一行锚点在 y=240 | 448px 总宽度虽然足够，但本地 assigner 将高度扩成 y=230..238 后仍没有该层候选行。 |
| 100px 宽框、stride8、reg_max8，锚点 x=4 | 总宽上限 112px，但到右边的距离 96px 超过单边 56px；锚点 x=52 才能同时满足两边。 |
| 小框扩展后接纳位于原框上方的锚点 | 候选规则通过，但原框顶部目标是负数，非负 DFL 无法表示。 |
| 距离 6.995 stride units，reg_max8 | 落在支持内部，却超过训练 clamp 上限 6.99。 |

零距离是有效训练目标，但有限 logits 的 softmax 通常只能逼近端点。因此 `training_unclipped_anchors>0` 不等同“有严格内部锚点”，更不等同“训练后上边缘误差为零”。上述反例均有独立数值测试。

## 本轮候选与实测预检

候选为 [configs/terrain-yolo.yaml](../configs/terrain-yolo.yaml)，包内资源保持逐字节相同。它保留旧 backbone，追加一个 64 通道 stride2 Conv，将 Detect 输入改为 `[2,5,6]`，实际 stride 为 `[4,8,16]`，`reg_max=16`。配置 SHA256：

`b4a42896b69b9a55d1fc2324e2c4b55cc12d8fdd4ac69ab26583957b00839111`

新 v0.3 数据集 manifest SHA256：

`f73d7390980fa6aedea3b3a12ada2f0a8f3658519e0ca435ca93414bbd02aac4`

该数据集 val 包含 192 图、576 个平台框和 192 个梯子框。正式不计时审计中仅有 8 种唯一输入框，实际 assigner 一批处理 67,200 对候选关系：

| 完整四边训练目标覆盖 | 旧模型结构 `[4,8], R=8` | 候选结构 `[4,8,16], R=16` |
|---|---:|---:|
| platform | 104 / 576 | 576 / 576 |
| ladder | 192 / 192 | 192 / 192 |
| 全宽薄平台子集 | 0 / 192 | 192 / 192 |
| 全部地形 | 296 / 768 | 768 / 768 |

这是**完整框训练目标覆盖率**，不是模型 recall。一个无法完整回归 GT 的结构仍可能输出部分重叠框达到 IoU50；两个指标不能互换。

对 `[0,232,320,236]` 地板，新增 stride16 层有 20 个候选锚点，`reg_max16` 下其中 10 个四边无需截断，x 从 88 到 232、间隔 16，y=232。改成 `reg_max12` 的数学比较只剩 x=152、168 两个；本轮未训练 reg_max12，也未将它纳入候选搜索。推荐 16 保留更宽的候选范围。地板的顶部距离为零，严格内部支持数量仍为零，后续必须测上边缘和端点定位误差。

### 独立 CPU 成本探测

在正式训练开始前，使用同一机器、随机权重、固定零输入 `[1,3,320,320]`、FP32、2 个 PyTorch 线程分别测量。预热 3 次，计时 20 次；包括 backbone/head/DFL decode，不含预处理、I/O 或 NMS。该测量与数据集内容无关，正式新 val 审计没有在训练竞争 CPU 时重新计时。

| 结构 | 参数量 | 前向 p50 | 前向 p95 |
|---|---:|---:|---:|
| 旧 tiny | 165,798 | 17.47 ms | 39.20 ms |
| 候选 terrain | 398,357 | 22.06 ms | 29.54 ms |

旧模型的一次样本达到 84.99ms，反映共享机器抖动。不能因候选 p95 较低而宣布它更快，也不能用这些数值直接推算训练时长。逐次原始计时保留在 `artifacts/v03-experiment/preliminary-terrain-audit.json`；正式几何证据为同目录 `terrain-candidate-audit.json` 和 `terrain-baseline-audit.json`。首批结构报告生成后才增加构建前预算 guard；这些旧报告不会被改写成“当时已经执行新增 guard”。

### 与质量评估的全宽地板分组区分

本报告的 `full_width_thin_platforms` 是结构诊断子集：原图左边约为 0、右边约为图宽（绝对容差 0.001px），且高度不超过 8.001px。

`terrain_metrics` 的 `full_width_floor` 则是**质量评估的 GT 分层**：平台左边为 0、右边为图宽，绝对容差 `1e-6`，没有高度限制。其召回、边缘/端点误差由真实预测与标签匹配产生。两组在当前 fixture 地板上重合，但名称、容差、作用和指标都不同，不能视为通用同义字段。质量规则见 [EVALUATION.md](EVALUATION.md) 和 [MODEL_TERRAIN.md](MODEL_TERRAIN.md)。

## 验证与后续准入

`tests/test_terrain_audit.py` 包括数学反例、非法输入、构建前资源拒绝、配置资源一致性、源文件篡改和独占报告测试。`MAPLE_TRAINING_SMOKE=1` 才运行实际新旧结构审计、未知 DFL 头拒绝，以及候选的 1 epoch 训练 → ONNX 导出 → ORT 推理技术链测试。该 smoke 不设置精度门槛，不能替代正式训练实验。

```bash
python -m pytest tests/test_terrain_audit.py -q
MAPLE_TRAINING_SMOKE=1 python -m pytest tests/test_terrain_audit.py -q
```

结构预检通过后，仍需执行冻结训练计划、仅使用 val 作候选决定、最终 held-out test、模型地形来源隔离测试、实际模拟器任务与感知重推理，以及有明确时长和资源采样的稳定性验证。真实地图素材、真实录像和正式游戏规则属于独立的外部验收；本工具不能将合成域结果转换成这些验收结论。
