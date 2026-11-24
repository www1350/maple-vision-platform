# 地形评估：召回率、顶边与端点

`vision benchmark --terrain-metrics` 在同一批检测结果上追加地形几何报告。它解决一个具体问题：通过 IoU=0.5 的框，仍可能不足以支持平台定位。例如，宽度只有真值一半的地板框可以刚好通过 IoU50，但另一端会偏离半个画面；高度翻倍的框也可能通过，顶边却不准确。

这个报告读取可见标注与检测器输出，不访问模拟器碰撞体，不重建遮挡后的真值，也不合并碎框来改变原始模型检测结果。当前实验门禁由训练前保存的 [preregistration.json](../reports/release-0.3.0/preregistration.json) 定义；评估器本身不会替用户选择及格线。

## 使用

```bash
python -m maple_vision vision benchmark artifacts/v03-experiment/dataset \
  --output artifacts/terrain-validation.json --split val \
  --model local-model/model.onnx --manifest local-model/manifest.json \
  --terrain-metrics
```

Python 调用是在既有 API 后追加 keyword-only 参数：

```python
benchmark_dataset(dataset, output, detector, split="val", terrain_metrics=True)
```

省略该参数时，legacy / lifecycle 报告仍分别为 schema 1 / 2，字段和既有指标不变。显式启用时，报告为 schema 3，保留原有聚合和 lifecycle 分组指标，并新增 `terrain_geometry`，其内部 schema 为 1。这不改变数据集或模型 manifest 的版本。

每张图只执行一次用于评分的 `detect`，加上原有至多三次预热；地形统计直接复用已得到的预测。`latency_ms` 仍只计 `detect(frame)`，包括预处理、推理和解码，不包含 PNG 读取、匹配或生成报告。

schema 3 的原生 ONNX 报告另外记录 `detector.full_model_manifest`、实际已加载字节的 `loaded_model_sha256` 和 `inference_runtime`。运行选项从已创建的 ORT session 读取，包含实际 providers、intra/inter-op 线程与 execution mode；它们不是依据训练线程数猜出的值。首次预热前与评分后都检查同一份模型/manifest/阈值/运行选项；发现改变就拒绝保存报告。该检查不能证明测量期间整台机器没有其它进程，实验过程仍须记录并避免同时训练。

## 匹配与分层

平台、梯子分别做普通的一对一匹配：同一张图、同一类别，先按预测置信度排序；同置信度按原输入顺序，同 IoU 按原真值顺序。每个预测只匹配一个尚未匹配的真值，IoU 达到所声明阈值才算成功。预测置信度低于阈值时不进入统计。匹配约定与通用评估器一致。

**先在全部平台真值上完成匹配，再提取整条地板子集。** 这防止一个更适合局部平台的预测，又被拿来单独“证明”整条地板已经检出。整条地板的定义仅是：类别为 `platform`、原图坐标 `left=0` 且 `right=image_width`，绝对容差 `1e-6`。它不是固定 320 像素，也不推断某个框一定可行走或属于底层地板。

| 字段 | 含义 |
|---|---|
| `ground_truth / tp / fn / recall` | 包含所有对应真值，漏检继续计入召回率分母 |
| `platform.fp / ladder.fp` | 置信度阈值后未匹配的该类预测 |
| `full_width_floor.fp` | 为 `null`，这是按真值分层的集合，没有独立预测类别；误检看全部平台 |
| `absolute_error_pixels.left/top/right/bottom` | 成功匹配后，对应边坐标的绝对误差，单位是原始图像像素 |
| `absolute_error_pixels.width/height` | 匹配框宽、高的绝对误差 |
| `matched_iou` | 成功匹配对象的 IoU 分布 |
| `worst_matched_examples / missing_examples` | 最多 20 个案例，包含图索引、真值索引及匹配时的预测框 |

每个误差分布报告 `samples / mean / p50 / p95 / max`，分位数用 `(n-1)*q` 处的线性插值。左右端点分别报告，不能把两端合成一个平均数掩盖某一端偏差。最差案例按六个误差中的最大值排序，样本总数和指标不受案例截断影响。

## 空集、漏检与解释边界

匹配框误差是**条件统计**。如果 100 个地板只匹配到一个，那个框的误差可以为零，但召回率仍只有 0.01；不能据此认为地形已经可靠。没有匹配框时，误差样本数为 0，均值、分位数和最大值都是 `null`，不会把未知写成零误差。

没有整条地板真值时，`ground_truth=0`、`recall=null`。本轮模型准入要求地板真值及匹配样本非空，不能让空集合通过误差门禁。

输入必须是一一对应的 `EvaluationSample` 和 `(width, height)`，坐标在声明原图范围内。每张图每一侧的地形对象最多 500 个，最多 100000 张图和一百万个输入对象；超限明确拒绝，以限制额外匹配的工作量。通用评估仍负责所有七类的 AP、TP/FP/FN 与未知类别误检；地形报告不替代它。

这套指标描述所给数据，不自动证明模型未见过这些图片、通过未见地图，或达到真实游戏精度。本轮 train/val/test 使用独立合成会话，但共享两张内置地图；关联帧不能当作相同数量的独立任务。专项检测、任务闭环、实际重新推理和长时间运行也必须分别验收。

## 独立反例

[test_terrain_metrics.py](../tests/test_terrain_metrics.py) 用手写期望值验证：半宽框与增厚框虽然通过 IoU50，仍产生明确边缘误差；局部平台匹配不能被整条地板统计抢走；漏检不会成为零误差样本；重复框、低置信度、跨类别和同分排序保持一对一规则；不同图宽和已知数列的 p95 不依赖固定场景。另有完整 benchmark 检查评分帧数与实际检测调用次数，防止几何报表重复调用模型。
