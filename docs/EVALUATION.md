# 检测评估与基准

## 命令背后的 Python 接口

```python
benchmark_dataset(
    dataset, output, detector=None, split="test", confidence_threshold=0.25, iou_threshold=0.5,
    *, terrain_metrics=False
)
```

`detector=None` 使用合成调色板基线；传入 `OnnxDetector` 才测试实际 ONNX 推理。
调用先验证数据集，再读取独立 YOLO 标签；评估不读取模拟器状态或渲染器标签接口。
输出为独占创建的 JSON 文件，已有报告不会覆盖。

显式指定 `terrain_metrics=True` 或 CLI `--terrain-metrics` 会输出 schema3，保留已有指标，并从相同预测追加全宽地板召回和原图坐标下的顶边/端点误差。没有匹配框时误差为未知，不写成零；详情见 [地形专项评估](TERRAIN_EVALUATION.md)。默认旧报告不增加这些字段。

纯计算接口供独立测试使用：

```python
samples = [EvaluationSample(ground_truth=(...), predictions=(...))]
evaluate_predictions(samples, class_names=CLASS_NAMES, confidence_threshold=0.25, iou_threshold=0.5)
```

## 匹配和指标定义

预测先应用明确记录的置信度阈值，再按类别、置信度降序匹配。同分保持输入顺序。
一个预测最多匹配同一图片内一个同类别真实框；优先选择未占用且 IoU 最大的框，IoU 等于阈值时计为匹配。
一个真实框最多匹配一个预测。重复检测计 FP，错误类别同时产生 FP 与 FN，未登记的预测类别也计 FP。

汇总提供 micro TP/FP/FN、precision、recall 和 F1；每类提供同样统计和 AP。
precision 在没有预测时为 null，recall/AP 在没有该类真实目标时为 null，而非假装 0 或 1。
F1 使用 `2TP / (2TP + FP + FN)`；分母为 0 时为 null。
宏平均 AP 只纳入有真实目标的类别，同时记录纳入类别数和全部类别计数，避免隐藏缺失类别。

AP 使用 0 到 1、间距 0.01 的 101 个召回率采样点，先构造向右的 precision 最大值包络，未达到的召回位置取 0。
这采用了 [COCO 官方评估实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) 中的召回采样与包络方法。
本项目只计算声明的单一 IoU，不实现 COCO 的 crowd/ignore、面积桶、maxDets 汇总或 0.50:0.05:0.95 多阈值平均。
因此不是完整 COCO mAP50–95；报告中的 `map50_95` 始终为 null。

IoU 为 0.5 时 `ap50` / `map50` 有值；其他 IoU 时仅 `ap_at_iou` / `map_at_iou` 有值，避免把 AP75 等误标成 AP50。
AP 在声明的置信度截断后计算；若 ONNX 适配器自身有更高阈值或 NMS，也会先影响候选集合，相关参数记录在 detector 元数据中。
比较不同模型时必须保持阈值、NMS、输入尺寸、类别、数据和匹配设置一致。

## 时延与可复现性

先对最多 3 张图片热身，热身不进入时延样本；随后完整测试指定分区。
计时范围是 `detect(frame)`，包含该适配器的预处理、推理和后处理，不包含 PNG 解码、磁盘读取和标签读取。
使用 `perf_counter_ns`，报告 p50/p95、平均、最小、最大和样本数；分位数使用相邻排序样本线性插值。
同时保存 Python、操作系统、CPU 架构、逻辑核数、依赖版本、数据 manifest 哈希和模型哈希（适用时）。
这些结果只描述当前机器和这次运行，不能当成所有 Windows/Linux 机器的性能保证。

## 证据边界

当前数据集来源是 `synthetic-fixture-v1`，报告明确标为 `synthetic_fixture`。
test 会话与 train/val 分开，并不自动证明模型在别处从未见过这些图像；`model_training_overlap_verified` 保留为 false。
小样本、高重复场景或调色板基线的高分，都不能替代独立真实录像、跨新地图泛化、遮挡压力测试和实际操作闭环验证。

测试包含可手算的 AP 例子、重复框、跨图片误匹配、错误类别、无真实目标、漏检、置信度边界、IoU 边界、未知类别、报告防覆盖和损坏数据阻止推理。
