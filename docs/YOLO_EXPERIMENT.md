# YOLO 合成数据实验与选模：2026-11-24

本次实际执行了 **生成数据 → 从头训练两个候选 → 按验证集选择 → ONNX 导出 → ONNX Runtime smoke 推理**。最终选定 320 像素、60 epoch 候选，验证集 mAP50 为 **0.9429702**，mAP50-95 为 **0.8892385**。

所有图像、地图和规则均为本项目自写的几何图形模拟器测试素材，scope 为 `synthetic_fixture`。本报告没有真实游戏识别效果或账号安全结论。模型输出的保留测试集评估由独立入口另行执行；选模过程没有使用保留测试集推理结果。

## 固定数据与实验设置

| 项目 | 本次实际值 |
|---|---|
| 数据集生成种子 | 20261124 |
| 会话与帧数 | 18 会话 × 16 帧 = 288 张 PNG |
| 总标注框数 | 2,859 |
| 类别 | player、monster、loot、portal、npc、platform、ladder |
| 模拟地图 | meadow、cavern；两张地图在三个 split 均有出现 |
| 数据划分单位 | 完整会话；同一会话不会跨 split |
| 训练集 | 10 会话，160 帧，1,593 个标注框 |
| 验证集 | 4 会话，64 帧，633 个标注框 |
| 保留测试集 | 4 会话，64 帧，633 个标注框 |
| 结构 | `configs/tiny-yolo.yaml`，本地 Conv/C2f/SPPF/Detect 小型双尺度检测模型 |
| 初始化 | 从头训练，pretrained=False，无预训练下载 |
| 固定训练参数 | batch=8、seed=42、AdamW、lr=0.001、warmup_epochs=0 |
| 执行设备 | CPU，PyTorch 2 线程，DataLoader workers=0 |
| 增强 | 透视、旋转、剪切、翻转、Mosaic、MixUp、Copy-Paste、颜色扰动均关闭 |
| 选模依据 | 验证集 mAP50-95；每个候选使用其验证集最佳 checkpoint |

图像和标签先经过独立完整性检查，检查包括文件 SHA-256、类别范围、框坐标边界、会话与帧顺序、跨 split 重复图像、空 split 和路径约束。原始数据集的 manifest 和各图片/标签哈希在两次候选之间保持一致。

## 两个候选的实测结果

以下来自 Ultralytics 对各自最佳 checkpoint 的 **验证集** 汇总。它们不是固定 confidence=0.25 下的保留测试集结果，也不能直接等同于“角色识别准确率”。

| 指标 | 候选 A：160px，30 epoch | 候选 B：320px，60 epoch |
|---|---:|---:|
| precision(B) | 0.6971001185 | 0.9591662127 |
| recall(B) | 0.6760712284 | 0.9348114246 |
| mAP50(B) | 0.7021576290 | 0.9429702087 |
| mAP50-95(B) | 0.6500317161 | **0.8892384878** |
| 训练函数墙上耗时 | 98.245567 秒 | 571.254583 秒 |
| 所记录完整流程耗时 | 100.507403 秒（含生成） | 572.217825 秒（复用同一数据） |
| ONNX checker | 通过 | 通过 |
| ONNX Runtime smoke 推理 | 通过 | 通过 |
| 选模结果 | 保留为对照 | **选定，交给独立保留测试集评估** |

两次候选同时改变了输入尺寸和训练轮数。因此它们能支持本次选模决定，不能用来区分“提高分辨率”和“增加训练步数”各自贡献。验证集只有 4 个会话，邻近帧有相关性；这些数据还不足以证明跨真实场景泛化或原计划中的高准确率目标。

## 实现中发现并修复的问题

第一次训练结束后，Ultralytics 在原始标签目录生成了 `train.cache`、`val.cache`。库的 `cache=False` 只关闭图像缓存；更新后的严格数据集校验正确拒绝了这两个未登记文件，没有降低校验标准。

训练封装随后改为：按 manifest 复制输入到本次运行的 `input_snapshot`，再次核对内容，第三方缓存只写入这个私有工作目录；训练结束清理私有标签缓存并重新验证快照。原始实验目录仅移除了已确认由第一次训练产生的两个缓存文件，manifest 和登记图片/标签均未更改。第二次训练前后，原始数据集和训练快照验证都通过。真实训练回归测试已覆盖该隔离要求。

## 最终 ONNX 契约

| 项目 | 候选 B 实际导出值 |
|---|---|
| 输入 | float32，`[1, 3, 320, 320]` |
| 输出 | `[1, 11, 8000]`，11 = 4 个 xywh 参数 + 7 类分数 |
| manifest output_format | raw_xywh |
| 导出 | opset=17、dynamic=False、nms=False、simplify=False |
| 推理提供方 | CPUExecutionProvider |
| 模型来源 scope | synthetic_fixture |

推理端在 `onnx_detector.py` 内完成预处理、尺寸还原和外部 NMS，并在加载前核对模型哈希。导出成功与验证集选模不会自动把模型提升为真实场景已验收检测器。

## 版本与内容指纹

实际依赖：Ultralytics **8.4.218**、PyTorch **2.15.0+cpu**、NumPy **2.5.5**、Pillow **13.0.0**、ONNX **1.23.2**、ONNX Runtime **1.32.0**。不同版本或平台可能产生不同权重与指标；相同 seed 不构成跨平台逐字节一致性承诺。

| 文件 | SHA-256 |
|---|---|
| 两次使用的数据集 manifest | `ecc0ff6c4e9bbebc9915f0313bb3b227a99ecc321051ee2ae2e8a0eff096ad67` |
| 模型 YAML | `122a5e4040cdf15f5acdd692974c55b1b3260794601b1f8d5756595866b81bf6` |
| 候选 A best.pt | `9f0a88172b28557ef87ad3efd18aab4fad15b51156411d838e6c603ba98dfd2e` |
| 候选 A model.onnx | `c9fcb1a81c4d347d7e4a1e33445047b1898e6e6e744d67ae7e007d985e38bf28` |
| **候选 B best.pt** | `0b1b9e2d3fdbc71fa4296efbe4226fa0bdea31d4b87ac6dcfd6856e214e7abec` |
| **候选 B model.onnx** | `56cafce92c63d284744b9030d661486911e30c46b03000c605b530bc91d9ae04` |

本地结果位于 `artifacts/yolo-experiment/experiment.json` 和 `experiment-320.json`；逐 epoch 指标位于相应训练目录的 `run/results.csv`；日志为 `experiment.log`、`experiment-320.log`。首次 320px 预检失败日志保留为 `experiment-320-preflight-failure.log`。这些二进制和批量生成数据为可重建实验产物，不直接提交到源码 Git 历史。

最终选定模型为 `artifacts/yolo-experiment/export-320/model.onnx`，相邻 `manifest.json` 用于哈希校验与输入输出合同。对应权重为 `artifacts/yolo-experiment/training-320/run/weights/best.pt`。

## 再执行最终候选

使用新的输出目录，安装训练与推理可选依赖后运行：

```python
from pathlib import Path
from maple_vision.dataset import generate_dataset
from maple_vision.training import train_yolo, export_yolo

root = Path("artifacts/yolo-reproduction")
generate_dataset(root / "dataset", sessions=18, frames_per_session=16, seed=20261124)
trained = train_yolo(
    root / "dataset",
    root / "training",
    Path("configs/tiny-yolo.yaml"),
    epochs=60,
    image_size=320,
    batch=8,
    seed=42,
)
export_yolo(Path(trained["weights"]), root / "export", image_size=320)
```

工程回归另有 `MAPLE_TRAINING_SMOKE=1 python -m pytest -q tests/test_training.py`：隔离修复后 34 项全部通过，包括真正训练、导出、源数据与副本重新校验，并由 `OnnxDetector` 读取模拟器渲染帧。普通测试不会把这个可选的深度学习集成测试冒充为已执行。


## 选定后独立测试与闭环（2026-11-24）

选定上述 320px / 60 epoch 检查点后，根代理在 64 帧 test 分区进行了独立评估。confidence=0.25、IoU=0.5，AP 使用该置信度截断后的 101 点插值精度包络。

| 指标 | 实测 |
|---|---:|
| 目标数 / TP / FP / FN | 633 / 535 / 0 / 98 |
| micro precision | 1.000000 |
| micro recall | 0.845182 |
| macro AP50 | 0.926450 |
| CPU detect p50 / p95 | 16.65 / 40.94 ms |

该结果不是完整 COCO mAP50–95。平台类别召回率约 0.4948，是当前模型的明确短板；其他动态目标表现较好，仍不能外推到真实素材。完整逐类计数、硬件/软件信息和模型/数据集哈希见 [原始报告](../reports/release-0.1.0/yolo-benchmark.json)。小样本及单机时延并非长期或跨硬件保证。

用于任务验证的混合感知器明确采用 YOLO 动态对象 + 合成场景像素地形/HUD。动态对象缺失时不会用调色板结果冒充 YOLO。标飞、龙骑、牧师分别在 327、349、317 个模拟步完成两图任务；标飞包含两次真实模拟跳跃攻击，所有任务均由独立真值校验并重放。报告见 [混合闭环结果](../reports/release-0.1.0/yolo-missions.json)。这三个样例只证明该检查点可接通系统，不足以证明正式域可靠率。

评估中发现原生 ONNX Runtime 的默认遥测会尝试外联，自动审批拒绝了该连接。修复后，包入口和推理/导出边界均在初始化原生库前设置 `ORT_DISABLE_TELEMETRY=1`，并调用关闭遥测 API；CI 同样在进程启动前设置环境变量。上述报告来自关闭遥测后的成功重测。实现及导入顺序有回归测试。官方说明：[ONNX Runtime telemetry API](https://onnxruntime.ai/docs/api/python/api_summary.html#telemetry-events)。
