# 本地 YOLO 训练、ONNX 导出与验证

## 已实现的流程

`training.py` 提供从本地 YAML 构建模型、在完整性校验通过的数据集上进行 CPU 训练、导出 ONNX、执行一次实际 ONNX Runtime 推理并生成模型 manifest 的闭环。

这条训练链路已经用合成图形数据真正执行过，实际训练权重与 ONNX 输出均已产生。两次完整候选训练及验证集选模结果见 [YOLO_EXPERIMENT.md](YOLO_EXPERIMENT.md)。训练完成和模型推理成功只说明工程链路可运行；检测质量由独立评估给出，不能据此声称真实游戏识别准确。

## 输入与输出

```python
from pathlib import Path
from maple_vision.dataset import generate_dataset
from maple_vision.training import train_yolo, export_yolo

root = Path("artifacts/my-experiment")
generate_dataset(root / "dataset", sessions=12, frames_per_session=8, seed=42)
trained = train_yolo(
    root / "dataset",
    root / "training",
    Path("configs/tiny-yolo.yaml"),
    epochs=3,
    image_size=320,
    batch=8,
    seed=42,
)
exported = export_yolo(
    Path(trained["weights"]),
    root / "export",
    image_size=320,
    scope="synthetic_fixture",
)
```

所有输出目录必须事先不存在，失败不会覆盖此前实验。训练产物包括 `run/weights/best.pt`、模型配置副本、`input_snapshot` 数据副本、训练所用图片列表、Ultralytics 的训练记录及 `training.json`。导出产物包括 `model.onnx`、`manifest.json`、`export.json`。部署目录不重复保存 `.pt` 权重。

`training.json` 保存数据集 manifest SHA-256、模型 YAML SHA-256、权重 SHA-256、种子、训练参数、依赖版本、验证集指标和实际墙上耗时。导出报告保存输入/输出张量形状、ONNX 校验和、opset 和实测推理结果。

输入数据先校验，再把 manifest 中登记的图片和标签复制到本次运行的 `input_snapshot`，并再次核对实际复制字节。图片列表只引用这份副本。原始 `data.yaml` 中的下载代码或未登记的旁路图片不会被训练入口采用。Ultralytics 即使设置 `cache=False` 仍会写标签缓存，因此缓存只允许出现在本次私有工作目录；训练结束清理这些缓存并重新验证快照，原始数据集不被修改。

图片列表和上游运行参数记录训练时的本地绝对路径；发布后这些临时路径不能直接复用。`input_snapshot` 本身保持完整且可验证，再次训练时把它作为 `train_yolo` 的 dataset 输入并选择新的 output，封装会重新生成正确路径。

## 训练配置

`configs/tiny-yolo.yaml` 是本项目自写的小型 Conv/C2f/SPPF/Detect 结构，7 个类别，双尺度检测头。它用于快速验证接口、训练、导出、推理一致性，并不声称是最佳架构。Ultralytics 是可选依赖，其自身许可适用于该依赖与相关使用。

当前入口接受局部、显式的检测模型 YAML，只允许列出的基础模块和有限配置键；不接受模型昵称、远程路径、预训练下载或自定义 activation 表达式。默认从头训练，`pretrained=False`。

CPU 固定为 2 个 PyTorch 工作线程，DataLoader `workers=0`、`amp=False`、`compile=False`。训练入口设置种子和 deterministic 模式，但不同 PyTorch 版本或 CPU 平台仍可能产生数值差异；报告记录环境版本，未承诺跨平台权重逐字节相同。

初始配置关闭透视、旋转、剪切、翻转、Mosaic、MixUp、Copy-Paste 和颜色扰动，便于建立像素场景的可解释基线。后续增强实验应逐项添加，并用独立测试集比较。

参数限制：epochs 1–200、image_size 64–1280 且为 32 的倍数、batch 1–64；batch × image_size² 受额外输入规模限制。训练可能耗时，需根据本机实际硬件选择参数。

## 无隐式网络依赖

入口只加载已经存在的本地 YAML、数据集和 `.pt` 文件；缺少依赖时明确失败，不在运行中安装软件或下载权重。运行期间设置 Ultralytics 离线和禁止自动安装配置，关闭分析和第三方训练平台回调，避免修改用户持久化设置，并暂时禁止 Python socket 连接。

安装依赖本身需提前完成。运行时的 socket 限制是用于避免库的意外网络请求的协作式进程保护，不是执行不可信代码的安全沙箱。建议在独立 CLI 进程运行训练；它暂时修改进程内库设置，退出时恢复。`.pt` 应来自可信的本地训练；封装启用已验证版本支持的受限加载器，但不将模型加载视为任意文件安全执行机制。

## ONNX 契约

导出参数显式设置 static input、CPU、batch=1、opset=17、`simplify=False`、`nms=False`。随后使用真正导出模型进行 ONNX checker 检查和 ONNX Runtime 推理，拒绝非有限输出，依据实际张量形状写入 `raw_xywh` 或 `end2end_xyxy`，不靠文件名猜测。

`manifest.json` 与 `onnx_detector.ModelManifest` 完全一致，包含 schema_version、model_sha256、class_names、input_width、input_height、output_format 和 scope。推理适配器会验证模型哈希再加载。训练/导出功能不自动把模型替换为默认检测器，也不把合成数据的模型标记为真实录像已验证。

## 验证证据与再执行

初次本地工程 smoke 使用 6 会话 × 4 帧，共 24 张合成图、240 个标注；以 image_size=160、batch=4、seed=42 训练 1 epoch，实际耗时约 8.07 秒。它成功产生 ONNX 输入 `[1,3,160,160]`、输出 `[1,11,2000]`，其中 11=4 个框参数+7 个类别得分，ONNX Runtime 推理成功。**该次仅 1 epoch 的验证集指标为 0**，因此它是工程 smoke，不能当作模型验收成绩。

重跑单独的真实训练/导出集成测试：

```bash
MAPLE_TRAINING_SMOKE=1 python -m pytest -q tests/test_training.py
```

Windows PowerShell：

```powershell
$env:MAPLE_TRAINING_SMOKE = "1"
python -m pytest -q tests/test_training.py
```

常规 CI 可以只运行依赖轻量的参数、来源和隔离边界测试；真实训练测试显式启用，避免默认安装整套深度学习环境。

参考上游接口：[Ultralytics Train](https://docs.ultralytics.com/modes/train/) 与 [Ultralytics Export](https://docs.ultralytics.com/modes/export/)。本次实际执行依赖为 Ultralytics 8.4.218、PyTorch 2.15.0+cpu、ONNX 1.23.2、ONNX Runtime 1.32.0；这些是本次证据，不代表未来任意版本都已经验收。
