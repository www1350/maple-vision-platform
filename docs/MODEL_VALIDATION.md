# 模型套件、资源采样与完整重新推理验证

本模块扩展 v0.1 的离线验证能力：同一个显式提供的检测器可参与场景套件和墙钟长稳；新的重新推理验证会实际执行感知、监督和策略，并比较完整观测与决策。运行对象仍是项目自建的合成模拟器，模型成绩不代表真实游戏画面或正式规则已经验收。

## 1. 三种验证的区别

| 入口 | 实际执行 | 可以得出的结论 |
|---|---|---|
| `verify_run(path)` / `lab replay` | 从初始状态执行记录中的实际动作，复核状态、事件、采样像素和任务目标 | 动作与模拟环境证据一致；固定返回 `perception_reproduced=false` |
| `verify_perception(path, detector=None)` / `lab reinfer` | 先做动作重放，再从初始状态重新运行每一帧感知、Supervisor 和 VisualAgent，执行新算出的动作并比较 | 在声明的数值容差下，观测、监督与策略输出实际复现 |
| `run_suite` / `run_soak` | 使用指定检测器新执行任务或故障场景 | 对本次环境、模型、种子和场景的实际任务结果；套件通过并不要求所有故障任务完成 |

重新推理不会把 trace 中的动作喂给新策略。新动作来自本次重新计算的图像观测；记录仅用作逐步比较的预期值。初始模拟状态只进入模拟器，动态真值不进入感知或策略。动作重放仍负责独立环境结果检查。

## 2. 稳定 Python API

```python
from pathlib import Path

from maple_vision.lab import run_soak, run_suite, verify_perception
from maple_vision.onnx_detector import OnnxDetector

directory = Path("examples/fixture-model")
detector = OnnxDetector(directory / "model.onnx", directory / "manifest.json")

suite = run_suite(Path("artifacts/model-suite"), seeds=3, detector=detector)
soak = run_soak(
    Path("artifacts/model-soak"),
    seconds=60,
    detector=detector,
    resource_interval_seconds=1.0,
    max_resource_samples=1024,
)
verification = verify_perception(Path("artifacts/model-run"), detector=detector)
```

签名：

```python
run_suite(output: Path, seeds: int = 3, detector: Detector | None = None) -> dict
run_soak(
    output: Path,
    seconds: int = 60,
    detector: Detector | None = None,
    *,
    resource_interval_seconds: float = 1.0,
    max_resource_samples: int = 1024,
) -> dict
verify_perception(output: Path, detector: Detector | None = None) -> dict
```

省略 detector 时保持默认像素基线。指定 ONNX 时使用与 `run_episode` 相同的混合管线：动态对象来自模型，平台、梯子和 HUD 来自合成场景像素解析器。模型实例在各 episode 间复用；OnnxDetector 逐帧无策略状态，Supervisor、VisualAgent、模拟器和故障适配器每轮重新创建。

suite/soak 保留三个内置职业的矩阵，不增加任意职业或外部地图的隐含支持。单个运行已有的职业参数和 SpritePack 则完整参与重新推理验证。

CLI 示例：

```bash
python -m maple_vision lab run --output artifacts/model-run --model examples/fixture-model/model.onnx --manifest examples/fixture-model/manifest.json
python -m maple_vision lab reinfer artifacts/model-run --model examples/fixture-model/model.onnx --manifest examples/fixture-model/manifest.json
python -m maple_vision lab suite --output artifacts/model-suite --seeds 3 --model examples/fixture-model/model.onnx --manifest examples/fixture-model/manifest.json
python -m maple_vision lab soak --output artifacts/model-soak --seconds 60 --resource-interval 1 --max-resource-samples 1024 --model examples/fixture-model/model.onnx --manifest examples/fixture-model/manifest.json
```

## 3. 新运行包的感知证据

既有 manifest、动作 trace 和动作重放合同保留。新 `summary.json` 增加 `perception_evidence`，其独立 `schema_version=1` 记录：

- 明确的像素或混合管线名称，以及检测器实现名。
- 完整模型 manifest、实际加载字节的 SHA-256、显式 confidence 与 IoU 阈值，包括默认值。
- 最后一帧观测。
- 终止原因、终止时的 Supervisor 结果，以及本轮实际产生的终止策略决策。

步数预算耗尽时，最后一帧已被感知，但预算以外的 Supervisor 或策略没有执行，因此对应终止字段为 `null`。监督停止时保留实际 guard，策略决策为 `null`；策略完成或失败时两者均保留。这些区别在验证时逐项比较，不能补造没有实际执行的最后决策。

旧 v0.1 包没有保存终止观测与模型阈值，不能从现存动作推导它们。它们仍可执行 `verify_run`，但 `verify_perception` 会明确拒绝缺少版本化证据的旧包并要求重新生成。验证器不修改、迁移或覆盖原证据包。

## 4. 完整重新推理如何核验

1. 先绑定原 manifest 字节，再检查文件哈希和动作重放；预检前后 manifest 必须一致，后续路径再次做目录范围校验。动作重放在执行超出配置预算的下一行前拒绝，不先执行整段超预算轨迹。核对初始状态、配置、嵌入职业参数和素材、最终状态、图像及独立任务目标。
2. 在第一次 `detect()` 前验证模型身份。必须提供标准 `OnnxDetector`，其只读 `loaded_model_sha256` 来自传给原生会话的同一份已校验字节。这个哈希、完整 manifest、阈值和管线均须与记录一致。省略模型只能验证记录为默认像素管线的运行。
3. 重新恢复同一个初始模拟状态和可选 ClassSpec/SpritePack，按记录的 RunConfig 创建全新监督与策略，重建相同故障序列。
4. 感知初始帧；每次执行新策略算出的动作后，再感知下一帧。每个执行步骤比较帧像素哈希、完整观测、监督结果、策略决策、实际动作/事件、视觉效果结论和模拟状态哈希。
5. 比较终止观测、实际终止 guard/decision，以及任务状态、理由、恢复/暂停计数等语义摘要。记录跳步、缺帧、多余步骤、提前终止或错误预算均会失败。
6. 再次核对原 manifest 与文件哈希，确认验证期间证据内容未变化，才返回成功报告。

一个执行了 N 步的运行会调用感知 N+1 次，最后一帧也包含在内。重复帧故障保留旧的帧身份，但每次收到的重复画面仍实际经过感知；不利用缓存把它记为新推理。

无法证明身份的自定义 Detector 或 OnnxDetector 子类可以用于试验性的运行/套件，但不能获得本入口的模型复现结论。任意可执行 Python 本身仍属于受信实现，这不是抵御恶意修改进程的权限沙箱。

## 5. 数值容差与身份限制

观测中的浮点测量使用 `math.isclose(rel_tol=1e-6, abs_tol=1e-6)`：包括检测框、检测分数和血蓝比例。整数帧序号、时间、类别、检测顺序、已知/未知状态、来源、地图、对话标记，以及所有监督与策略决策仍精确比较。缺失目标和多余目标不能被容差掩盖。

浮点漂移导致目标筛选、NMS 顺序、动作选择或停止理由改变时，验证失败；容差不会让不同决策通过。报告明确列出容差，不称为浮点逐字节相同。

模型字节/manifest/阈值不匹配直接失败。Python、依赖版本或源码身份不同则在 `runtime_identity_matches` 中明确报告，同时保留记录时和验证时的身份。即使这些身份不同，只有实际重新计算并匹配的输出才能得到 `perception_reproduced=true`；这表示当前实现复现了记录输出，不能宣称原硬件或原依赖环境完全重建。报告也不把新用时等同于旧用时。

文件哈希提供完整性，不是第三方签名或来源真实性证明。验证一个失败任务会返回该失败任务确实复现，不能把 `verified=true` 解读成任务完成。

## 6. 墙钟与资源合同

soak 使用 `time.monotonic()` 计算实际经过时间，模拟时长单独记录。每轮结束写 checkpoint，失败时停止。运行可能超过请求时长一个 episode，不中断正在执行的本轮模型调用。

套件中的正常任务和 soak 的每一轮都**只执行一次**：在该次新输出目录下创建 `active-episode`，保存完整动作/观测/终止证据和每 16 步一张 PNG。成功后只清理本轮创建的 staging 包；失败则将同一份包移动为 `failure-*`，不重新运行检测器制造失败证据。四种指定故障场景的原始包始终保留。已有目录会被拒绝，不清理预先存在的内容；同时最多保留一个活动 episode，外加要求保留的失败与故障包。

`episode_capture` 明确记录单次执行、截图步长、完整 trace、保留策略及采集 I/O 开销。suite 的 `actual_wall_seconds` 和 soak 的实际墙钟/资源包含截图、JSONL、GIF 写入及本轮清理成本；suite 的 `episode_wall_seconds` 是各轮核心执行循环计时之和，不能冒充包括全部 I/O 的总时间。这里不是关闭证据采集后的纯模型速度基准。

异常退出时仍保存失败摘要和已取得的资源记录；若异常发生在 episode 写完整包之前，保留 `active-episode` 并明确标为 `partial_episode_bundle`，不把不完整目录称为已验证的完整重放包。

资源采样由独立的轻量线程完成，不受“一轮模型任务很长，只能在轮次之间看一次”的限制。默认请求每秒一次，间隔允许 0.05–60 秒；保留上限允许 2–10000 条、默认 1024。操作系统调度可能延迟采样，因此记录实际时刻，间隔不是实时保证。

| 字段 | 来源与单位 | 解释 |
|---|---|---|
| `elapsed_wall_seconds` | `time.monotonic`，秒 | 从采样器启动开始的实际经过时间 |
| `process_cpu_seconds` | `time.process_time`，秒 | 本进程所有线程的累计 CPU 增量，不含子进程 |
| `cpu_percent_since_previous` | CPU 增量/实测墙钟增量×100 | 单个逻辑 CPU 为 100%；多线程可以超过 100%，不是全机归一化百分比 |
| `current_rss_bytes` | Linux `/proc/self/statm` 驻留页数×`SC_PAGE_SIZE`，字节 | 当前进程驻留内存；不以 Python 分配器数据替代 |
| `lifetime_peak_rss_bytes` | Linux/macOS `getrusage(RUSAGE_SELF).ru_maxrss`，转换为字节 | 整个进程生命周期峰值，可能在本次 soak 前发生 |

当前 Windows 没有实现 RSS 读取；macOS 仅有生命周期峰值，当前 RSS 保持未知。不可读取或未支持的计数为 `null`，并附 source 与原因。未知内存不是 0，也不能代替“没有内存增长”的结论。

保留条数超限时，对旧时间点做确定性抽稀，保留首点与最新点；累计测量数、丢弃数和保留数分别记录。最大实测当前 RSS 和 CPU 比例包含已被抽稀的测量，生命周期峰值继续独立存放。采样仍可能漏过短暂峰值，不能把“最大采样值”称为精确历史峰值。

这些资源值包含进程内渲染、策略、验证、采样和其他线程的工作，不只包含模型。其他进程并发训练会影响实际用时与调度，但不计入本进程 CPU/RSS。比较性能时须单独固定或记录并发负载；本次开发期间存在并发训练，不将其耗时用作空闲机器基准。

只有请求时长至少 86400 秒、实际墙钟达到 86400 秒且任务检查通过时，才可标记 24 小时门禁通过。60 秒请求、加速模拟、异常退出或一次异常长的短请求都不会得到该标记。

## 7. 回归与可复现检查

`tests/test_model_validation.py` 覆盖：

- 像素完整任务、四类故障及终止帧的实际重新执行。
- 观测/监督/策略/效果篡改，重新计算外层哈希仍不能掩盖语义差异。
- 错误预算、漏帧、旧证据、未知检测器，以及有容差测量与精确身份的边界。
- 原数据库与注册素材删除后的嵌入规则/画面重建。
- 模型套件/长稳参数接线、失败包、异常状态和资源来源/单位/有界保留。
- 瞬时检测失败后下一轮可恢复的反例，证明保留的是原始失败而非第二次执行；预先存在的 staging 内容不被删除。
- 随仓库交付的实际 ONNX 完整任务；逐次统计 `detect()`，验证所有 N+1 帧确实推理且原包未改写。
- 模型哈希、完整 manifest 或阈值不匹配时，第一次 detect 前拒绝。

```bash
ORT_DISABLE_TELEMETRY=1 python -m pytest tests/test_model_validation.py tests/test_lab.py -q
```

原生 ONNX 依赖缺失时，对应原生测试明确跳过，不能记为模型通过。完整发布成绩仍以具体提交的测试和 CI 记录为准。本专题不授予真实域、通用模型推广或 24 小时长稳验收。
