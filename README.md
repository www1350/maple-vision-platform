# Maple Vision Platform

多职业、多地图视觉研究平台的工程基础。当前交付范围是 **P0**：可安装 Python 包、明确的数据协议、完整 NoOp 管线、本地 SQLite 初始化、命令行和自动验收。

## 当前可以运行什么

```text
DemoSource → Frame → Observation → WorldState → Intent
          → NoOpController → Verification → JSONL
```

`DemoSource` 生成带序号和时间戳的测试帧。`NoOpPerception` 明确返回 `UNKNOWN`；控制器返回 `SKIPPED`；验证器返回 `NOT_APPLICABLE`。这证明工程管线能运行，并不表示游戏识别、任务执行或模拟器物理已经实现。

| 已实现 | 后续阶段 |
|---|---|
| 不可变数据协议、时序与 ID 校验 | YOLO/OpenCV/OCR 与游戏状态 |
| 完整 NoOp 单步管线 | 战斗、技能、拾取、吃药、导航 |
| CLI、TOML 配置、JSONL 记录 | 真实录像解码与视觉模拟器 |
| SQLite v1 迁移及只读检查 | 小册子同步、完整游戏知识表 |
| 知识、素材、Detector、Simulator 抽象接口 | 具体适配器、素材导入、模型训练 |
| Linux/Windows CI、隔离 wheel 安装检查 | 多职业与多地图验收 |

## 安装（Python 3.12）

在项目根目录创建虚拟环境。Linux/macOS：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-build-isolation -e .
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.venv\Scripts\python.exe -m maple_vision --help
```

下面命令中的 `python` 指虚拟环境的解释器。安装后也可用 `maple-vision` 控制台入口。P0 的运行时代码只使用 Python 标准库，不需要 GPU、模型文件、网络服务或 LLM 凭据；开发工具安装需要包源。

## 三帧演示与日志

```bash
python -m maple_vision demo --frames 3 --output demo.jsonl
```

标准输出为 JSON 摘要，四项计数 `frames_processed`、`intents_emitted`、`executions_skipped`、`verifications_not_applicable` 都应为 3。输出文件每帧一行，包含观测、状态、意图、执行回执和验证结果，另附 `schema_version` 与 `run_id`。

日志文件采用独占创建：已有文件不会被覆盖。重跑时使用新的文件名。父目录必须存在；运行中发生错误时可能保留已经写出的部分轨迹，程序以非零状态退出。日志不包含原始像素。

配置示例：

```bash
python -m maple_vision demo --config configs/demo.toml --output configured-demo.jsonl
```

`--frames` 可覆盖配置中的帧数。配置版本、模式、拼错的字段、非整数或越界值都会被拒绝。P0 只支持 `mode = "demo"`。

## SQLite 基础

```bash
python -m maple_vision db init knowledge.db
python -m maple_vision db check knowledge.db
```

`init` 创建或校验 v1 `source_records` 表，可以重复运行并保留已有数据。`check` 通过只读连接检查，不创建文件、不迁移、不写入。遇到未来版本或不兼容结构会返回错误。

该数据库目前只有来源记录基础表，不包含已爬取的小册子资料。真实数据写入在 P1 实施。

## 本地验收

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -q
python -m build --wheel --no-isolation
```

CI 同时在 Ubuntu 和 Windows 的 Python 3.12 上执行这些检查，并在新的虚拟环境中安装 wheel、离开源码目录运行 CLI，以避免源码路径掩盖打包问题。开发工具版本固定在 `requirements-dev.txt`。

## 文档与开发边界

- [完整方案](MASTER_SPEC.md)
- [P0 接口与数据语义](docs/API_CONTRACTS.md)
- [P0 验收报告](docs/P0_ACCEPTANCE.md)
- [实施计划](docs/IMPLEMENTATION_PLAN.md)
- [P0 机器可读验收合同](engineering/p0-contract.yaml)
- [后续阶段依赖](engineering/milestones.yaml)
- [Coding AI 规范](AGENTS.md)

保留原 `run_once(source, perception)` 观测接口；新增 `Pipeline.tick()` 完整管线。P0 验收后按 `AGENTS.md` 停止，后续从 P1 开始。

执行研究以离线素材和独立模拟器为范围。当前没有系统输入、游戏客户端控制或反作弊规避后端，也不以本地测试推断正式账号安全。
