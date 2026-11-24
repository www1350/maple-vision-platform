# Maple Vision Platform v0.4.0

一个可在本机运行、训练和验证的**离线视觉研究平台与独立模拟器**。它包含本地知识目录、素材登记、合成数据集、YOLO 训练与 ONNX 推理、多职业/多地图任务，以及可重放的验证证据。

**完整方案入口：[完整方案与 AI 落地指南](docs/FINAL_PLAN.md)**。其中集中说明架构、数据流、从零实施顺序、各阶段完成标准，以及下一位编程 AI 接手时应遵守的接口与验证要求。

**v0.4 新增外部地图包。** 同一份严格校验的 JSON 可用于模拟、渲染、三职业任务、生命周期数据和独立重放；示例已扩展为 `grove → quarry → ruins` 三图。地图内容、来源与摘要随数据和运行保存，删除原文件后可以复验。固定画布下支持 2–8 图单出口循环路线，地图格式有效不等于任意布局都能完成。[地图包使用](docs/MAP_PACKS.md) · [本轮执行方案](docs/V0_4_PLAN.md) · [本轮实际验收](docs/RELEASE_0_4_ACCEPTANCE.md)

**v0.3 工具工程验收已通过，地形实验已完成并保留失败候选。** 新增训练前的检测头/锚点检查、全宽地板召回与边缘误差、显式模型地形模式，以及训练前冻结的单候选实验。候选虽检出192/192条整幅地板，但总平台召回只有81.94%，未达到98%门槛，已明确拒绝推广；test保持未评分。本地完整回归 **834 项通过、0 跳过**，隔离安装包验证也已通过；日志流生命周期问题已修复；实现提交 `3340d8f` 的[Ubuntu、Windows及原生CI](https://github.com/www1350/maple-vision-platform/actions/runs/45421893081)全部通过，最终归档检查与合并记录见 [PR #4](https://github.com/www1350/maple-vision-platform/pull/4)。完整状态见 [v0.3 发布验收](docs/RELEASE_0_3_ACCEPTANCE.md) 和 [执行方案](docs/V0_3_PLAN.md)。

v0.2 已验收的模型继续保存在 `examples/lifecycle-model/`，其范围是 YOLO 动态对象加像素地形/HUD；生命周期数据、知识增量计划和实际感知/策略重新执行也继续兼容。[v0.2 历史验收](docs/RELEASE_0_2_ACCEPTANCE.md) · [PR #3](https://github.com/www1350/maple-vision-platform/pull/3) · [v0.1 历史验收](docs/RELEASE_ACCEPTANCE.md)。

默认任务使用可解释的像素颜色基线，便于独立检查工程闭环。可选的 tiny YOLO 从本地配置随机初始化训练；训练完成、导出成功、检测效果、任务完成是四项不同的验收，不会互相冒充。当前规则和画面都是项目自行构造的 `synthetic_fixture`，不代表已经核实的国服数据或正式游戏操作。

## 先跑一个完整任务

需要 Python 3.12。在仓库根目录安装固定版本的开发环境；日常运行不需要 GPU、Redis、云服务或 LLM API。

**Linux：**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m maple_vision lab run --output artifacts/first-run --job assassin --seed 0
.venv/bin/python -m maple_vision lab replay artifacts/first-run
```

**Windows PowerShell：**

```powershell
py -3.12 -m venv .venv
$maplePython = ".\.venv\Scripts\python.exe"
& $maplePython -m pip install -r requirements-dev.txt
& $maplePython -m pip install --no-build-isolation --no-deps -e .
& $maplePython -m maple_vision lab run --output artifacts/first-run --job assassin --seed 0
& $maplePython -m maple_vision lab replay artifacts/first-run
```

运行目录包含 JSON 摘要、完整动作轨迹、初始/最终状态、采样 PNG、预览 GIF 和哈希清单。正常任务依次完成两张地图的战斗、拾取、NPC 对话与切图；独立验证器检查实际环境结果。输出目录必须是新路径，重跑请换名。

更多命令见 [快速开始](docs/QUICKSTART.md)。安装包源需要网络；上述模拟器、知识查询、素材处理与验证流程在依赖安装后本地运行，**不消耗 LLM token**。

## 加载三张外部地图

```bash
python -m maple_vision maps check examples/maps/three-map-fixture.json
python -m maple_vision lab run --map-pack examples/maps/three-map-fixture.json --output artifacts/three-map-run
python -m maple_vision lab replay artifacts/three-map-run
python -m maple_vision lab reinfer artifacts/three-map-run
python scripts/verify_external_maps.py --output artifacts/three-map-acceptance
```

最后一条独立验收包含三职业、13 个正常/故障场景、486 帧数据与像素 val 分组、实际 60 秒短时运行，保留完整报告和原始证据。它不训练或推广学习模型；新地图上的 YOLO 准确率需要另外取得证据。省略 `--map-pack` 时继续使用原有两张地图。

## 已实现的能力

| 模块 | 当前实现 |
|---|---|
| 知识目录 | SQLite v2，版本与来源追溯、冲突保留、快照验证；本地导出可生成自包含差异计划，在同一写事务中核对基线、导入和结果；已核实的合成职业参数可注入运行 |
| 素材登记与渲染 | 本地授权 PNG/JPEG/WEBP 解码、内容寻址、哈希与来源记录；显式 SpritePack 接入渲染、数据集和可重放任务 |
| 地图与数据集 | 两张内置地图继续兼容，外部 MapPack 支持 2–8 图循环；lifecycle schema3 完整嵌入地图并复核每图六阶段、动作、像素、标签和覆盖；旧数据 v1/v2 保留 |
| 模拟器 | 固定步长、确定性种子、保存恢复、平台与梯子、单段跳跃、战斗/冷却/Buff/资源/拾取/药品/NPC/切图 |
| 职业样例 | 标飞的弹药与跳标、龙骑的近身/范围技能、牧师的治疗与 MP 差异 |
| 视觉 | 像素基线、哈希绑定的 ONNX；分组指标及全宽地板召回/边缘误差；显式选择模型地形时七类对象全部来自检测器，HUD仍是合成像素，策略只接收观测 |
| YOLO 工具 | 本地配置、真实 CPU 训练、独立输入副本与 ONNX；训练前核对实际 DFL/stride/正样本候选和四边范围，分别测结构与实际部署成本 |
| 任务与监督 | 有界任务与恢复、观测新鲜度、重复帧/时钟回退/未知状态处理、独立目标核验 |
| 验证实验 | 短暂/持续黑屏、重复帧、输入冻结；指定模型参与 suite/soak，采样进程资源；reinfer 实际重新运行每帧感知、监督与策略；预检/文件变化与保留原始失败轮次已通过本地回归 |
| 本地录像与 OCR | PNG 帧目录/MP4 导入、完整性检查与感知回放；通过已安装的 Tesseract 识别文本 |
| 回话 | 模拟器 NPC 对话；人工确认的本地玩家回复草稿，不含发送后端 |

知识目录中的三个现有职业可通过 `--knowledge-db` 与 `--game-version` 显式加载完整 `ClassSpec`。加载器要求所有候选均为 VERIFIED、无冲突、`synthetic_fixture`；职业参数、哈希与来源进入运行证据，重放无需原数据库。地图布局可以通过 `--map-pack` 注入，通用药品/Buff 等规则仍为样例常量。注册素材可通过成对的 `--asset-registry` 与 `--sprite-map` 参数接入渲染、数据集和任务，运行包内嵌素材后可独立重放。

本地录像已有导入、完整性验证和感知回放合同；固定录像不提供反事实动作验证。真实素材的识别效果、真实规则、新职业，以及超出当前地图 schema 的内容仍需各自的适配与验收。

## 地形研究与模型对象模式

`--terrain-source model` 可用于 `lab run/suite/soak/reinfer` 和 `vision observe`，必须提供本地 `--model` 与 `--manifest`。此模式不会补回漏检地形；所有对象来自模型，HP/MP、地图和对话标记仍来自 fixture UI。省略开关时，旧像素/混合模式行为和运行包身份保持兼容。[完整模式合同](docs/MODEL_TERRAIN.md)

模型应先做结构预检和独立检测评估，再进入任务验证。下面使用自行准备的本地数据与导出模型，输出路径必须不存在：

```bash
python -m maple_vision model audit-terrain artifacts/terrain-data --config configs/terrain-yolo.yaml --output artifacts/terrain-audit.json --split val --iterations 0
python -m maple_vision vision benchmark artifacts/terrain-data --split val --model local-model/model.onnx --manifest local-model/manifest.json --terrain-metrics --output artifacts/terrain-validation.json
python -m maple_vision lab run --output artifacts/model-terrain-run --model local-model/model.onnx --manifest local-model/manifest.json --terrain-source model
python -m maple_vision lab reinfer artifacts/model-terrain-run --model local-model/model.onnx --manifest local-model/manifest.json --terrain-source model
```

`audit-terrain` 需要训练依赖，`--iterations 0` 省略随机权重延迟测量；它只读取 train/val 几何做结构分析。`--terrain-metrics` 的 schema3 报告复用同批预测，分别报告漏检和匹配框误差；“通过 IoU50”不代表地板端点已经准确。[指标定义与反例](docs/TERRAIN_EVALUATION.md) · [本轮冻结门禁](reports/release-0.3.0/preregistration.json)

本轮失败候选以 [examples/terrain-candidate](examples/terrain-candidate/) 单独保留，包含实际 ONNX、manifest 和复现方法。标准库脚本可独立核对保存的报告：

```bash
python scripts/check_terrain_experiment.py --preregistration reports/release-0.3.0/preregistration.json --candidate reports/release-0.3.0/terrain-model-validation.json --baseline reports/release-0.3.0/lifecycle-baseline-validation.json --output artifacts/rechecked-terrain-gates.json
```

预期退出码为 **1**，表示有效报告有质量门禁未通过，不是脚本崩溃。14项中仅 `platform_recall` 未达标；模型和原始结果保留，未改标签、阈值或及格线，也未追加第二次候选训练。[实际决定](reports/release-0.3.0/model-decision.json)

## 生命周期数据与知识增量

在已安装环境中，使用全新的输出路径生成生命周期数据，并计划/应用一份合成知识导出：

```bash
python -m maple_vision dataset create artifacts/lifecycle-data --profile lifecycle --sessions 18 --frames 48 --seed 20261125
python -m maple_vision dataset check artifacts/lifecycle-data
python -m maple_vision knowledge init artifacts/sync-knowledge.sqlite
python -m maple_vision knowledge plan-sync artifacts/sync-knowledge.sqlite fixtures/knowledge-demo.json artifacts/knowledge-plan.json
python -m maple_vision knowledge apply-sync artifacts/sync-knowledge.sqlite artifacts/knowledge-plan.json
```

知识计划捕获源文件原字节和完整目录基线；查看计划后应用的是同一份证据。源文件变化需要新建计划，数据库变化会使旧计划失败；缺失条目保留、冲突不自动解除、未知不自动核实。当前只实现本地导出适配器，没有自动获取最新小册子资料。[知识同步合同](docs/KNOWLEDGE_SYNC.md)

显式使用同一 ONNX 与 manifest 完成运行、重新推理验证和模型测试。下面使用已通过本地混合管线门禁的 v0.2 lifecycle 模型；旧模型继续保存在 `examples/fixture-model/`：

```bash
python -m maple_vision lab run --output artifacts/v02-model-run --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
python -m maple_vision lab reinfer artifacts/v02-model-run --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
python -m maple_vision lab suite --output artifacts/v02-model-suite --seeds 3 --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
python -m maple_vision lab soak --output artifacts/v02-model-soak --seconds 60 --resource-interval 1 --max-resource-samples 1024 --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
```

`lab replay` 保持动作与环境重放语义；`lab reinfer` 额外执行真实感知与策略，只支持包含新版证据的运行包，旧包继续使用 replay。预检/文件变化边界和保留实际失败轮次一次已有本地审查与回归证据；实现和限制见 [模型验证](docs/MODEL_VALIDATION.md)。源代码、模型及测试提交 `f17e7487f5938acfaf312e05b8eecb242f06a492` 的[跨平台 CI](https://github.com/www1350/maple-vision-platform/actions/runs/45394348732)已通过；后续文档归档提交在 [PR #3](https://github.com/www1350/maple-vision-platform/pull/3) 执行同一套检查。60 秒长稳是短时验证，不是 24 小时验收。

## 模型与发布证据

- [v0.4 发布验收记录](docs/RELEASE_0_4_ACCEPTANCE.md) · [v0.4 执行计划](docs/V0_4_PLAN.md)：外部地图、兼容、独立重放、安装与准确提交 CI。
- [v0.3 发布验收记录](docs/RELEASE_0_3_ACCEPTANCE.md) · [v0.3 有界实验计划](docs/V0_3_PLAN.md)：已完成的历史研究与拒绝候选，推广分支保持关闭。
- [v0.3 地形实验与漏检诊断](docs/YOLO_TERRAIN_EXPERIMENT.md)：同集比较、全部104个遗漏的逐真值账本、原始检测框检查与实际对照图。
- [v0.2 发布验收记录](docs/RELEASE_0_2_ACCEPTANCE.md)：上一版本准确提交、测试、操作系统和实测结果。
- [v0.2 lifecycle 模型实验](docs/YOLO_LIFECYCLE_EXPERIMENT.md)：选模时序、检测指标、三职业实际重新推理、短时长稳与架构限制。
- [v0.1 发布验收](docs/RELEASE_ACCEPTANCE.md) · [v0.1 YOLO 实验](docs/YOLO_EXPERIMENT.md)：历史模型、原数据划分和原环境结果。
- [生命周期数据与分组评估](docs/DATASET_LIFECYCLE.md)：v0.2 数据合同、冻结输入及验证范围。
- [阶段状态与后续工作](docs/IMPLEMENTATION_PLAN.md)：区分已实现、已验证与需要外部素材的项目。

v0.2 模型在冻结后的保留 test 上测得 AP50 **0.944822**、micro precision **0.997818**、recall **0.872774**：192 帧、1572 个真值目标，TP/FP/FN 为 **1372/3/200**。三职业任务和实际 reinfer 全部通过，模型 suite 的 **13/13** 场景符合预期。60 秒请求实际运行 **61.510525 秒**、完成 **10 轮**，记录 **63** 个资源样本，最大采样当前 RSS 为 **90.09375 MiB**；24 小时门禁未完成。指标和运行证据见[本轮实验](docs/YOLO_LIFECYCLE_EXPERIMENT.md)。

上述 v0.2 模型的地形验收未通过：其实际 `reg_max=8`、stride4/8，单框最大宽度 112px，无法在 IoU=0.5 下匹配 320px 全宽地板。它仍按原混合管线使用；v0.3 新结构与新模式的实际结果单独验收。原模型和报告保留，默认像素基线继续可用。所有成绩来自自绘合成场景，没有真实游戏识别效果结论；工具不会按模型名自动下载权重。

```bash
python -m maple_vision lab run --output artifacts/shipped-model-run --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
```

## 开发与自动验收

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -q
python -m build --wheel --no-isolation
python scripts/wheel_smoke.py --wheel dist/maple_vision_platform-0.4.0-py3-none-any.whl
```

这里的 `python` 应为上面虚拟环境的解释器。CI 配置同时检查 Ubuntu、Windows 的 Python 3.12，并在仓库之外新建仅含基础运行依赖的环境验证 wheel；另设 Ubuntu CPU 训练、ONNX 导出与推理检查。CI 成绩及版本见验收记录，不能从工作流文件存在推断已经通过。

依赖分为 `requirements-runtime.txt`、`requirements-dev.txt` 与 `requirements-training.txt`。训练环境的安装和可选测试开关见快速开始。已有 P0 的 `demo`、`db`、`run_once` 与 NoOp 合同保持兼容。

## 文档导航与范围

- [整体设计](MASTER_SPEC.md)
- [v0.3 执行计划](docs/V0_3_PLAN.md) · [v0.3 机器可读门禁](engineering/v03-contract.yaml)
- [模型地形模式](docs/MODEL_TERRAIN.md) · [地形专项评估](docs/TERRAIN_EVALUATION.md)
- [v0.2 执行计划](docs/V0_2_PLAN.md) · [v0.2 机器可读门禁](engineering/v02-contract.yaml)
- [接口合同](docs/API_CONTRACTS.md) · [数据与来源](docs/CATALOG.md)
- [原子知识增量计划](docs/KNOWLEDGE_SYNC.md) · [生命周期数据](docs/DATASET_LIFECYCLE.md) · [模型套件与重新推理](docs/MODEL_VALIDATION.md)
- [知识目录到运行职业参数](docs/CLASS_PROFILES.md)
- [注册素材接入渲染与数据集](docs/SPRITE_PACK.md) · [本地录像与 OCR](docs/RECORDINGS.md)
- [模拟器实际行为](docs/SIMULATOR_IMPLEMENTATION.md)
- [P0 历史验收](docs/P0_ACCEPTANCE.md) · [开发约束](AGENTS.md)

项目不包含正式客户端控制、系统按键注入、封包/内存修改或反作弊绕过。自建样例测试不能保证真实账号不被封禁。尚未核实的游戏素材、规则、真实录像泛化和完整 24 小时墙钟长稳，均须在验收记录中单独说明。
