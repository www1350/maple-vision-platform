# Maple Vision Platform v0.1

一个可在本机运行、训练和验证的**离线视觉研究平台与独立模拟器**。它包含本地知识目录、素材登记、合成数据集、YOLO 训练与 ONNX 推理、多职业/多地图任务，以及可重放的验证证据。

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

## 已实现的能力

| 模块 | 当前实现 |
|---|---|
| 知识目录 | SQLite v2，版本与来源追溯，批次原子导入，冲突保留，快照验证；已核实的合成职业参数可注入运行并随重放保存 |
| 素材登记与渲染 | 本地授权 PNG/JPEG/WEBP 解码、内容寻址、哈希与来源记录；显式 SpritePack 接入渲染、数据集和可重放任务 |
| 地图与数据集 | 两张自绘几何地图、地形/对象/镜头分离、可见区域标签、YOLO 格式、按会话拆分与完整性检查 |
| 模拟器 | 固定步长、确定性种子、保存恢复、平台与梯子、单段跳跃、战斗/冷却/Buff/资源/拾取/药品/NPC/切图 |
| 职业样例 | 标飞的弹药与跳标、龙骑的近身/范围技能、牧师的治疗与 MP 差异 |
| 视觉 | 像素基线、哈希绑定的 ONNX 适配器、检测基准；模型闭环采用 YOLO 动态对象与像素地形/HUD，策略只接收观测 |
| YOLO 工具 | 本地 tiny 配置、真实 CPU 训练、独立输入副本、ONNX 导出、真实运行时推理检查 |
| 任务与监督 | 有界任务与恢复、观测新鲜度、重复帧/时钟回退/未知状态处理、独立目标核验 |
| 验证实验 | 短暂/持续黑屏、重复帧、输入冻结、动作重放、状态及画面证据核验、按实际墙钟计时的长稳入口 |
| 本地录像与 OCR | PNG 帧目录/MP4 导入、完整性检查与感知回放；通过已安装的 Tesseract 识别文本 |
| 回话 | 模拟器 NPC 对话；人工确认的本地玩家回复草稿，不含发送后端 |

知识目录中的三个现有职业可通过 `--knowledge-db` 与 `--game-version` 显式加载完整 `ClassSpec`。加载器要求所有候选均为 VERIFIED、无冲突、`synthetic_fixture`；职业参数、哈希与来源进入运行证据，重放无需原数据库。地图、通用药品/Buff 等规则仍为样例常量。注册素材可通过成对的 `--asset-registry` 与 `--sprite-map` 参数接入渲染、数据集和任务，运行包内嵌素材后可独立重放。

本地录像已有导入、完整性验证和感知回放合同；固定录像不提供反事实动作验证。真实素材的识别效果、真实规则和新职业/地图包仍需各自的适配与验收。

## 模型与发布证据

- [发布验收记录](docs/RELEASE_ACCEPTANCE.md)：以具体提交、测试、操作系统和实测结果为准。
- [YOLO 实验记录](docs/YOLO_EXPERIMENT.md)：模型、数据划分、检测指标、闭环结果与未达到的门槛。
- [阶段状态与后续工作](docs/IMPLEMENTATION_PLAN.md)：区分已实现、已验证与需要外部素材的项目。

默认像素基线不是 YOLO。tiny YOLO 是在自绘合成数据上训练的模型，不是已经证明能识别真实游戏画面的通用成品。模型配置会打包进 wheel；仓库 `examples/fixture-model` 提供对应验收的 ONNX 样例，具体指标及局限见实验记录。模型闭环仍使用像素方法识别地形和 HUD，不能将混合系统成绩归为纯 YOLO。工具不会按模型名自动下载其他权重。

```bash
python -m maple_vision lab run --output artifacts/shipped-model-run --model examples/fixture-model/model.onnx --manifest examples/fixture-model/manifest.json
```

## 开发与自动验收

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -q
python -m build --wheel --no-isolation
python scripts/wheel_smoke.py --wheel dist/maple_vision_platform-0.1.0-py3-none-any.whl
```

这里的 `python` 应为上面虚拟环境的解释器。CI 配置同时检查 Ubuntu、Windows 的 Python 3.12，并在仓库之外新建仅含基础运行依赖的环境验证 wheel；另设 Ubuntu CPU 训练、ONNX 导出与推理检查。CI 成绩及版本见验收记录，不能从工作流文件存在推断已经通过。

依赖分为 `requirements-runtime.txt`、`requirements-dev.txt` 与 `requirements-training.txt`。训练环境的安装和可选测试开关见快速开始。已有 P0 的 `demo`、`db`、`run_once` 与 NoOp 合同保持兼容。

## 文档导航与范围

- [整体设计](MASTER_SPEC.md)
- [接口合同](docs/API_CONTRACTS.md) · [数据与来源](docs/CATALOG.md)
- [知识目录到运行职业参数](docs/CLASS_PROFILES.md)
- [注册素材接入渲染与数据集](docs/SPRITE_PACK.md) · [本地录像与 OCR](docs/RECORDINGS.md)
- [模拟器实际行为](docs/SIMULATOR_IMPLEMENTATION.md)
- [P0 历史验收](docs/P0_ACCEPTANCE.md) · [开发约束](AGENTS.md)

项目不包含正式客户端控制、系统按键注入、封包/内存修改或反作弊绕过。自建样例测试不能保证真实账号不被封禁。尚未核实的游戏素材、规则、真实录像泛化和完整 24 小时墙钟长稳，均须在验收记录中单独说明。
