# 从安装到独立验证

本文命令对应当前 v0.3 CLI，并保留既有流程。默认运行对象是内置的独立模拟器；例子中的职业、地图与数值为合成样例。安装依赖需要访问包源，运行无需登录游戏、提供 LLM 密钥或下载远程模型。

## 1. 安装

使用 Python 3.12。推荐从仓库根目录安装固定版本的开发环境，其中包含测试与 ONNX 推理依赖，尚不包含较大的 PyTorch/YOLO 训练依赖。

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -m maple_vision --help
```

### Windows PowerShell

无需修改系统执行策略，也无需激活脚本；直接使用虚拟环境解释器：

```powershell
py -3.12 -m venv .venv
$maplePython = ".\.venv\Scripts\python.exe"
& $maplePython -m pip install -r requirements-dev.txt
& $maplePython -m pip install --no-build-isolation --no-deps -e .
& $maplePython -m pip check
& $maplePython -m maple_vision --help
```

后文通用示例使用 `python -m ...`。在上述 PowerShell 会话中，将开头替换为 `& $maplePython -m ...`；直接运行脚本时使用 `& $maplePython scripts/wheel_smoke.py ...`。Linux 激活环境后可以直接复制。

基础运行只需要 `requirements-runtime.txt` 中的 NumPy 和 Pillow。若直接安装已构建 wheel，可以先装此文件，再执行 `python -m pip install --no-deps path/to/package.whl`。`model` 训练/导出另需下面的可选训练环境。

## 2. 跑通多职业、多地图任务

```bash
python -m maple_vision lab run --output artifacts/assassin-001 --job assassin --seed 0
python -m maple_vision lab replay artifacts/assassin-001
python -m maple_vision lab run --output artifacts/knight-001 --job dragon_knight --seed 1
python -m maple_vision lab run --output artifacts/cleric-001 --job cleric --seed 2
```

正常任务在 `meadow`、`cavern` 两张地图完成战斗、拾取、对话与切图。运行默认采用像素基线，策略读取画面中的角色、怪物、掉落、平台、门、NPC 和 HP/MP 条；策略拿不到模拟器真值。真值只用于渲染与独立验收。

检查 stdout JSON 中的 `status` 和 `independent_goal_check`。任务状态为 `completed` 且独立目标检查通过，才算该样例任务完成。接受攻击指令不等于命中；视觉目标消失也不会被直接当成确认击杀。

输出目录包括：

| 文件 | 用途 |
|---|---|
| `summary.json` | 结果、动作/事件计数、实际与模拟时间、配置及版本信息 |
| `trace.jsonl` | 每步观测、监督决策、请求/实际动作、执行结果、状态哈希 |
| `initial-state.json` / `final-state.json` | 独立模拟器状态与确定性随机数状态 |
| `frames/*.png` | 按采样间隔保存的视觉输入 |
| `final.png` / `preview.gif` | 最终输入画面与采样预览 |
| `manifest.json` | 文件清单、哈希和重放范围 |

`lab replay` 重放实际动作，检查时序、执行结果、状态、采样画面、最终画面与独立目标报告。它不宣称能把普通录像变成支持反事实操作的环境，也不把没有重跑的神经网络推理当成已重现。

所有运行输出目录须不存在，防止覆盖旧证据。日志较大时可用 `--record-stride 10` 减少采样图片；完整动作轨迹仍保留。`--max-steps` 限制任务预算。

## 3. 故障、恢复与长稳

```bash
python -m maple_vision lab suite --output artifacts/suite-001 --seeds 3
python -m maple_vision lab soak --output artifacts/soak-001 --seconds 60
```

套件包含三职业、多种子与故障场景：短暂黑屏、持续黑屏、重复帧、输入冻结。短暂问题应恢复；持续未知状态应停止；无法恢复的执行故障应失败。故障测试的成功条件是按预期处理故障，不是把每个场景都强行标成任务成功。

长稳入口按墙上时间计时并输出实际耗时和模拟时间。60 秒测试不算 24 小时测试；`--seconds 86400` 才请求完整 24 小时运行，需要本地计算机持续工作。是否完成以报告为准，不能用加速模拟时间代替真实墙钟时长。

CLI 成功退出为 `0`；参数、文件、数据校验错误为 `2`；`lab run` 明确失败/停止，或套件未通过时为 `1`。故障套件自身全部符合预期时返回 `0`。stdout 是 JSON 结果，错误信息写 stderr，便于其他编程 AI 或脚本接入。

## 4. 本地知识目录

```bash
python -m maple_vision knowledge init artifacts/knowledge.sqlite
python -m maple_vision knowledge import artifacts/knowledge.sqlite fixtures/knowledge-demo.json
python -m maple_vision knowledge query artifacts/knowledge.sqlite classes assassin fixture-v1
python -m maple_vision knowledge snapshot artifacts/knowledge.sqlite artifacts/knowledge-snapshot-001
python -m maple_vision knowledge verify-snapshot artifacts/knowledge-snapshot-001
python -m maple_vision lab run --output artifacts/catalog-run-001 --job assassin --knowledge-db artifacts/knowledge.sqlite --game-version fixture-v1
```

若从这里开始运行而尚无 `artifacts` 父目录，先创建它。示例导入文件明确标为 `synthetic_fixture`，不是已经爬取的小册子资料。真实来源须用允许使用的导出数据，保留源 URI、游戏版本、时间、许可说明与核实状态，格式见 [CATALOG.md](CATALOG.md)。

默认查询只返回 `VERIFIED` 且未冲突的候选。检查未知资料或冲突记录可增加 `--allow-unverified`，这不会将其升级为可信游戏规则。运行时职业加载器更严格：指定职业/版本的全部候选必须为 VERIFIED、无冲突的 `synthetic_fixture`，且 payload 必须包含完整 `ClassSpec` 字段。样例文件的三个职业已采用此格式。

通过 `--knowledge-db` 可以改变现有职业的生命/MP 上限、普通攻击伤害与距离、消耗和冷却、移动和跳跃速度、治疗量及跳标能力。参数有明确整数边界，并与任务职业 ID 一致。输入数据库仅在启动时读取；保存会内嵌完整职业参数与哈希，重放无需原数据库。来源证据单独写入运行报告。地图、通用 Buff/药品及范围技能半径等规则仍为内置样例，不会把任意知识条目自动变成新玩法。画面素材通过下面的 SpritePack 入口单独接入。

`knowledge` 使用 schema v2。旧 P0 命令 `db init` / `db check` 仍使用 v1；不要用旧 `db` 命令打开已经升级的 v2 文件。升级已有库时发现不兼容结构会拒绝并保留原数据。

## 5. 授权素材登记与渲染

```bash
python -m maple_vision assets import artifacts/registry ./my-sprite.png --entity-id fixture.player --game-version fixture-v1 --source-uri urn:my-project:player --license CC0-1.0 --permission self_authored
python -m maple_vision assets list artifacts/registry
python -m maple_vision assets verify artifacts/registry ASSET_ID_FROM_IMPORT
```

支持单帧 PNG、JPEG、WEBP。`asset_id` 从首次导入的 JSON 中取得；先检验解码和来源字段，再写入按哈希寻址的图片及元数据。示例授权值适用于你自行创作的图片；第三方图片应填写实际许可或明确授权说明，不能仅通过改字段获得授权。

按 [SPRITE_PACK.md](SPRITE_PACK.md) 创建类别到 `asset_id` 的映射 JSON 后，在 `dataset create` 或 `lab run` 后增加成对参数 `--asset-registry artifacts/registry --sprite-map ./sprite-map.json`，即可把注册图片用于实际渲染。运行包保留素材及其哈希，重放不需要原注册表。自定义图片需要适配的检测器；默认颜色基线不能识别任意素材，识别不足时会暂停或停止。

## 6. 生成数据集并建立视觉基线

```bash
python -m maple_vision dataset create artifacts/dataset-001 --sessions 18 --frames 8 --seed 7
python -m maple_vision dataset check artifacts/dataset-001
python -m maple_vision vision benchmark artifacts/dataset-001 --output artifacts/pixel-benchmark.json --split test
python -m maple_vision vision observe artifacts/assassin-001/final.png
```

数据集包含 PNG、YOLO 标签、`data.yaml` 与 manifest。train/val/test 按独立会话划分，不将同一会话相邻帧拆进不同集合。验证检查路径、文件哈希、图片尺寸、标签范围、会话归属、时间序号、类别与地图覆盖。

颜色基线仅用于自绘几何画面，可帮助发现管线和任务逻辑问题。它没有真实游戏图像泛化能力。合成测试划分也不等于真实录像独立评估；对真实素材做实验时必须保留来源、会话和场景分组，另做独立验收。

## 7. 安装可选 CPU YOLO 训练环境

训练不是默认安装的一部分。固定 CPU wheels 从 PyTorch 官方 CPU 索引安装，其余固定依赖由项目训练锁文件提供；这些安装步骤需要网络，训练过程使用本地配置和本地数据。

Linux 激活环境后：

```bash
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu torchvision==0.29.2+cpu
python -m pip install -r requirements-training.txt
python -m pip check
```

Windows PowerShell：

```powershell
& $maplePython -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.15.0+cpu torchvision==0.29.2+cpu
& $maplePython -m pip install -r requirements-training.txt
& $maplePython -m pip check
```

CPU 训练自动验收作业配置在 Ubuntu；Windows 的基础运行、数据处理、原生 ONNX 适配器和 wheel 验收单独执行。Windows 训练环境能否完全通过，应以该平台实测为准。

## 8. 训练、导出、评估与候选闭环

仓库附带自绘场景的验收模型。已安装 `requirements-dev.txt` 后可以先运行：

```bash
python -m maple_vision lab run --output artifacts/shipped-model-001 --job assassin --model examples/fixture-model/model.onnx --manifest examples/fixture-model/manifest.json
python -m maple_vision lab replay artifacts/shipped-model-001
```

这是混合视觉：动态玩家、怪物、掉落、NPC 和传送门来自 YOLO；地形、梯子与 HUD 由像素基线读取。该组合在合成场景中的结果不能归为纯 YOLO 的地形识别成绩，不能推断真实游戏泛化。检测基准报告仍展示模型对全部声明类别的表现，未用像素结果替换模型的薄弱类别。

从头构建自己的合成候选：

```bash
python -m maple_vision model train artifacts/dataset-001 --output artifacts/training-001 --epochs 3 --image-size 160 --batch 8 --seed 7
python -m maple_vision model export artifacts/training-001/run/weights/best.pt --output artifacts/export-001 --image-size 160
python -m maple_vision vision benchmark artifacts/dataset-001 --output artifacts/yolo-benchmark.json --split test --model artifacts/export-001/model.onnx --manifest artifacts/export-001/manifest.json
python -m maple_vision lab run --output artifacts/yolo-mission-001 --job assassin --model artifacts/export-001/model.onnx --manifest artifacts/export-001/manifest.json
```

默认 `tiny-yolo.yaml` 随包安装，也可以通过 `--config configs/tiny-yolo.yaml` 显式选择仓库内配置。它是本项目的小型检测结构，由 Ultralytics 提供网络层；从随机权重开始，不按远程模型名自动下载预训练参数。

训练会为数据创建独立输入副本，避免训练缓存修改源数据集。导出结果包含 `model.onnx`、模型哈希清单与导出报告；推理必须同时提供模型和对应 manifest。权重加载只用于可信的本地 `.pt` 文件，第三方来源和许可应单独核实。

这里的 3 epoch 是操作示例，不是精度承诺。候选模型可能检测不到角色，或在平台/遮挡/小目标上失效；这时闭环会暂停或失败，并保留证据。不能因训练结束或 ONNX 导出成功就把默认基线切换为模型。

验收顺序是：

1. 训练产出及文件完整性通过。
2. ONNX 输出合同和真实运行时推理通过。
3. 独立 test 划分的检测质量、覆盖、延迟满足明确目标。
4. 多职业、多地图、故障和任务结果由独立验证器通过。
5. 如目标包含真实录像，单独验证真实录制会话；合成成绩不能替代。

已经运行的模型实验与失败原因见 [YOLO_EXPERIMENT.md](YOLO_EXPERIMENT.md)。正式结论以该报告及 [RELEASE_ACCEPTANCE.md](RELEASE_ACCEPTANCE.md) 的样本量、版本和实测为准。

## 9. 玩家回复草稿

```bash
python -m maple_vision chat-draft --incoming "现在有空吗？" --reply "稍等，我忙完回复你。"
```

命令输出 `requires_human_confirmation: true` 和 `sent: false`。它不调用聊天服务，也不发送玩家消息。模拟器 NPC 对话是另一个本地玩法机制。

## 10. 本地验收与打包

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -q
python -m build --wheel --no-isolation
python scripts/wheel_smoke.py --wheel dist/maple_vision_platform-0.4.0-py3-none-any.whl
```

`wheel_smoke.py` 在仓库之外创建全新虚拟环境，只安装固定 NumPy/Pillow 与构建 wheel，验证 CLI、P0/v1 数据库、知识快照及增量计划、素材、legacy/lifecycle 数据、完整视觉任务和实际重新推理。它还检查两份模型配置均已打包，并用自定义检测器验证模型地形路线不依赖颜色检测或训练包。stdout 是一份 JSON，默认报告位于 `artifacts/wheel-smoke.json`；可用 `--output` 改路径。`dist` 只有一个 wheel 时可省略 `--wheel`。

训练集成测试默认跳过，避免基础开发环境隐式安装和运行大型依赖。安装训练环境后，显式启用真实训练/导出/推理验证：

Linux：

```bash
MAPLE_TRAINING_SMOKE=1 python -m pytest tests/test_training.py tests/test_terrain_audit.py -q
```

Windows PowerShell：

```powershell
$env:MAPLE_TRAINING_SMOKE = "1"
& $maplePython -m pytest tests/test_training.py tests/test_terrain_audit.py -q
Remove-Item Env:MAPLE_TRAINING_SMOKE
```

CI 配置包含 Ubuntu/Windows 基础检查和独立 Ubuntu CPU 训练检查；具体通过记录见发布验收文档。真实游戏素材、真实规则、真实录像泛化及完整 24 小时长稳属于各自的验收项。项目没有正式客户端控制或反检测后端，本地结果不能保证正式账号安全。

## 11. 本地录像与 OCR

把获准分析的 PNG 帧放入 `./recording-frames`，使用补零文件名保持顺序，并填写真实来源与许可。PNG 目录必须声明 FPS；下面以 20 FPS 为例，输出路径须为新目录：

```bash
python -m maple_vision recording import ./recording-frames artifacts/recording-001 --source-uri urn:my-recording:session-001 --permission explicit_permission --license "Owner-provided recording; local analysis permitted" --fps 20
python -m maple_vision recording check artifacts/recording-001
python -m maple_vision recording observe artifacts/recording-001 --output artifacts/recording-observations-001
```

`recording observe` 可成对增加 `--model` 与 `--manifest`。默认颜色基线仅适用于本项目几何场景；无标注录像不会生成准确率或动作成功率。MP4 导入另需可选 OpenCV/FFmpeg 后端，时间戳采用声明或元数据 FPS 估算。详细限制见 [RECORDINGS.md](RECORDINGS.md)。

OCR 需要本机已安装 Tesseract 和所选语言包。先查看可用语言，再按实际列表选择；下例使用 `eng`：

```bash
tesseract --list-langs
python -m maple_vision vision ocr artifacts/recording-001/frames/000000.png --language eng
```

Tesseract 不在 PATH 时，为 `vision ocr` 指定 `--binary` 的绝对路径；语言数据在自定义目录时增加 `--tessdata-dir`。仅在已安装 `chi_sim` 时使用该语言或 `eng+chi_sim`；程序不会自动下载语言包。OCR 输出是识别文本与分数，真实游戏小字和中文效果须另行验证。

## 12. 地形研究与可重复的质量判断

本轮地形候选在验证集上未达到总平台召回门槛，保存在 [terrain-candidate](../examples/terrain-candidate/README.md)，没有替换已验收的 lifecycle 混合模型。无需训练或调用 LLM，即可用标准库核对保存的结果：

```bash
python scripts/check_terrain_experiment.py --preregistration reports/release-0.3.0/preregistration.json --candidate reports/release-0.3.0/terrain-model-validation.json --baseline reports/release-0.3.0/lifecycle-baseline-validation.json --output artifacts/rechecked-terrain-gates.json
```

这条命令预期返回 **1** 并保存13/14通过、平台召回未通过的结果。返回0才表示所给报告的质量数字过线，返回2表示输入缺陷或输出已存在；任何返回码都不会自动推广模型或证明真实域可靠性。脚本检查报告中的数据/模型/阈值/人口/运行配置，实际重新评分使用 `vision benchmark --terrain-metrics --split val`，详见候选目录的复现命令。

训练环境可在训练前做可表示性检查：

```bash
python -m maple_vision model audit-terrain artifacts/reproduced-terrain-data --config configs/terrain-yolo.yaml --split val --iterations 0 --output artifacts/structure-audit.json
```

该命令只允许 train/val，构建前检查有界拓扑和分配形状，之后检查实际DFL/stride、分配器候选和每边目标范围。`--iterations 0` 不测随机权重延迟，不能据此宣称部署速度。

给自己已经完成独立验收的本地模型选择模型地形时，`lab run/suite/soak/reinfer` 都必须显式使用相同的 `--terrain-source model` 和模型/manifest。`vision observe` 也支持该开关；`vision benchmark` 本来就直接评价模型，不使用该开关。此模式七类对象全部来自模型，UI仍为合成像素，不会用颜色地形补漏。[模式合同](MODEL_TERRAIN.md) · [地形误差定义](TERRAIN_EVALUATION.md) · [本轮完整计划](V0_3_PLAN.md)
