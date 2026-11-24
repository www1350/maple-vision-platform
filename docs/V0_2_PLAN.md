# v0.2.0 执行与验收补充

本文把 v0.1 之后这一轮工程限定为三个可验收的闭环：**数据覆盖与分组评估、知识增量计划、指定模型的独立验证**。整体架构和从零安装见 [FINAL_PLAN.md](FINAL_PLAN.md)，本轮机器可读门禁见 [v02-contract.yaml](../engineering/v02-contract.yaml)，最终准确提交的结果由 [RELEASE_0_2_ACCEPTANCE.md](RELEASE_0_2_ACCEPTANCE.md) 记录。

当前代码版本是 **0.2.0 发布候选**。功能实现与局部回归已经推进，但完整工作树测试、远端 CI、新验证器最终门禁和候选模型推广尚未统一验收。本文不把这些待办写成成功结果，也不修改 [v0.1 验收](RELEASE_ACCEPTANCE.md)、[v0.1 模型实验](YOLO_EXPERIMENT.md) 或旧 `platform-contract.yaml`。

## 1. 本轮范围与验收状态

| 工作 | 当前新增实现 | 本轮需要完成的验收 | 保留的边界 |
|---|---|---|---|
| 知识增量 | 本地导出适配器、自包含 plan、完整基线、原子 apply、严格幂等 | 源字节/元数据篡改、过期基线、并发写入、批次失败与超预算回滚 | 没有网络抓取、最新小册子数据或运行时自动规则替换 |
| 生命周期数据 | schema v2、真实动作往返两图、六阶段与位置覆盖、重放核对像素/标签/统计 | 种子可重复、会话隔离、重哈希伪造拒绝、每图实际命中样本、legacy v1 字节兼容 | 自绘合成世界，两地图仍出现在所有 split |
| 分组评估 | 阶段、地图、地图×阶段、位置、重叠覆盖标签 | 聚合匹配不变、分区计数可回加、无该类真值时误检照计 FP、分组 AP 独立计算 | 单 IoU、指定置信度截断后的 AP，不是完整 COCO mAP50–95 |
| 指定模型验证 | suite/soak 接收显式检测器；新版 reinfer 实际执行感知/监督/策略 | 模型身份先验、完整 N+1 帧、终止状态、错误拒绝、预检与文件变化边界 | 新证据合同不反向补造旧运行包；混合感知仍用像素读取地形/HUD |
| 长稳资源与失败证据 | 墙钟/CPU/RSS 采样、有界保留；保留实际失败轮次的实现正在收口 | 原始失败轮次只捕获一次、异常路径资源报告、来源/单位/未知与保留预算 | 不是 24 小时已通过，不把未知 RSS 写成 0 |
| 模型实验与发布 | 新数据冻结，旧模型的新 val 基线已测；首个候选训练已完成，选模与独立验收进行中 | val 选模、选择后 test、三职业两图闭环、准确提交 CI 与 wheel 安装 | 工程发布不自动授予模型推广或真实域资格 |

新验证器的预检/竞态防护和失败轮次保留虽然已有实现，**最终行为仍以本版发布门禁为准**；在根代理完成独立反例、集成与准确提交检查前保持待验证状态。

## 2. 冻结输入与比较方法

本轮固定数据来自 `dataset create --profile lifecycle --sessions 18 --frames 48 --seed 20261125`：

| 项目 | 固定值 |
|---|---|
| manifest schema / generator | `2` / `lifecycle-v1` |
| 独立会话与帧数 | 18 × 48 = 864 |
| 可见标注 | 7098 |
| train / val / test | 480 / 192 / 192 帧 |
| 职业 | assassin、dragon_knight、cleric |
| 地图 | meadow、cavern |
| 阶段 | spawn、combat、loot、cleared、dialogue、transition |
| 位置 | left_edge、left、middle、right、right_edge |
| manifest SHA-256 | `670f6dae181f0f4b3107ba6a47cb733f19cfbc216c4e8e8c23f96b756601b3ff` |

本地实验路径为 `artifacts/v02-experiment/dataset`。这些可重建图像不自动加入源码 Git；验收应归档生成参数、manifest 哈希、覆盖报告和实际命令。不要修改正在训练的冻结输入，也不要把修复验证器当作重新选择有利样本的机会。[数据定义与覆盖证据](DATASET_LIFECYCLE.md)

旧模型 `examples/fixture-model/model.onnx` 的 SHA-256 为 `56cafce92c63d284744b9030d661486911e30c46b03000c605b530bc91d9ae04`。它在**本轮新 val 的 192 帧**上测得 AP50 **0.900697**、平台 recall **0.467014**，对应本地 `artifacts/v02-experiment/prior-model-validation.json`。这只是同版本数据上的候选比较基线，不是新训练模型的结果；不能直接与 v0.1 旧 test 的分数比较并宣称提升。

首个候选采用本地 tiny 配置、320px、40 epoch、batch 8、seed 42、CPU 2 线程；2026-11-24 已完成训练，记录位于 `artifacts/v02-experiment/training-tiny-320/training.json`，验证集比较和最终选择尚在执行。是否训练额外候选，由 val 暴露的具体错误和已记录预算决定；不能根据保留 test 反复调参。训练完成、成功导出、检测指标、闭环任务与默认模型推广各自报告。新模型数字、最终权重选择与推广结论由实际实验报告填写，本文不预填成功值。

## 3. 执行顺序与命令

以下命令在仓库根目录、已安装依赖的 Python 3.12 环境中运行。所有输出使用新路径；先创建 `artifacts` 父目录。可选训练环境安装仍遵守 [TRAINING.md](TRAINING.md) 和现有固定依赖。依赖安装后，下面流程使用本机计算，不调用 LLM API。

### 3.1 验证本地知识增量计划

```bash
python -m maple_vision knowledge init artifacts/v02-plan-knowledge.sqlite
python -m maple_vision knowledge plan-sync artifacts/v02-plan-knowledge.sqlite fixtures/knowledge-demo.json artifacts/v02-knowledge-plan.json
python -m maple_vision knowledge apply-sync artifacts/v02-plan-knowledge.sqlite artifacts/v02-knowledge-plan.json
python -m maple_vision knowledge apply-sync artifacts/v02-plan-knowledge.sqlite artifacts/v02-knowledge-plan.json
python -m maple_vision knowledge check artifacts/v02-plan-knowledge.sqlite
```

第一次 apply 应核对计划的完整源字节与目录基线并原子导入；数据库保持完整预期状态时，再次 apply 明确返回 `already_applied`。本例是合成 fixture；替换为真实资料时必须提供实际的来源、版本、授权和核实状态。

审查计划时看 added/unchanged/changed/missing 与 status_changes。missing 包括任何未在此次导出出现的基线证据，甚至是部分导出未涉及的其他版本，不能解释为上游已经删除。修改原源文件不改变已捕获的计划；修改后的资料应新建计划。数据库变化使旧计划 stale，只有完整预期结果已经存在时才幂等。详细 API、大小限制、事务与 sidecar 合同见 [KNOWLEDGE_SYNC.md](KNOWLEDGE_SYNC.md)。

### 3.2 生成并冻结 lifecycle 数据

```bash
python -m maple_vision dataset create artifacts/v02-plan-data --profile lifecycle --sessions 18 --frames 48 --seed 20261125
python -m maple_vision dataset check artifacts/v02-plan-data
python -m maple_vision vision benchmark artifacts/v02-plan-data --output artifacts/v02-prior-model-val.json --split val --model examples/fixture-model/model.onnx --manifest examples/fixture-model/manifest.json
```

先检查 manifest 哈希、split、类别与覆盖，之后将这份数据作为固定输入。训练生成器可以读取模拟真值规划真实动作，但检测器拿到的 `ImageFrame` 只含 RGB 与帧身份；coverage 不进入模型输入。

默认省略 `--profile` 仍使用 legacy v1。Python API 保留历史默认帧数；选择 lifecycle 时必须显式给出至少 12 帧，CLI 的 lifecycle 默认是 48 帧。完整阶段、位置和标签语义以 [DATASET_LIFECYCLE.md](DATASET_LIFECYCLE.md) 为准。

### 3.3 训练、验证集比较与固定候选

```bash
python -m maple_vision model train artifacts/v02-plan-data --output artifacts/v02-plan-train --config configs/tiny-yolo.yaml --epochs 40 --image-size 320 --batch 8 --seed 42
python -m maple_vision model export artifacts/v02-plan-train/run/weights/best.pt --output artifacts/v02-plan-export --image-size 320
python -m maple_vision vision benchmark artifacts/v02-plan-data --output artifacts/v02-candidate-val.json --split val --model artifacts/v02-plan-export/model.onnx --manifest artifacts/v02-plan-export/manifest.json
```

查看聚合和分组结果，特别是左侧出生、细长平台、空场景误检、掉落、对话和切图。分组 TP/FP/FN 可以在不重叠分区内回加，AP 不可相加，覆盖标签组也不可相加。根据预先记录的准入要求作选择；没有预先定义质量门槛时只能报告实验比较，不能补写“已达到稳定成品”的结论。

只有候选已固定、权重/manifest/阈值和选择理由已记录后，才执行保留 test：

```bash
python -m maple_vision vision benchmark artifacts/v02-plan-data --output artifacts/v02-candidate-test.json --split test --model artifacts/v02-plan-export/model.onnx --manifest artifacts/v02-plan-export/manifest.json
```

同时记录数据哈希、模型哈希、输入大小、依赖、confidence/IoU、每类与各组误差、延迟，以及尚未覆盖的场景。test 不达标就保留失败；继续依据它调参会使它失去独立留出资格。

### 3.4 新运行、实际重新推理和模型矩阵

```bash
python -m maple_vision lab run --output artifacts/v02-plan-model-run --job assassin --seed 0 --model artifacts/v02-plan-export/model.onnx --manifest artifacts/v02-plan-export/manifest.json
python -m maple_vision lab replay artifacts/v02-plan-model-run
python -m maple_vision lab reinfer artifacts/v02-plan-model-run --model artifacts/v02-plan-export/model.onnx --manifest artifacts/v02-plan-export/manifest.json
python -m maple_vision lab suite --output artifacts/v02-plan-model-suite --seeds 3 --model artifacts/v02-plan-export/model.onnx --manifest artifacts/v02-plan-export/manifest.json
```

`replay` 核对实际动作与环境；`reinfer` 从原初始状态实际重新计算观测、Supervisor、策略和动作，包括最后一帧。执行 N 步时应实际感知 N+1 次。模型身份或阈值不同在首个 detect 前拒绝；数值测量容差、精确身份字段、故障和终止行为见 [MODEL_VALIDATION.md](MODEL_VALIDATION.md)。

预期通过的故障场景可能以停止或失败终止，不应全部算作任务完成。一个失败任务被准确复现，表示失败证据一致，不表示原任务成功。suite/soak 预期保留原始失败轮次一次，不能重新执行一遍替代已经发生的失败；这一项与预检/文件变化防护仍需最终门禁确认。

### 3.5 短时模型长稳与资源测量

```bash
python -m maple_vision lab soak --output artifacts/v02-plan-model-soak --seconds 60 --resource-interval 1 --max-resource-samples 1024 --model artifacts/v02-plan-export/model.onnx --manifest artifacts/v02-plan-export/manifest.json
```

核对实际墙钟、完成轮次、独立任务检查、失败包和资源报告。进程 CPU 时间包括本进程所有线程；多线程 CPU 百分比可以大于 100%。Linux 当前 RSS 与进程生命周期峰值分别记录，平台不支持的值保持 `null` 并附原因。最大采样值不是精确瞬时峰值，资源报告也不是独立模型耗时测量。

默认最多保留 1024 个采样点，超限后抽稀并记录累计测量数、保留数和丢弃数；采样间隔不是实时调度保证。开发期间的并发训练负载必须单独说明。请求 60 秒的命令只做短时验收，不会因为某轮意外很慢而自动取得 24 小时资格。

## 4. 发布门禁与应归档的证据

| 门禁 | 必需证据 | 当前发布状态 |
|---|---|---|
| 准确提交基础检查 | commit SHA、Ruff、mypy、完整 pytest、跳过原因 | 待最终执行与归档 |
| 原生工具链 | 实际 CPU 训练、ONNX 导出、原生推理与适配器测试 | 待当前提交最终归档；不能挪用旧提交结果 |
| 知识计划 | 差异、事务、并发、篡改、超预算与幂等反例 | 局部验证完成，待全树/远端 CI |
| lifecycle 与分组 | 冻结哈希、轨迹/像素/标签核验、覆盖与聚合反例、legacy 兼容 | 局部验证完成，待全树/远端 CI |
| 新验证器 | 首次推理前检查、预算、文件变化、N+1 推理、终止比较、原始失败只捕获一次 | 最终门禁待确认 |
| 指定模型 suite/soak | 模型哈希、实际任务与故障结果、短时墙钟和资源记录 | 待最终模型及环境验收 |
| wheel 与跨平台 | 干净环境、仓库外运行、Ubuntu/Windows 当前提交 CI URL | 待最终执行与归档 |
| 新模型推广 | 固定选模规则、val 比较、选择后 test、逐阶段误差和独立闭环 | 待决定；平台发布不自动授予 |

最终记录由根代理汇总到 `docs/RELEASE_0_2_ACCEPTANCE.md`，再更新 `engineering/v02-contract.yaml` 和 `engineering/milestones.yaml`。更新状态必须附实际证据，未完成字段保持 pending/null，不用计划值代替结果。发布文档产生新提交后，最终提交仍需自己的 CI；已通过的上一个提交不自动覆盖新增改动。

v0.1 的历史报告、数据和样例模型分别保留。若要发布新模型，应先确定是新增版本化样例还是明确升级默认引用，并记录旧模型和新模型的对应哈希；不能悄悄替换二进制后继续使用旧实验说明。

## 5. 兼容合同

- P0 的 `demo`、`db`、NoOp 与原 v1 数据库合同保留；v1→v2 目录升级仍为显式操作。
- 原 `knowledge import/query/snapshot` 不改变语义；计划/应用是附加入口。
- `dataset create` 默认 legacy，v1 manifest 和旧实验不迁移；lifecycle 使用显式新 profile/schema。
- 旧 `lab replay` 继续只核对动作与环境。没有新版终止感知证据的旧包拒绝 reinfer，不回填推测内容。
- 默认像素基线保持可用；指定 ONNX 的验证需显式模型/manifest，动态对象不能由像素基线静默补全。
- 历史文档与验收合同归档保留；当前发行门禁使用独立 v02 合同。

## 6. 本轮之后仍未完成

完整 v8 继续包括网络小册子来源与持续更新、最新核实数据、其余知识类型的规则映射、动画/锚点、外部地图与职业包、真实 UI/OCR、跟踪与 BeliefState、通用任务 DSL、管理界面，以及独立真实录像标注和真实域验收。生命周期合成数据不替代这些内容。

24 小时门禁也仍未完成：需要固定提交、依赖、模型、数据和机器，**请求时长与实际墙钟均至少 86400 秒**，任务检查通过，并保留资源趋势、异常和实际失败证据。它是单独执行与验收的长时任务，不是本轮短时 soak 的另一种名称。

全部动作仍执行在独立模拟器中。真实游戏客户端控制、反作弊绕过和账号安全保证不属于该平台可验证的结论。
