# v0.2.0 执行与验收补充

本文把 v0.1 之后这一轮工程限定为三个可验收的闭环：**数据覆盖与分组评估、知识增量计划、指定模型的独立验证**。整体架构和从零安装见 [FINAL_PLAN.md](FINAL_PLAN.md)，本轮机器可读门禁见 [v02-contract.yaml](../engineering/v02-contract.yaml)，最终准确提交的结果由 [RELEASE_0_2_ACCEPTANCE.md](RELEASE_0_2_ACCEPTANCE.md) 记录。

当前代码版本是 **0.2.0 发布候选，本地工程与合成混合管线模型门禁已通过，最终发布提交远端 CI 待确认**。新验证器的预检/文件变化和保留原始失败轮次已完成独立审查与回归；所选模型新增在 `examples/lifecycle-model/`。最终测试计数由发布记录统一维护，本文不重复固化随测试收口变化的计数。[v0.1 验收](RELEASE_ACCEPTANCE.md)、[v0.1 模型实验](YOLO_EXPERIMENT.md) 和旧 `platform-contract.yaml` 保持不变。

## 1. 本轮范围与验收状态

| 工作 | 当前新增实现 | 已完成的本地验收，最终提交 CI 待确认 | 保留的边界 |
|---|---|---|---|
| 知识增量 | 本地导出适配器、自包含 plan、完整基线、原子 apply、严格幂等 | 源字节/元数据篡改、过期基线、并发写入、批次失败与超预算回滚 | 没有网络抓取、最新小册子数据或运行时自动规则替换 |
| 生命周期数据 | schema v2、真实动作往返两图、六阶段与位置覆盖、重放核对像素/标签/统计 | 种子可重复、会话隔离、重哈希伪造拒绝、每图实际命中样本、legacy v1 字节兼容 | 自绘合成世界，两地图仍出现在所有 split |
| 分组评估 | 阶段、地图、地图×阶段、位置、重叠覆盖标签 | 聚合匹配不变、分区计数可回加、无该类真值时误检照计 FP、分组 AP 独立计算 | 单 IoU、指定置信度截断后的 AP，不是完整 COCO mAP50–95 |
| 指定模型验证 | suite/soak 接收显式检测器；新版 reinfer 实际执行感知/监督/策略 | 模型身份先验、完整 N+1 帧、终止状态、错误拒绝、预检与文件变化边界 | 新证据合同不反向补造旧运行包；混合感知仍用像素读取地形/HUD |
| 长稳资源与失败证据 | 墙钟/CPU/RSS 采样、有界保留；保留实际失败轮次一次 | 失败不重跑、异常路径资源报告、来源/单位/未知与预算；所选模型短时运行通过 | 不是 24 小时已通过，不把未知 RSS 写成 0 |
| 模型实验与发布 | 新数据与模型冻结；所选 lifecycle 模型完成本地混合管线准入 | val 选择、选择后 test、三职业两图闭环、实际 reinfer、13场景与短时soak | 纯 YOLO 地形未通过，真实域未验收，最终远端 CI 待确认 |

本地已核验新验证器的预检/文件变化边界、实际 N+1 帧重新推理和保留原始失败轮次一次。发布状态仍需最终提交自己的远端检查；初始实现提交 `4e9914411c1b3f49b2f8eb81d7c17948af8ddb82` 已进入 [PR #3](https://github.com/www1350/maple-vision-platform/pull/3)，初始 [CI 45390705416](https://github.com/www1350/maple-vision-platform/actions/runs/45390705416) 不能自动覆盖后续测试、模型或文档提交。

首轮远端 Ubuntu 与原生训练作业通过，Windows 暴露了 **legacy 原始 manifest 哈希与 Linux 参考值不同** 的测试假设问题。原 v0.1 与当前生成器已在分别固定的 LF、CRLF 环境中逐文件证明一致；现在用原 v0.1 派生的严格 RGB、标签/YAML 和 manifest 内容指纹做跨平台验收，仍保留原始哈希完整性与同环境字节确定性。没有跳过 Windows 或白名单接受新哈希，实际 Windows 内容一致性由下一轮 CI 确认。[对照记录](../reports/release-0.2.0/legacy-content-diagnosis.json)

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

本地实验路径为 `artifacts/v02-experiment/dataset`。这些可重建图像不自动加入源码 Git；生成参数、manifest 哈希、覆盖报告和实际命令随本轮报告归档。冻结输入保持不变，不能把修复验证器当作重新选择有利样本的机会。[数据定义与覆盖证据](DATASET_LIFECYCLE.md)

旧模型 `examples/fixture-model/model.onnx` 的 SHA-256 为 `56cafce92c63d284744b9030d661486911e30c46b03000c605b530bc91d9ae04`。它在**本轮新 val 的 192 帧**上测得 AP50 **0.900697**、平台 recall **0.467014**，对应本地 `artifacts/v02-experiment/prior-model-validation.json`。这只是同版本数据上的候选比较基线，不是新训练模型的结果；不能直接与 v0.1 旧 test 的分数比较并宣称提升。

所选首个候选采用本地 tiny 配置、320px、40 epoch、batch 8、seed 42、CPU 2 线程，实际训练 **1067.895699 秒**。其 ONNX SHA-256 为 `7e9f61576f63be400909db6f9eb6cb89b5b935385d01a24155c7f765d2beb6e3`，在同一新 val 上 AP50=**0.946251768**、platform recall=**0.661458333**。选模工程准则是在首次候选 val 后制定，**不是预注册准则**；同时明确参考独立列出的合成任务套件。

选择于 **2026-11-24 12:06:01 UTC** 固定，随后对保留 test 评估一次，不根据 test 再调参。没有训练第二候选，也没有修改冻结标签。完整选模时序、数据、原始指标和架构检查见 [YOLO_LIFECYCLE_EXPERIMENT.md](YOLO_LIFECYCLE_EXPERIMENT.md)。

### 本地模型准入结果

| 验证 | 实测结果 | 适用范围 |
|---|---|---|
| 保留 test | 192帧、1572GT；TP/FP/FN=1372/3/200；AP50=0.944822452，micro P=0.997818182、R=0.872773537 | confidence≥0.25、IoU=0.5；只评估冻结的合成数据 |
| 三职业任务 | assassin/dragon_knight/cleric 为327/325/317步，独立目标检查全部通过 | 两张内置地图的混合感知管线 |
| 实际 reinfer | 328/326/318帧，感知、策略与终止帧核对全部通过 | 同一模型哈希、manifest与阈值；包含每轮最后一帧 |
| 模型 suite | 13/13场景符合预期 | 正常任务与故障停止分别判定 |
| 模型短时soak | 请求60秒，实际61.510525秒，完成10轮；63资源样本；最大采样当前RSS90.09375MiB | 本进程短时结果；24小时门禁未完成 |

这组结果支持新增 `examples/lifecycle-model/` 作为本轮**合成混合管线**样例；不会覆盖旧模型或更改默认像素基线。报告归档到 `reports/release-0.2.0/`，发布结论见 [RELEASE_0_2_ACCEPTANCE.md](RELEASE_0_2_ACCEPTANCE.md)。

### 纯 YOLO 地形仍未通过

实际导出模型的 `reg_max=8`、stride 为4/8，DFL距离 bins 为0…7。单框最大宽度为 `2×7×8=112px`，而全宽地板为320px，单框IoU上界为0.35，达不到IoU=0.5。这解释了新val中192个全宽地板必然漏配，不能扩大解释为全部平台误差。

针对“增加stride32头”的候选，实际安装版本匹配器在当前320输入、4px高地板及letterbox位置上，为该分支分配的可用地板正样本锚点数为0。理论框宽增加并不能解决这个训练分配问题，因此没有训练该候选。此结论限定于实际记录的架构、输入几何与匹配器版本，不是所有stride32模型的通用结论。

当前任务使用YOLO动态对象加像素地形/HUD。后续纯地形研究应先检查可表示框范围、实际正样本分配与CPU成本，再做新的受控实验；不能合并碎框、改标签或调低IoU来冒充本轮纯YOLO地形通过。

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

查看聚合和分组结果，特别是左侧出生、细长平台、空场景误检、掉落、对话和切图。分组 TP/FP/FN 可以在不重叠分区内回加，AP 不可相加，覆盖标签组也不可相加。准入要求必须如实记录制定时点：本轮在首次val后记录工程要求，不能事后称为预注册实验。所获结论限于本次合成混合管线，不是“稳定通用成品”。

只有候选已固定、权重/manifest/阈值和选择理由已记录后，才执行保留 test：

```bash
python -m maple_vision vision benchmark artifacts/v02-plan-data --output artifacts/v02-candidate-test.json --split test --model artifacts/v02-plan-export/model.onnx --manifest artifacts/v02-plan-export/manifest.json
```

同时记录数据哈希、模型哈希、输入大小、依赖、confidence/IoU、每类与各组误差、延迟，以及尚未覆盖的场景。test 不达标就保留失败；继续依据它调参会使它失去独立留出资格。

### 3.4 新运行、实际重新推理和模型矩阵

下面直接使用已选定并保存的新模型；若要评估自己重新训练的权重，应将所有模型/manifest参数成对改为新导出路径，单独记录该权重的结果。

```bash
python -m maple_vision lab run --output artifacts/v02-plan-model-run --job assassin --seed 0 --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
python -m maple_vision lab replay artifacts/v02-plan-model-run
python -m maple_vision lab reinfer artifacts/v02-plan-model-run --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
python -m maple_vision lab suite --output artifacts/v02-plan-model-suite --seeds 3 --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
```

`replay` 核对实际动作与环境；`reinfer` 从原初始状态实际重新计算观测、Supervisor、策略和动作，包括最后一帧。执行 N 步时应实际感知 N+1 次。模型身份或阈值不同在首个 detect 前拒绝；数值测量容差、精确身份字段、故障和终止行为见 [MODEL_VALIDATION.md](MODEL_VALIDATION.md)。

预期通过的故障场景可能以停止或失败终止，不应全部算作任务完成。一个失败任务被准确复现，表示失败证据一致，不表示原任务成功。suite/soak 保留原始失败轮次一次，不能重新执行一遍替代已经发生的失败；这一项与预检/文件变化防护已通过本地审查与回归，最终提交CI仍须覆盖。

### 3.5 短时模型长稳与资源测量

```bash
python -m maple_vision lab soak --output artifacts/v02-plan-model-soak --seconds 60 --resource-interval 1 --max-resource-samples 1024 --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
```

核对实际墙钟、完成轮次、独立任务检查、失败包和资源报告。进程 CPU 时间包括本进程所有线程；多线程 CPU 百分比可以大于 100%。Linux 当前 RSS 与进程生命周期峰值分别记录，平台不支持的值保持 `null` 并附原因。最大采样值不是精确瞬时峰值，资源报告也不是独立模型耗时测量。

默认最多保留 1024 个采样点，超限后抽稀并记录累计测量数、保留数和丢弃数；采样间隔不是实时调度保证。开发期间的并发训练负载必须单独说明。请求 60 秒的命令只做短时验收，不会因为某轮意外很慢而自动取得 24 小时资格。

## 4. 发布门禁与应归档的证据

| 门禁 | 必需证据 | 当前发布状态 |
|---|---|---|
| 准确提交基础检查 | commit SHA、Ruff、mypy、完整 pytest、跳过原因 | 本地源码检查通过；新增测试后的最终计数以发布记录为准 |
| 原生工具链 | 实际 CPU 训练、ONNX 导出、原生推理与适配器测试 | 本地通过；最终提交远端CI待确认 |
| 知识计划 | 差异、事务、并发、篡改、超预算与幂等反例 | 本地审查与全树回归通过 |
| lifecycle 与分组 | 冻结哈希、轨迹/像素/标签核验、覆盖与聚合反例、legacy 兼容 | 本地审查与全树回归通过 |
| 新验证器 | 首次推理前检查、预算、文件变化、N+1 推理、终止比较、原始失败只捕获一次 | 本地独立审查与回归通过 |
| 指定模型 suite/soak | 模型哈希、实际任务与故障结果、短时墙钟和资源记录 | 所选模型13/13场景、61.510525秒/10轮通过 |
| wheel 与跨平台 | 干净环境、仓库外运行、Ubuntu/Windows 当前提交 CI URL | 最终发布提交远端CI待确认 |
| 新模型推广 | 准则制定时点、val比较、选择后test、逐阶段误差和独立闭环 | 合成混合管线本地通过；纯YOLO地形未通过 |

最终记录由根代理汇总到 `docs/RELEASE_0_2_ACCEPTANCE.md`，再更新 `engineering/v02-contract.yaml` 和 `engineering/milestones.yaml`。更新状态必须附实际证据，未完成字段保持 pending/null，不用计划值代替结果。发布文档产生新提交后，最终提交仍需自己的 CI；已通过的上一个提交不自动覆盖新增改动。

v0.1 的历史报告、数据和样例模型分别保留；v0.2 采用新增 `examples/lifecycle-model/` 的方式保存模型与manifest，旧 `examples/fixture-model/` 不变。当前命令示例显式选择新模型，省略模型参数时仍使用像素基线。

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
