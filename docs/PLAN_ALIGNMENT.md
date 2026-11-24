# 原始 v8 总纲与当前实现对照

核对日期：2026-11-25。实现基线为已合并的 v0.4，提交 `320989e0bf4b906cb4d7ed7e6cf8eef33bcd5473`。

**结论：工程主链已经在受限合成场景中运行并验收，原始 22 章总纲尚未完整实现。** 多职业、多地图、数据生产、模型工具和验证平台均已有可运行基础；通用能力编译、任务规划、状态融合、真实数据、产品界面和完整发布门禁仍有实质缺口。版本号、测试数量及某一增量进入 P13 都不能换算成总体完成百分比。

## 1. 原方案来源与文档职责

用户指定的权威原文为 [Maple Vision Platform v8.0 — 最终工程实施总纲](https://chatgpt.com/s/t_6b05d66c4fac81918da3094a5f80f7dc)，共有 22 章。此前仓库中的 `MASTER_SPEC.md` 是浓缩摘要，省略了完整数据库设计、能力编译器、任务分层、模型比较、产品界面等细节，**不等同于这份完整原文**。本次已恢复共享页提供的完整 Markdown 正文，并保存交互内容的原始文本。

| 文件 | 职责 | 阅读时的边界 |
|---|---|---|
| [MASTER_SPEC.md](../MASTER_SPEC.md) | 原始 22 章设计基线，保留原文 | 目标、原型与历史指令不是当前实现或验收结果 |
| [共享页交互源码](source-plan/shared-plan-v8.source.txt) | 保存原页的交互图示与界面原型源文本 | 仅归档，不执行；Markdown 回退文本不能复现全部交互元素 |
| [来源与校验记录](source-plan/shared-plan-v8.metadata.json) | 原网址、提取方式、字节数、SHA-256、旧摘要身份 | 校验内容一致性，不把网页原型当作已开发产品 |
| 本文 | 原始各章与当前代码、证据和缺口的对应关系 | 将设计要求与受限实现明确分开 |
| [FINAL_PLAN.md](FINAL_PLAN.md) | 当前实现的复现、接口和编程 AI 接手指南 | 补充实施事实，不缩减或替代原始总纲 |
| [版本验收](RELEASE_0_4_ACCEPTANCE.md)与 PR | 实际执行、失败、修复、准入与准确提交检查 | 只覆盖明确标出的版本、数据、模型和场景 |

原文第 21 章的“只开始 P0，验收后停止”是初始阶段指令。用户后续已授权连续实现、测试并交付，见 [AGENTS.md](../AGENTS.md)；该后续授权继续有效，不应因恢复历史原文再次停在 P0。其余架构要求不能用“先做样例”永久替代。

原文执行范围已经是自建模拟器或明确授权的测试环境，且排除正式客户端自动控制、输入伪装和反作弊绕过。当前离线验证也不能证明正式账号不会被封禁。

## 2. 状态判定规则

- **已实现并验收**：有运行代码和指定版本的独立证据；只对报告覆盖的范围成立。
- **部分实现**：有实际功能，但原章的重要能力、通用性或数据仍缺失。
- **未实现**：只有设计、接口占位或原型；不能按名称存在计作完成。
- **未验收或未通过**：功能工具可能存在，但目标效果没有证据或已明确失败。

下表逐章覆盖原文；“当前实现”均以 v0.4 基线及保留的历史证据为准。后续新增接口必须更新对应行，并关联新的独立验收，不能回写旧实验结果。

## 3. 原始 22 章逐章映射

| 原章 | 原始交付要求 | 当前实现、状态与证据 | 仍需完成的内容 |
|---|---|---|---|
| 01 总体架构 | 数据、素材、录像进入知识底座；Tracking、状态融合、规划、调度、受控执行和验证组成主链 | **部分实现**：`Simulator → ImageFrame → Perception → Supervisor → VisualAgent → SimAction → Simulator` 与独立 oracle 已运行；数据/训练支路存在。[当前接口](API_CONTRACTS.md) | 原图中的 Tracker、完整 WorldState/BeliefState、Planner、Sequencer、Scheduler、Session Manager 未形成完整运行链；须补接口迁移与跨层证据 |
| 02 技术选型 | 单机 Python/SQLite，YOLO/ONNX、本地 OCR、Pygame CE、图结构与 FastAPI UI，运行时无 LLM | **部分实现，存在替代选型**：Python 3.12、SQLite、Pillow、可选 OpenCV/YOLO/ORT、Tesseract、pytest 等已用；本地运行不调用 LLM。[依赖](../pyproject.toml) | 模拟器为 Python 逻辑加 Pillow 渲染，尚无 Pygame 交互窗口、FastAPI/UI 或通用图引擎；领域协议主要为 dataclass 加显式校验，而非 Pydantic。替代选型须保留理由，不伪称原组件已实现 |
| 03 游戏数据底座 | 小册子获准同步、原始归档、规范化、差异审阅、快照，以及职业/技能/怪物/地图等领域表 | **部分实现**：`catalog.py`、`knowledge_sync.py` 支持本地导出、来源/版本/冲突、计划与原子应用、SQLite 快照；三职业合成参数可注入。[知识同步](KNOWLEDGE_SYNC.md) | 小册子网络适配、持续同步、已核实真实数据库未完成；当前是 `source_records` 与通用 `catalog_records`，未建原章全套领域表和关系约束；其它条目存在不代表规则已进入模拟器 |
| 04 Asset Registry | 按实体及版本关联图片、动画、技能效果、背景、UI，SQLite 保存索引 | **部分实现**：`asset_registry.py`、`sprite_pack.py` 登记并验证本地图片，静态 RGBA 可进入渲染、标签和自包含运行包。[素材包](SPRITE_PACK.md) | 当前索引为本地元数据文件；原计划 SQLite 资源索引、动画帧/锚点、完整实体关联、获准真实素材库尚未交付 |
| 05 地图渲染平台 | 逻辑结构与画面分离，镜头、坐标、图层、效果；评估 MapleNecrocer；完整地图内容包 | **部分实现**：Pillow 分层渲染、视差、静态 SpritePack、可见标签和部分镜头变换；v0.4 有严格地图 JSON。[地图包](MAP_PACKS.md) | MapleNecrocer 许可/兼容性/功能评估没有完成；外部地图固定画布，未实现完整图层包、动态效果、动画、宽世界镜头与地图导入工具 |
| 06 Dataset Factory | 合成优先、自动标注、会话/场景隔离，同时建立真实录像测试与适合像素游戏的增强研究 | **部分实现**：legacy/lifecycle/外部地图三类版本、动作轨迹和像素/标签重放核验、每图六阶段、会话划分；v0.4 486 帧已检查。[生命周期数据](DATASET_LIFECYCLE.md) | 缺独立真实标注集、未见地图留出、动画/伤害数字/技能效果等系统遮挡研究；相同地图的新会话不能充当跨地图泛化；增强实验未完整覆盖原表 |
| 07 YOLO 模型平台 | 比较全怪物分类、通用检测、检测加分类三种设计及 Nano/Small/Medium；统一质量指标和 Model Registry | **部分实现；模型准入受限**：真实 CPU 训练、ONNX 导出/推理、七类通用对象评估、地形审查存在；v0.2 混合管线准入，v0.3 地形候选被拒绝。[模型实验](YOLO_LIFECYCLE_EXPERIMENT.md)、[失败实验](YOLO_TERRAIN_EXPERIMENT.md) | 未完成 A/B/C 系统对照、全怪物身份分类、两阶段分类器、多规模比较、完整 mAP50-95 与真实跨图/小目标/遮挡验收；现有权重 manifest 不是完整 Model Registry |
| 08 Runtime 感知 | Capture、YOLO/OpenCV/OCR、Tracker、StateEstimator、WorldState、BeliefState；每字段时间/来源/置信度/有效期 | **部分实现**：RGB 观测、像素/ONNX 检测、合成 HUD/地图/对话解析、本地录像及 OCR、新鲜度监督可用。[视觉](VISION_IMPLEMENTATION.md)、[录像](RECORDINGS.md) | 没有完整多帧关联、字段级融合与失效、完整 World/BeliefState；真实 HUD、小地图、中文聊天/系统消息准确率未验收；没有可把任意真实录像直接变成可用状态的适配层 |
| 09 多地图系统 | WorldMapGraph 与 PlatformGraph 两层拓扑、MapRegistry/Recognizer/Studio/Validator，导航与职业能力分离 | **部分实现**：外部 2–8 图单出口循环、静态标记身份、几何检查和三图闭环；地图内容可随存档复验。[v0.4 验收](RELEASE_0_4_ACCEPTANCE.md) | 分支地图图、平台可达性图、路线规划、梯子驱动导航、MapStudio 未实现；固定路线 ID 不是两层图系统；结构合法不等于可解 |
| 10 多职业能力引擎 | Class Package、Capability Compiler、Skill Registry、Action DSL，覆盖各职业及复杂机制 | **部分实现**：标飞、龙骑、牧师三种固定 job；弹药、跳标、范围攻击、治疗和参数注入。[职业配置](CLASS_PROFILES.md) | 外部能力包、技能等级/效果表、编译器与 DSL 未实现；刀飞、冰雷、火毒、弓手等未交付；当前龙骑范围样例消耗 MP，不代表原章提到的 HP 消耗技能已实现；Debuff/召唤/毒雾等未覆盖 |
| 11 完整任务系统 | 独立 Task、仲裁、Planner、Sequencer、Scheduler、生命周期/取消/暂停/抢占；多类业务任务 | **部分实现，运行链简化**：有限状态机顺序战斗/拾取/NPC/切图，插入生存与 Buff 优先处理；一次输出一个 `SimAction`。[策略](../src/maple_vision/agent.py) | 尚无通用 Task → Plan → Intent → Execution → Verification 业务链、独立任务候选/仲裁与冲突调度、配置任务图及复杂背包任务；玩家聊天仅封装给定回复为草稿，不识别聊天或自动生成/发送 |
| 12 Action Verification | 各动作按效果验收，统一 SUCCESS/LIKELY_SUCCESS/UNKNOWN/FAILED/TIMEOUT | **部分实现**：`verify_visual_effect` 记录可见变化；oracle 检查任务真值，动作接受不自动算成功；replay/reinfer 实际复验。[验证合同](MODEL_VALIDATION.md) | 当前可见结果为 observed/pending/unknown/not_applicable，未统一为原章完整语义；P0 Verification 是 NoOp 合同，不能当成已接通的业务层；逐技能、Buff、背包及超时验证还需扩展 |
| 13 独立模拟器 | 固定种子/步长、保存/重置/单步，高速逻辑和视觉运行，物理、角色、怪物、技能、背包等独立机制 | **部分实现，核心样例已验收**：逻辑模拟与 RGB 闭环、平台/梯子、单段跳、战斗、Buff、资源、拾取、NPC、切图和确定性保存恢复。[模拟器实现](SIMULATOR_IMPLEMENTATION.md) | 怪物当前静态、无巡逻/寻路/自动刷新；默认金币掉落、资源计数背包，非装备系统；完整状态效果、怪物/物品包与经核实的游戏物理/技能还缺；没有 Pygame 产品界面 |
| 14 Supervisor 与恢复 | 独立监督、完整异常分类、停止/再观察/分类/恢复/验证/恢复任务，有限重试 | **部分实现**：新鲜度、未知、重复帧、死亡、卡住和攻击预算，以及 running/paused/stopped；四类故障场景有证据。[故障矩阵](../reports/release-0.4.0/external-suite.json) | 未实现原章全部异常状态及通用恢复计划，也无完整 UI 阻塞/资源故障/地图丢失语料；预期停止可算故障处理通过，不能算恢复成功或业务完成 |
| 15 Validation Lab | 可独立复现的 Failure Bundle、录像/真值/状态/任务/动作/版本报告，以及三种运行模式边界 | **部分实现，复现核心已验收**：动作轨迹、初末状态、采样 PNG、摘要、哈希与内嵌内容；实际 N+1 帧重推理；失败轮次保留。[API 合同](API_CONTRACTS.md) | 未完全采用原章逐层文件和 MP4 的标准失败包；真实 Golden Recording 与 sim-to-real 集缺失；现有本地录像观察不是完整实时 READ_ONLY 会话产品 |
| 16 AI Engineering Harness | 编程 AI、Git、机器合同、独立审查、CI、失败修复、发布门禁，允许逐步增加编排 | **基础流程已实施**：`AGENTS.md`、阶段合同、版本报告、独立审查、失败日志、跨平台 CI 与 PR 合并可追溯。[工程目录](../engineering)、[PR #5](https://github.com/www1350/maple-vision-platform/pull/5) | 没有独立运行的通用 Orchestrator 产品；原章允许先用开发流程实现，不能据此宣称已交付自动多 Agent 调度服务 |
| 17 最终项目目录 | apps/data-platform/vision-platform/packages/simulator/content/validation 等完整模块布局 | **部分映射，尚未按原目录拆分**：实际为单 Python 包 `src/maple_vision`、CLI、tests、scripts、engineering、docs、examples/reports | 已有能力可映射到原模块；缺失模块不能靠创建空目录补齐。是否按原结构拆分应通过 ADR 与兼容迁移决定，保持单机模块化 |
| 18 P0–P13 开发计划 | 分阶段依赖与验收；首轮三职业、两地图打通，再扩展通用性 | **首轮主链已完成，后续部分完成**：P0 验收；v0.1 基础链、v0.2 生命周期/模型、v0.3 工具与失败研究、v0.4 外部地图均有记录。[阶段状态](../engineering/milestones.yaml) | P1–P13 原章完整范围仍有缺口；当前增量到了 P13 只表示其发布阶段。后续合同必须写原章映射、输入、输出、失败边界和独立验收 |
| 19 最终验收指标 | 地图/平台/任务/复合技能/拾取/生存/恢复目标、24 小时、零严重未恢复故障、Golden 回归、运行时零 LLM | **部分证据，完整门禁未通过**：工程与限定样例可复现，默认运行无 LLM；本轮实际短时运行约 61 秒。[实际验收](RELEASE_0_4_ACCEPTANCE.md) | 尚未对完整分布定义并测量全部指标、置信区间与独立样本；24 小时未做，真实域未验收；13 个预期场景不能替代所有业务成功率目标 |
| 20 最终产品界面 | 职业/地图/模式/模型选择、任务能力与配置复制的产品界面 | **未实现产品 UI**：目前可用入口为 CLI，文件报告供查看；原页交互原型已保留源码 | FastAPI/Web UI、Map/Dataset Studio、Replay Viewer、Validation Dashboard 尚未交付；原型中的控件不代表实际支持相应职业或模型 |
| 21 Coding AI 总控提示词 | 完整总规格、强制边界、阶段 DAG、测试/审查/修复/发布与首次 P0 任务 | **部分符合，原文已恢复**：仓库已有开发规则、合同、文档与 CI；无运行时 LLM、真值边界和版本证据得到约束 | Task 分层等强制要求仍有实现偏差，不能只因文档写了就判通过；按下节逐项闭合。历史 P0 停止条款由后续连续授权替代，不暂停已获授权工作 |
| 22 五个核心资产与总结 | Game Knowledge Database、Asset Registry、Dataset Factory、Game Simulator、Validation Lab 支撑持续扩展 | **五类基础都有实现，成熟度不同**：本地目录/素材、合成数据、模拟器和验证可运行 | 真实资料覆盖、动画、通用内容合同、完整感知任务与真实域验证仍缺；当前是受限研究基础，尚非完整 v8 平台 |

## 4. 必须显式处理的架构偏差

### A. 业务运行链尚未遵守完整任务分层

原要求是 Task → Plan → Intent → Execution → Verification。实际业务链从 `VisualAgent.decide(VisualObservation)` 直接产出包含 `SimAction` 的 `AgentDecision`，`lab.py` 驱动 `Simulator.step` 后记录效果与 oracle。原 P0 的 Intent/Execution/Verification 及 WorldState 仍是独立 NoOp 基础，不能把两条链的类型名称拼起来宣称原架构已完成。

补齐时需先冻结版本化 Task/Plan/Intent、动作冲突与时间合同，再迁移任务和职业能力；保留旧证据读取与确定性重放。现有策略只读观测、控制只接动作的边界应继续保持，不能为实现 Planner 读取动态 Ground Truth。

### B. 数据底座与资产索引采取了精简存储

通用 catalog 可保存版本化 payload，但不等同原章的 skills/skill_levels/monster_drops/map_connections/quest_requirements 等领域表及关系校验。当前三职业参数映射只消费明确核实的合成范围；不能把尚未映射的真实参考条目直接转为运行规则。素材索引目前也存为文件，尚未实现原计划的 SQLite 索引。

后续应选择显式迁移或经审查的等价领域 schema，定义实体引用、技能等级、版本一致性和冲突处理，再接运行规则。该选择属于架构决策，不能静默删减原章要求。

### C. 模拟器与产品界面的实现形态不同

现有 Python 模拟逻辑和 Pillow 渲染可无窗口运行；OpenCV 用于可选录像导入等能力。Tesseract 符合原方案“或其他本地 OCR”的选项，但中文游戏小字仍未验收。Pygame CE、FastAPI、Web UI、MapStudio 等尚未开发。

固定画布几何与静态 sprite 是当前内容子集，不代表完整镜头/动画/技能特效系统。产品层应复用确定性内核，而不是为了原型截图重新写一套规则。

### D. 七类通用检测不等于完整模型比较

当前类别为 player、monster、loot、portal、npc、platform、ladder。没有细分全部怪物身份，也没有完成 Detector + Classifier 分离或 Nano/Small/Medium 的统一对照。现有单 IoU AP 与几何指标不能替代原章全部 mAP50-95、遮挡、小目标和跨地图指标。

实验中可先导出 ONNX 以评估实际运行模型；这种导出不是原章所要求的模型准入或 Registry 发布。v0.2 的混合模型只在其合成范围通过；v0.3 单候选平台召回 472/576，低于 98% 门槛，已明确失败。其 test 和推广分支按原协议关闭，不重训或修改门槛掩盖失败。下一研究须另立候选、数据、评价及停止合同。

### E. 状态、职业、地图和回复都尚非通用实现

`VisualObservation` 和原 P0 WorldState 不能替代带字段级有效期的完整状态融合。三种固定 `ClassSpec` 可调参数也不能替代能力编译器。循环路线与静态地图标记不能替代地图图/平台图和真实 MapRecognizer。

模拟器 NPC 对话可运行；`chat_draft(incoming, reply)` 只封装调用方给定回复，既无聊天理解、自动草稿生成，也无发送后端。后续草稿能力仍须保留人工确认边界。怪物当前不自动刷新，当前清场式任务不是持续刷怪机制。

### F. 验证和发布必须沿用各自的证据范围

文件哈希证明内容一致，不能证明来源许可或官方规则正确；纯接口隔离也不是进程权限沙箱。固定录像能检验既有画面的感知，无法通过更换动作改变画面，不能验收反事实技能或药品效果。真实领域需独立录像和标注，长稳需实际墙钟。

当前证据包已有可靠重放核心，但尚未完全实现原章 Failure Bundle 各层文件与 MP4。工程测试通过、模型准入、真实域通过、24 小时通过必须分列。

## 5. P0–P13 的当前执行视图

| 阶段 | 已完成的可用基础 | 按原总纲剩余的主要交付 |
|---|---|---|
| P0 | 工程、接口、NoOp、SQLite 迁移、CLI、CI | 维护兼容，不重复开工 |
| P1 | 本地目录、快照、差异计划和参数加载 | 获准网络源、调度、领域规则与完整数据 |
| P2 | 静态素材登记与 SpritePack | 动画、锚点、效果及实体资源关联 |
| P3 | 分层渲染、有限镜头、外部地图 JSON | 宽世界、完整图层包、地图工具评估/导入 |
| P4 | 合成生命周期数据与轨迹校验 | 真实留出、跨地图、遮挡/动画增强实验 |
| P5 | 确定性物理与三职业样例机制 | 怪物动态、复杂状态/物品、规则校准 |
| P6 | 真训练/推理/导出/评估和限定模型准入 | 三模型设计、多规模、完整指标与真实域 |
| P7 | 帧观测、合成 HUD、OCR 适配和监督 | Tracker、StateEstimator、WorldState/BeliefState |
| P8 | 三职业静态配置和动作差异 | 能力编译器、技能/复合动作 DSL、全职业包 |
| P9 | 顺序任务、动作和独立效果核验 | Task/Plan/Intent 分层、仲裁、调度、通用导航 |
| P10 | 有限故障检查、暂停/停止与恢复预算 | 完整异常分类、恢复计划与证据 |
| P11 | 动作/感知重执行、故障矩阵、录像导入 | 完整失败包、真实 Golden、随机复杂场景 |
| P12 | 三职业及外部地图矩阵 | 通用内容插件、地图/平台图、产品管理界面 |
| P13 | 安装包、跨平台 CI、短时资源测量 | 固定配置的实际 24 小时与完整发布门禁 |

阶段表反映整个原方案；某增量只需验收其中相关子集。不要把版本中的 P13 发布活动当作整张表已经完成。

## 6. 已关闭的 v0.4 交付与仍开放的总体验收

[PR #5](https://github.com/www1350/maple-vision-platform/pull/5) 的最终 head `92e673a92c6dd8196a3393d943a78912084631fa` 已通过 [CI 45456607613](https://github.com/www1350/maple-vision-platform/actions/runs/45456607613)，并合并为本文开头的基线提交。最终 head 与合并提交的 tree 均为 `a93041c326b126d3e2203862a023909e0af282a4`；此版本交付已经闭合。

- 本地全量 1011 通过、0 失败/跳过；最终 Ubuntu 1004 通过/7 跳过、Windows 1000 通过/11 跳过、原生 155 通过/0 跳过。计数有重叠，不能相加。
- 三职业三图任务、13 个预期正常/故障结果、486 帧数据已验证。13 个结果包含预期停止/失败，不代表 13 次业务任务全完成。
- 本轮新地图使用像素基线，val 有 40 个误检和 12 个漏检；没有评估 YOLO 在新地图上的准确率或泛化。
- 请求 60 秒，实际 61.050688 秒、53 轮；完整 24 小时门禁仍开放。完整 v8 与真实域均未验收。

旧 `milestones.yaml` 的 `pending_gates: [final_archive_head_ci]` 及文档中的归档待检查语句，是该提交形成时的历史状态；最终 PR 回执已关闭该门禁。不能以此重开 v0.4 或不断生成记录自身 SHA 的归档提交；也不能因此关闭整个原方案的剩余门禁。

## 7. 后续编程 AI 的工作队列

以下为后续工作包提案，尚未实现，执行前须按依赖冻结合同与验收分母。既有用户授权允许继续推进不依赖外部资料的工程；缺真实资料不应成为全部工作的停止条件。

| 工作包 | 对应原章 | 具体交付与独立完成条件 |
|---|---|---|
| 数据与素材接入 | 03–06 | 取得获准导出/接口合同，保存原始来源；实现领域 schema、版本迁移和同步调度；动画/锚点/技能效果进入渲染并验证标签。明确区分自绘测试内容与真实资料 |
| 状态与任务合同 | 01、08、11、12、14 | 版本化字段有效期、Tracker/融合，以及 Task/Plan/Intent/Execution/Verification；加入抢占、取消、资源冲突、过期意图、部分成功和有限恢复反例；保留旧回放兼容 |
| 图与能力系统 | 09、10、13 | 世界图/平台图与能力分离；定义可达性和梯子导航任务；外部职业/技能包经编译校验执行；用新职业和分支地图证明扩展无需复制 Runtime |
| 视觉研究与真实评估 | 06–08、19 | 先建立独立会话/地图留出与真实标注，再冻结模型设计、预算和门槛；系统对比 A/B/C 与规模，记录失败、置信区间和部署成本；不重用已查看 test 选模 |
| 验证产品与界面 | 15–17、20 | 标准失败包、精确录像时序、内容/模型管理、Replay Viewer 和 Dashboard；UI 读取真实后端状态，禁用未实现选项；不能把原页原型直接当成程序 |
| 最终发布 | 18、19、21 | 固定提交、依赖、数据、模型和硬件；独立指标达到合同；真实运行 86400 秒并保留资源趋势及原始失败；通过后才能声明对应范围发布 |

每项任务应列：原章编号、当前接口、允许修改的模块、schema 迁移、成功/失败反例、实际命令、证据目录和未完成限制。完成后更新本对照，不改原文目标来迁就已有实现，不降低测试或用合成成绩替代真实域。

## 8. 接手入口

- 设计基线：[原始 MASTER_SPEC](../MASTER_SPEC.md)；实现使用：[FINAL_PLAN](FINAL_PLAN.md)、[API 合同](API_CONTRACTS.md)、[数据格式](DATA_SCHEMAS.md)。
- 当前结果：[v0.4 验收](RELEASE_0_4_ACCEPTANCE.md)、[完整 PR 回执](https://github.com/www1350/maple-vision-platform/pull/5)；历史模型：[v0.2 验收](RELEASE_0_2_ACCEPTANCE.md)、[v0.3 失败与工程验收](RELEASE_0_3_ACCEPTANCE.md)。
- 执行约束：[AGENTS.md](../AGENTS.md)、[阶段计划](IMPLEMENTATION_PLAN.md)、[Milestone DAG](../engineering/milestones.yaml)。

当前交付的价值是可运行、可审查、可复现的离线基础。达到原始完整总纲，还必须逐项补齐上述架构和产品能力，并取得对应的独立数据与运行证据。
