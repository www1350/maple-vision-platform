# Maple Vision Platform v8.0 — 最终工程实施总纲

这次将之前所有方案，以及群友提供的实际技术经验统一收敛。后续以此为唯一 Master Spec，不再反复重构架构。

相比最初的方案，最大的改变是：**不再从编写脚本开始，而是从游戏数据、静态资源、场景渲染、自动训练和独立验证开始。**

最终构建一个多职业、多地图、知识驱动的视觉 Agent 研究平台，支持复杂技能、交互、战斗、资源管理，并能够由 Coding AI 独立开发和验证。执行范围为自建模拟器或明确授权的测试环境，不依赖正式游戏账号进行自动化测试，也不包含反作弊绕过或隐蔽控制。

---

# 01\. 最终总体架构

**冒险岛小册子**

职业、技能、地图、怪物、掉落、道具

**静态资源**

动画帧、贴图、地图结构、技能特效

**真实录像**

机制校验、视觉测试、真实场景

**Game Data Foundation**

SQLite · Asset Registry · 版本管理 · 数据校验

**Visual Agent Runtime**

Capture → YOLO / OpenCV / OCR

Tracking → WorldState → BeliefState

Task Arbitration → Planner → Skill Engine

Action Sequencer → Scheduler → Intent

**Controlled Execution**

Simulator Controller → Observation → Verification → Recovery

**Game Simulator**

独立物理、技能、怪物、资源、渲染

**Validation Lab**

Replay、故障注入、回归、长稳测试

**AI Engineering Harness**

任务拆解 → 开发 → 测试 → 失败归因 → 修复 → 审查 → 发布

---

# 02\. 技术选型

尽量简单，不做不必要的分布式架构。

| 模块 | 技术选型 |
|--------|--------|
| 主语言 | Python 3.12 |
| 游戏知识数据库 | SQLite |
| 静态资源存储 | 本地文件系统 |
| 数据同步 | HTTPX / 获准的数据访问接口 |
| 图片处理 | OpenCV、Pillow |
| 目标检测 | Ultralytics YOLO |
| 模型推理 | ONNX Runtime |
| 文字识别 | PaddleOCR 或其他本地 OCR |
| 地图研究工具 | MapleNecrocer 候选评估 |
| 模拟器 | Pygame CE |
| 地图图结构 | NetworkX / 自定义 Graph |
| 数据协议 | Pydantic |
| 配置 | YAML |
| 测试 | pytest、Hypothesis |
| 日志 | JSONL |
| 录像 | MP4 |
| 管理界面 | FastAPI + 简单 Web UI |
| AI 开发 | Codex / Claude Code |
| Runtime LLM Token | 0 |

**不需要 Redis、Kafka、微服务、向量数据库。**

第一版采用单机模块化架构即可。

---

# 03\. Game Data Foundation：游戏数据底座

## 3.1 数据来源

| 数据来源 | 负责解决的问题 |
|--------|--------|
| 小册子 | 游戏里有什么、属性是什么 |
| 静态资源 | 游戏对象长什么样 |
| 地图工具 | 地图素材如何组织 |
| 真实录像 | 游戏实际表现是什么 |
| 独立模拟器 | Agent 行为是否正确 |

这几种数据不能互相替代。

## 3.2 小册子数据同步

url冒险岛怀旧服小册子https://mxdc.dvg.cn/

提前同步相对静态的数据，存储到本地数据库。

```text
Knowledge Sync
    ↓
Raw Archive
    ↓
Normalizer
    ↓
Schema Validator
    ↓
Version Diff
    ↓
Knowledge Review
    ↓
SQLite Snapshot
```

同步范围：

| 数据 | 更新策略 |
|--------|--------|
| 职业、转职关系 | 版本更新 |
| 技能、技能等级 | 版本更新 |
| 怪物属性 | 版本更新 |
| 怪物掉落 | 版本校验 |
| 地图、地图连接 | 低频更新 |
| 装备、道具 | 低频更新 |
| 药水、飞镖 | 低频更新 |
| NPC、任务 | 低频更新 |
| 市场价格 | 可选独立缓存 |

访问方式必须先确认站方许可和实际接口，不假设存在公开 API，不绕过微信登录、付费或访问限制。

## 3.3 数据库设计

```text
game_knowledge.db

├── game_versions
├── source_records
├── sync_jobs
├── knowledge_snapshots
│
├── classes
├── advancements
├── skills
├── skill_levels
├── skill_effects
│
├── monsters
├── monster_drops
│
├── maps
├── map_connections
├── map_platforms
├── map_portals
│
├── items
├── equipment
├── consumables
│
├── npcs
├── quests
├── quest_requirements
│
├── asset_index
├── mechanics_rules
└── validation_evidence
```

每条数据记录：

```text
entity_id
game_version
source_url
source_type
fetched_at
payload_hash
confidence
verification_status
```

验证状态：

```text
VERIFIED
PROVISIONAL
UNKNOWN
CONFLICTED
```

小册子数据与实际游戏存在差异时，必须保留冲突记录，不允许直接覆盖。

## 3.4 Runtime 数据访问

采用：

```text
Local-first
Static-first
Versioned
Evidence-backed
```

运行时优先查询本地 SQLite。

外部网站查询作为异步、可选的知识补充机制，不得阻塞实时感知和决策。新取得的数据必须先验证，再进入下一个知识快照。

---

# 04\. Asset Registry：静态资源平台

结合群友的经验，资源管理必须成为独立模块。

```text
assets/
├── monsters/
├── characters/
├── skills/
├── items/
├── maps/
├── backgrounds/
├── npcs/
└── ui/
```

管理合法取得、获准使用的素材：

| 资源 | 用途 |
|--------|--------|
| 怪物动画帧 | 视觉模型研究 |
| 角色动画 | 动作状态研究 |
| 技能特效 | 遮挡鲁棒性测试 |
| 地图贴图 | 场景渲染 |
| 背景图层 | 镜头与视差研究 |
| 道具图标 | 物品识别 |
| UI 素材 | 状态识别 |
| NPC 素材 | 对话交互研究 |

图片和动画存储在文件系统，SQLite 保存资源索引。

静态资源与小册子数据通过实体 ID 和版本关联。

```text
Monster Entity
├── Monster ID
├── Knowledge Record
├── Animation Assets
├── Reference Samples
└── Validation Evidence
```

---

# 05\. 地图渲染平台

这是最新聊天截图中最值得吸收的部分。

群友提到，解包得到的背景不是原地图大小，而是跟随镜头变化的，因此直接拼贴可能出现问题。

这意味着我们必须把地图的**逻辑结构与视觉渲染分离**。

概念性图层模型，用于独立模拟器和视觉研究，并非已确认的国服客户端内部渲染顺序。

地图平台需要管理：

```text
MapDefinition
CameraModel
CoordinateSystem
BackgroundLayers
TerrainLayers
DecorationLayers
EntityLayers
EffectLayers
```

## 5.1 MapleNecrocer 工具

根据群友截图，候选工具名称为：

```text
MapleNecrocer_20260423
```

据他的描述，该工具能够编辑部分旧版 WZ 格式地图，并辅助导出纯净背景。

应当优先评估已有工具，避免重复开发。但其实际版本兼容性、导出功能、许可和安全性仍需要独立确认，不能直接假定兼容当前国服。

## 5.2 地图内容包

```text
map_packages/
└── map_001/
    ├── manifest.yaml
    ├── definition.yaml
    ├── geometry.json
    ├── platforms.json
    ├── portals.json
    ├── layers.json
    ├── signatures/
    ├── assets/
    └── validation/
```

地图背景、物理碰撞结构、导航关系必须分别维护。

---

# 06\. Dataset Factory：自动数据集工厂

不再将人工录屏标注作为唯一训练方式。

采用：

**Synthetic-first + Real-world Validation**

image_group{"query":["Maplestory classic slime mushroom pig monster sprite sheet transparent pixel art"],"aspect_ratio":"1:1","layout":"carousel","num_per_query":1}

怪物动画

image_group{"query":["Maplestory classic Henesys hunting ground empty forest map background platforms"],"aspect_ratio":"1:1","layout":"carousel","num_per_query":1}

地图素材

image_group{"query":["Maplestory classic poison mist fire poison mage skill effect"],"aspect_ratio":"1:1","layout":"carousel","num_per_query":1}

技能特效

**Scene Dataset Factory**

授权素材 · 独立场景 · 自动标注 · 场景一致性

**合成训练集**

自动生产标签

**真实录像测试集**

独立人工核验

**Dataset Validator → Training → Evaluation**

## 6.1 吸收群友的训练经验

群友提出怪物动画需要与地图场景对应，不能只是把孤立的怪物帧交给 YOLO。

其本质是减少合成训练数据与实际画面的分布差异。

但也需要防止模型过度依赖特定地图背景，导致换地图后识别失败。因此必须保留跨地图、跨背景和未见场景测试。

## 6.2 数据增强原则

| 增强方式 | 默认策略 |
|--------|--------|
| 大幅透视 | 关闭 |
| 大幅旋转 | 关闭 |
| 垂直翻转 | 关闭 |
| 强烈光照变化 | 关闭 |
| 强烈色彩变化 | 关闭 |
| 水平翻转 | 根据素材验证 |
| 尺寸变化 | 根据实际分布验证 |
| 动画变化 | 纳入测试 |
| 技能遮挡 | 纳入测试 |
| 怪物重叠 | 纳入测试 |
| 伤害数字 | 纳入测试 |
| 背景变化 | 纳入测试 |
| 镜头变化 | 纳入测试 |

不能直接照搬自然图像的默认训练配置。

## 6.3 数据集版本管理

```text
datasets/
└── dataset_v001/
    ├── images/
    ├── labels/
    ├── dataset.yaml
    ├── manifest.json
    ├── source_records.json
    └── quality_report.json
```

训练集、验证集、测试集按会话和场景隔离，避免相邻录像帧造成数据泄漏。

---

# 07\. YOLO 模型平台

群友认为 YOLOv8n 底座太小，不能很好地覆盖全部怪物图鉴。

这个判断不能直接作为技术结论，必须通过 Benchmark 验证。

## 7.1 比较三种模型设计

**A. 全怪物分类模型**

每种怪物作为独立类别。

**B. 通用目标检测模型**

检测通用对象，不强制区分具体怪物。

**C. 检测与分类分离**

**Detector**

位置

**Classifier**

身份

**Knowledge**

资料

分别评估目标检测和细粒度识别。

## 7.2 模型规模

统一评估 Nano、Small、Medium，必要时使用更大模型作为离线准确率对照。

不要在架构层绑定 YOLOv8n，也不要仅因为有更新的 YOLO 就直接替换模型。

## 7.3 视觉鲁棒性验证

独立测试不同职业技能、怪物重叠和视觉遮挡造成的影响。

评估指标：

```text
Precision
Recall
mAP50
mAP50-95
Small Object Recall
Occlusion Recall
Cross-map Recall
Inference Latency
Memory Usage
```

模型通过独立测试后才能导出 ONNX 并进入 Model Registry。

---

# 08\. Runtime 感知系统

```text
Capture
   │
   ├── YOLO
   ├── OpenCV
   ├── OCR
   └── MapRecognizer
           ↓
       Observation
           ↓
         Tracker
           ↓
      StateEstimator
           ↓
        WorldState
           ↓
        BeliefState
```

| 技术 | 职责 |
|--------|--------|
| YOLO | 动态对象检测 |
| OpenCV | 小地图、HP/MP、固定 UI |
| OCR | 聊天、NPC 对话、系统消息 |
| Tracker | 多帧对象关联 |
| StateEstimator | 状态融合 |
| BeliefState | 管理未知和不确定状态 |

核心领域模型：

```text
WorldState
├── PlayerState
├── MonsterState[]
├── LootState[]
├── ResourceState
├── InventoryState
├── SkillState
├── BuffState
├── MapState
├── NavigationState
├── InteractionState
└── UIState
```

所有字段需要时间戳、来源、置信度和有效期。

视觉识别失败不能直接解释为目标死亡、物品拾取成功或技能释放成功。

---

# 09\. 多地图系统

地图采用两层结构：

上层是地图之间的连接，下层是单张地图的局部拓扑。

```text
WorldMapGraph
PlatformGraph
MapRegistry
MapRecognizer
MapStudio
GraphValidator
```

地图几何结构与职业移动能力分开管理。新增地图主要增加地图包和测试，不复制 Runtime。

---

# 10\. 多职业能力引擎

职业不能写成一套套独立脚本。

```text
Class Package
      ↓
Capability Compiler
      ↓
Skill Registry
      ↓
Action DSL
      ↓
Task Engine
```

| 职业 | 需要支持的机制 |
|--------|--------|
| 标飞 | 飞镖、远程攻击、跳标 |
| 刀飞 | 近战、组合技能 |
| 龙骑士 | 范围攻击、HP 消耗 |
| 其他战士 | 近战、Buff、职业技能 |
| 冰雷 | 范围魔法、控制 |
| 火毒 | 持续伤害、毒雾 |
| 牧师 | 治疗、辅助、Buff |
| 弓箭手 | 远程攻击、弹药 |
| 其他职业 | 能力插件扩展 |

实际技能机制以对应版本的资料和验证结果为准。

职业包：

```text
class_packages/
└── class_id/
    ├── manifest.yaml
    ├── advancement.yaml
    ├── skills.yaml
    ├── combos.yaml
    ├── resources.yaml
    ├── mechanics.yaml
    └── tests/
```

统一能力包括移动、跳跃、方向攻击、范围攻击、空中攻击、治疗、Buff、Debuff、召唤和资源消耗。

---

# 11\. 完整任务系统

所有业务行为统一纳入 Task Engine。

| Task | 职责 |
|--------|--------|
| SurvivalTask | HP/MP、吃药 |
| CombatTask | 战斗与职业技能 |
| BuffTask | Buff 状态管理 |
| LootTask | 掉落与拾取 |
| NavigationTask | 地图内导航 |
| MapTransitionTask | 地图切换 |
| InventoryTask | 背包与资源 |
| InteractionTask | NPC、对话 |
| ChatTask | 消息识别、回复草稿 |
| RecoveryTask | 异常恢复 |
| IdleTask | 等待 |

真实玩家的聊天回复保留人工确认，模拟器可以测试完整对话交互。

## 11.1 Task Arbitration

```text
Safety Gate
     ↓
Task Candidates
     ↓
Task Arbitration
     ↓
Task Instance
     ↓
Planner
     ↓
Action Sequencer
     ↓
Verification
```

任务需要生命周期、超时、取消、暂停、失败次数和抢占规则。

## 11.2 Action Engine

支持普通动作、复合动作、技能和交互。

例如跳标属于复合动作，而不是独立的普通攻击指令。其前置条件、动作阶段、效果验证由模拟器中的职业能力规则定义。

`ActionScheduler` 负责动作冲突管理，防止移动、技能、拾取、吃药和对话任务同时产生互相冲突的执行请求。

---

# 12\. Action Verification：动作验证

所有操作必须验证结果。

| 动作 | 验证依据 |
|--------|--------|
| 移动 | 位置变化 |
| 跳跃 | 空中状态 |
| 攻击 | 攻击事件及战斗进展 |
| 技能 | 技能效果 |
| Buff | 状态效果变化 |
| 吃药 | 资源恢复 |
| 拾取 | 掉落与背包变化 |
| 传送 | 地图身份变化 |
| 对话 | 对话状态变化 |

结果统一为：

```text
SUCCESS
LIKELY_SUCCESS
UNKNOWN
FAILED
TIMEOUT
```

**发出了动作，不代表动作成功。**

---

# 13\. 独立游戏模拟器

这是实现独立验证的基础。

模拟器产品示意：独立渲染画面、平台、怪物、掉落、HP/MP 和小地图。

模拟器实现：

```text
Simulation Clock
Physics Engine
Character System
Monster System
Combat Resolver
Skill Effect System
Status Effect System
Loot System
Inventory System
Resource System
Map System
NPC System
Dialogue System
Renderer
Ground Truth Exporter
```

支持两种模式：

| 模式 | 用途 |
|--------|--------|
| Fast Simulation | 大规模逻辑测试 |
| Visual Simulation | YOLO/OpenCV 完整视觉闭环 |

Agent 不能直接读取模拟器 Ground Truth。只有独立验证系统能够读取，用来判断 Agent 的识别和行为是否正确。

模拟器必须支持固定随机种子、固定时间步、暂停、单步、保存状态和场景重置。

---

# 14\. Supervisor 与异常恢复

Supervisor 独立于正常任务逻辑。

监控状态：

```text
NORMAL
PERCEPTION_DEGRADED
LOST
STUCK
NO_PROGRESS
ACTION_FAILURE
RESOURCE_FAILURE
MAP_UNKNOWN
UI_BLOCKED
RECOVERING
FATAL
```

处理原则：

```text
Failure
   ↓
Stop Current Task
   ↓
Reobserve
   ↓
Classify
   ↓
Recovery Plan
   ↓
Verify Recovery
   ↓
Resume or Stop
```

不能无限重复失败动作。

对于无法确认的状态，宁可暂停并生成报告，也不要假定系统正常。

---

# 15\. Validation Lab：独立验证平台

## 15.1 自动失败报告

```text
failures/F-000001/
├── replay.mp4
├── ground_truth.jsonl
├── observations.jsonl
├── world_states.jsonl
├── task_trace.jsonl
├── action_trace.jsonl
├── verification.jsonl
├── supervisor.jsonl
├── versions.lock
├── config.yaml
└── report.md
```

每个失败都应当可以独立复现。

Coding AI 修复一个问题之后，必须将该问题加入永久回归测试。

## 15.2 账号风险隔离

| 运行模式 | 权限 |
|--------|--------|
| REPLAY | 只读录像 |
| SIMULATOR | 完整模拟器执行 |
| READ\_ONLY | 合规的画面观察 |

正式游戏不加载自动控制器。

本地验证只能证明功能正确性、视觉兼容性和工程稳定性，不能证明正式服务器自动化操作不会被封号。

---

# 16\. AI Engineering Harness

这部分是让你的 Coding AI 能够独立完成项目的关键。

**Project Orchestrator**

需求拆解、依赖管理、进度控制

**Review Agent + CI**

**PASS**

合并、发布

**FAIL**

失败包、复现、修复

不需要一开始就开发复杂的多 Agent 调度平台。可以先用 Coding AI、Git、CI 和机器可读任务合同完成流程，成熟后再引入并行开发与自动修复。

---

# 17\. 最终项目目录

```text
maple-vision-platform/

├── apps/
│   ├── agent-runtime/
│   ├── game-simulator/
│   ├── map-studio/
│   ├── dataset-studio/
│   ├── replay-viewer/
│   └── validation-dashboard/
│
├── data-platform/
│   ├── knowledge-sync/
│   ├── booklet-provider/
│   ├── knowledge-store/
│   ├── asset-registry/
│   ├── map-importer/
│   └── snapshot-manager/
│
├── vision-platform/
│   ├── scene-renderer/
│   ├── dataset-factory/
│   ├── annotation-export/
│   ├── dataset-validator/
│   ├── training/
│   ├── benchmark/
│   ├── onnx-export/
│   └── model-registry/
│
├── packages/
│   ├── domain/
│   ├── capture/
│   ├── perception/
│   ├── ocr/
│   ├── tracking/
│   ├── state-estimation/
│   ├── belief-model/
│   ├── map-engine/
│   ├── capability-compiler/
│   ├── skill-engine/
│   ├── action-dsl/
│   ├── task-engine/
│   ├── planner/
│   ├── action-sequencer/
│   ├── scheduler/
│   ├── controller/
│   ├── verification/
│   ├── supervisor/
│   └── telemetry/
│
├── simulator/
│   ├── physics/
│   ├── combat/
│   ├── skills/
│   ├── monsters/
│   ├── loot/
│   ├── inventory/
│   ├── resources/
│   ├── interaction/
│   ├── map/
│   └── renderer/
│
├── content/
│   ├── class-packages/
│   ├── map-packages/
│   ├── monster-packages/
│   ├── item-packages/
│   └── knowledge-snapshots/
│
├── validation/
│   ├── unit/
│   ├── integration/
│   ├── golden-replay/
│   ├── scenario-runner/
│   ├── property-testing/
│   ├── fault-injection/
│   ├── differential-testing/
│   ├── sim-to-real/
│   └── soak-testing/
│
├── engineering/
│   ├── milestone-contracts/
│   ├── architecture-decisions/
│   ├── review-rules/
│   └── release-gates/
│
├── assets/
├── datasets/
├── models/
├── recordings/
├── failures/
├── reports/
├── configs/
└── docs/
```

---

# 18\. 从零到完整交付的开发计划

这次重新调整依赖关系，充分利用静态资源和自动训练数据生产。

主要开发依赖示意。数据、模拟器和视觉模型可以部分并行推进。

| 阶段 | 具体交付 | 验收 |
|--------|--------|--------|
| P0 | 仓库、协议、配置、CI | 工程可运行 |
| P1 | 小册子同步、SQLite | 数据可查询、可追溯 |
| P2 | 静态资源注册 | 素材版本关联 |
| P3 | 地图、图层、镜头模型 | 独立场景渲染 |
| P4 | Dataset Factory | 自动生成数据集 |
| P5 | 模拟器物理与机制 | 独立运行 |
| P6 | YOLO 训练与 Benchmark | 模型评估报告 |
| P7 | 感知、Tracking、WorldState | 状态正确性 |
| P8 | 多职业能力、技能、复合动作 | 跨职业兼容 |
| P9 | 任务、交互、调度、验证 | 任务闭环 |
| P10 | Supervisor、Recovery | 异常恢复 |
| P11 | Replay、随机场景、故障注入 | 自动回归 |
| P12 | 多职业、多地图扩展 | 通用性 |
| P13 | 性能、长稳、发布 | 完整验收 |

第一轮选择标飞、龙骑士、牧师三种机制差异明显的职业，以及两张结构不同的测试地图，先打通完整链路。

---

# 19\. 最终验收指标

| 指标 | 建议目标 |
|--------|--------|
| 地图识别准确率 | ≥99% |
| 平台定位准确率 | ≥98% |
| 核心任务成功率 | ≥98% |
| 复合技能成功率 | ≥98% |
| 拾取任务成功率 | ≥95% |
| 生存任务成功率 | ≥99% |
| 异常恢复成功率 | ≥95% |
| 连续模拟运行 | 24小时 |
| 严重未恢复故障 | 0 |
| Golden Regression | 全通过 |
| Runtime LLM 调用 | 0 |

以上为建议验收目标，不是已经取得的测试成绩。必须附带独立测试集、样本量、随机种子、硬件配置和失败记录。

---

# 20\. 最终产品界面

**Maple Vision Platform**

(产品原型)

---

角色职业

- 标飞
- 刀飞
- 龙骑士
- 冰雷
- 火毒
- 牧师
- 弓箭手
当前所选项不可用。

测试地图

- 测试地图 001
- 测试地图 002
- 随机地图
当前所选项不可用。

执行模式

录像回放

高速逻辑仿真

视觉闭环仿真

视觉模型

- Benchmark 最优模型
- Nano
- Small
- Medium
当前所选项不可用。

---

**任务能力**

---

知识来源 本地 SQLite

执行目标 独立模拟器

运行时 Token 0

[复制测试配置]

交互式产品设计原型，并非已经开发完成的程序。

---

# 21\. 直接交给 Coding AI 的总控提示词

将上面整份方案保存为 `MASTER_SPEC.md`，再给 Coding AI 以下指令。

:::writing{variant="standard" id="82951"}
# Maple Vision Platform v8.0 — Master Engineering Mandate

你是本项目的 Principal Architect、Computer Vision Engineer、Game Simulation Engineer、Data Engineer 和 Test Engineering Lead。

完整阅读 MASTER_SPEC.md，并将其作为项目唯一架构总规格。

## 项目目标

从空仓库构建一个参考中国冒险岛怀旧服的多职业、多地图、知识驱动、纯视觉感知、具备独立模拟器和自动化验证能力的 Agent 研究平台。

## 核心要求

1. 从冒险岛小册子等获准使用的数据源提前同步静态游戏知识，建立本地 SQLite 数据库。
2. 建立 Asset Registry，管理合法取得的静态素材、动画、地图和技能特效。
3. 评估 MapleNecrocer 等现有地图工具的许可、兼容性和离线素材处理能力，避免重复建设。
4. 建立地图图层、镜头模型、坐标系统和独立场景渲染能力。
5. 建立 Dataset Factory，支持合成训练数据、自动标注、数据质量校验和独立真实录像测试。
6. 针对像素游戏建立专用数据增强实验，不直接照搬自然图像训练配置。
7. 比较通用检测、细粒度分类和检测分类分离方案，通过 Benchmark 选择模型。
8. 训练框架与 Runtime 分离，模型通过 ONNX 部署。
9. 使用 YOLO、OpenCV、OCR、Tracking、WorldState 和 BeliefState 构建视觉感知体系。
10. 支持多职业能力包、技能 DSL、复合动作 DSL 和统一任务系统。
11. 支持战斗、跳跃攻击、技能、Buff、拾取、吃药、背包、导航、NPC 对话和聊天回复草稿。
12. 建立独立模拟器，支持高速逻辑仿真和完整视觉仿真。
13. 建立 Action Verification、Supervisor、Recovery 和 Session Manager。
14. 建立 Replay、随机场景、故障注入、差异测试、长稳测试和 Failure Bundle。
15. 建立自动化开发、独立测试、失败修复、代码审查和 Release Gate。

## 强制架构约束

- 所有静态知识优先提前同步，运行时优先使用本地只读数据库。
- 不绕过数据源登录、付费、反爬或其他访问限制。
- 所有知识、素材、模型、职业包、地图包必须版本化。
- 未验证数据不得标记为 VERIFIED。
- 合成训练集不得替代独立真实录像测试集。
- Agent 不得读取模拟器 Ground Truth。
- Policy 不得直接读取原始 Frame。
- Controller 不得直接读取 YOLO Detection。
- 所有任务必须经过 Task → Plan → Intent → Execution → Verification。
- Runtime 不依赖 LLM API。
- 所有失败必须能够复现，并生成 Failure Bundle。
- 不得通过删除测试、降低验收标准或修改模拟器规则掩盖实现错误。
- 执行范围仅限自建模拟器或明确授权的测试环境。
- 正式游戏仅作为符合规则的只读视觉研究对象。
- 不实现反作弊绕过、输入伪装、隐藏自动化、实时网络封包拦截、客户端注入或内存操纵。

## 开发方式

不要一次性生成全部代码。

首先建立以下文档：

- MASTER_SPEC.md
- ARCHITECTURE.md
- DATA_SCHEMAS.md
- API_CONTRACTS.md
- KNOWLEDGE_POLICY.md
- ASSET_POLICY.md
- VISION_TRAINING_SPEC.md
- SIMULATOR_SPEC.md
- IMPLEMENTATION_PLAN.md
- TEST_STRATEGY.md
- ACCEPTANCE_CRITERIA.md
- RELEASE_POLICY.md

将 P0–P13 拆解为具有明确依赖关系的 Milestone DAG。

每个 Milestone 必须定义：

- 输入
- 输出
- 依赖
- 实现文件
- 接口合同
- 单元测试
- 集成测试
- 验收指标
- 风险
- 失败处理
- 交付报告

独立任务允许并行开发，但必须遵守模块边界。

每完成一个 Milestone，执行静态检查、单元测试、集成测试和必要的 Replay/Simulator 测试。

测试失败时生成 Failure Bundle，定位问题，修复并增加回归测试。连续修复失败达到上限时停止并提交审查。

## 当前执行任务

只开始 P0。

1. 检查开发环境。
2. 初始化 Git Monorepo。
3. 建立 Python 工程配置和依赖管理。
4. 定义核心数据协议和模块边界。
5. 建立 SQLite Schema Migration 基础。
6. 建立 pytest、Ruff、mypy 和 CI。
7. 创建知识同步、素材导入、视觉模型和模拟器的抽象接口。
8. 实现最小 NoOp 端到端管线。
9. 生成完整 Milestone DAG。
10. 运行测试并输出 P0 验收报告。

P0 验收通过后停止，不提前实现其他阶段。
:::

---

# 22\. 最终方案总结

相比最初的架构，这个版本真正吸收了群友的技术经验。

| 最初方案 | 最终方案 |
|--------|--------|
| 人工录屏训练 | 静态素材与合成数据工厂 |
| 大量人工标注 | 自动标注＋独立核验 |
| 人工绘制地图 | 地图资料导入与结构校验 |
| 简单背景贴图 | 图层、镜头、视差模型 |
| 固定 YOLOv8n | 多模型 Benchmark |
| 单职业脚本 | 多职业能力编译器 |
| 在线查询游戏资料 | 本地 SQLite 知识快照 |
| 人工调试脚本 | Replay＋独立模拟器 |
| 人工发现 Bug | 自动测试与失败归因 |
| AI 只写代码 | AI 开发、测试、修复、审查 |

最终形成完整闭环：

**最终最重要的五个核心资产是：**

**Game Knowledge Database、Asset Registry、Dataset Factory、Game Simulator、Validation Lab。**

这五个基础设施建成以后，增加怪物、地图、职业、技能和测试场景的成本才会真正下降。YOLO 和 Agent Runtime 都可以持续迭代，而不需要每次从头开发。

---

If you want, I can:

- Expand on synthetic data integration in training
- Describe the architecture of the Visual Agent Runtime
- Explain the approach for versioned game data management