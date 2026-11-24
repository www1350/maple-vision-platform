# P0 接口合同

## 1. 已实现的数据模型

模型在 `src/maple_vision/contracts.py` 中，使用不可变 dataclass。以下是 P0 的实际接口，不是完整玩法模型。

| 模型 | 核心字段 | 约束 |
|---|---|---|
| Frame | sequence、timestamp_ms、pixels | 序号/时间为非负整数；像素是不可变 bytes |
| Detection | class_id、bbox、confidence | xyxy 有正面积、非负原点、有限值；置信度在 [0,1] |
| Observation | sequence、timestamp_ms、status、confidence、source_id、detections | 状态显式；Detection 采用 tuple |
| WorldState | sequence、timestamp_ms、status、confidence、source_id | P0 只传递已知性与来源，不生成虚构玩家坐标 |
| Intent | intent_id、source_sequence、created_at_ms、deadline_ms、action、reason | 截止时间晚于创建；P0 Action 只有 NOOP |
| Execution | intent_id、timestamp_ms、status、reason | SKIPPED 或 REJECTED |
| Verification | intent_id、timestamp_ms、status、reason | NOT_APPLICABLE、UNKNOWN、FAILED、TIMEOUT |
| TickResult | observation、world、intent、execution、verification | 同一次处理的完整轨迹 |

序号是单个 source 实例的序号；时间戳是同一 source 时钟域的毫秒值。不得把录像时间和系统墙上时间混用。一个管线实例内序号严格递增，帧时间和处理时间不倒退；重新开始会话时新建管线实例。

P0 的 `pixels` 是不解释内容的 opaque bytes；还没有图片解码器，不能把 `DemoSource` 当成真实 RGB 采集器或视觉模型。

`confidence` 是观测元数据，不是经过校准的概率。P0 默认观测为 UNKNOWN/0.0。既有 `Status(str, Enum)` 的字符串行为保持兼容。

## 2. 端口与依赖方向

```python
FrameSource.read() -> Frame | None
Perception.observe(frame: Frame) -> Observation
StateEstimator.estimate(observation: Observation, now_ms: int) -> WorldState
Policy.decide(world: WorldState, now_ms: int) -> Intent
Controller.execute(intent: Intent, now_ms: int) -> Execution
Verifier.verify(intent: Intent, execution: Execution) -> Verification
```

策略不读取图片，执行器不读取 Detection。NoOp 验证器没有实际环境效果可验证，因此跳过的动作记录为 NOT_APPLICABLE，不能记为成功率样本。

`ports.py` 声明 KnowledgeProvider、AssetImporter、ObjectDetector 及记录类型；`contracts.py` 保留 Simulator 协议。它们在 P0 没有网络、素材、推理或游戏物理实现。知识和素材记录是扩展合同；P1/P2 必须在各自边界实现来源、哈希和访问许可验证。

## 3. 管线与时间

- 原 `run_once(source, perception)` 保留原返回语义，只执行观测。
- 新 `Pipeline.tick(now_ms=None)` 返回 TickResult；source 结束时返回 None，不再调用后续模块。
- `now_ms=None` 使用当前帧时间，供确定性演示；提供 now_ms 时必须与帧处于同一时钟域。
- `FreshnessEstimator(max_age_ms=250)` 在观测年龄大于上限时输出 STALE；等于上限仍有效。未来时间戳会被拒绝。
- 管线在控制器之前校验帧、观测、世界状态、意图的序号和时间关联。回执与验证结果也必须关联同一意图。
- 处理时间倒退会被拒绝，防止过期数据重新变新。
- NoOp 意图有效区间为 `[created_at_ms, deadline_ms)`；在截止时间执行会被拒绝，验证为 TIMEOUT。
- NoOp 意图 ID 在一次运行内唯一，跨运行用 JSONL 的 run_id 区分。

## 4. CLI 与配置

```text
maple-vision demo [--config FILE] [--frames N] [--output FILE]
maple-vision db init PATH
maple-vision db check PATH
```

`python -m maple_vision` 提供相同命令。成功退出码 0，参数、配置、文件或数据库失败退出码 2。成功时 stdout 是一份 JSON 摘要，错误在 stderr；argparse 的 help 文本除外。

TOML 根字段只允许 `schema_version` 与 `runtime`，要求 schema_version=1。runtime 只允许 mode、frames、frame_interval_ms、max_observation_age_ms。整数约束分别为 frames:1–10000、间隔:1–60000ms、最大观测年龄:0–60000ms；bool 不接受为整数。

JSONL 每行含 schema_version、run_id 和 tick，不保存原始帧。现阶段是 trace writer，并非完整 Recorder、Replay 解码器或 Failure Bundle 系统。

## 5. 数据库迁移合同

```python
migrate(connection: sqlite3.Connection) -> None
initialize(path: str | Path) -> None
open_readonly(path: str | Path) -> sqlite3.Connection
```

迁移版本 0→1 或验证既有 v1。SAVEPOINT 将 DDL 和 user_version 更新组成原子操作；外层已有事务时，迁移不替调用方 commit/rollback。调用方可以继续回滚自己的事务及迁移内容。

已有 `source_records` 采用保守的规范化 DDL 比较，允许空白及关键词大小写差异，保留原项目合法 v1 Schema；不自动接受任意“看起来类似”的表。未知版本、结构冲突返回 ValueError，原数据不修改。

open_readonly 不创建数据库，使用 SQLite URI mode=ro 和 query_only；调用方负责 close。P0 保留原 v1 中 `id TEXT PRIMARY KEY` 的行为；P1 建立写入功能前，需要显式迁移补齐 NOT NULL 等数据约束，不能假定当前 Schema 已满足完整知识库需求。

## 6. 兼容性和下一阶段

保留 Frame 的前三个构造参数、Observation 的前三个构造参数、run_once、migrate 和 Simulator 原接口。新增字段有默认值。新输入校验明确拒绝原先不应该有效的负时间、无穷值和错误枚举。

P0 尚未冻结未来地图、技能、玩法及观察融合的数据结构；它们应在后续阶段采用显式版本迁移扩展，不能为了维持文档中的理想结构而伪造实现。
