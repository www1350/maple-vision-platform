# 带有效期的视觉状态与对象关联

`state_estimation.py` 为原始总纲第 08 章提供有界的时序基础。它消费已经生成的 `VisualObservation`，不读取原始图像、模拟器、隐藏资源、怪物血量或 Ground Truth。已有检测器的准确率不会因加入这一层而自动提高。

## 接口

```python
from maple_vision.state_estimation import StateConfig, TemporalStateEstimator

config = StateConfig()
estimator = TemporalStateEstimator(config)
# observation 来自现有 PixelPerception 或 HybridPerception。
belief = estimator.update(observation, now_ms=observation.timestamp_ms)
assert StateConfig.from_dict(config.to_dict()) == config
record = belief.to_dict()
```

`StateConfig`、`BeliefState`、`FieldBelief` 和 `TrackBelief` 均为不可变对象。配置的 JSON schema 为 1，必须提供完整字段，不接受未知字段、布尔数值、非法版本或非有限值。

| 配置 | 默认值 | 约束 |
|---|---:|---|
| `max_age_ms` | 250 | 0–60000；输入画面可接受的最大年龄 |
| `field_ttl_ms` | 1000 | 1–120000，且严格大于 `max_age_ms` |
| `track_ttl_ms` | 500 | 1–120000，且严格大于 `max_age_ms` |
| `max_tracks` | 128 | 1–512，限制保留对象总数 |
| `max_detections` | 128 | 1–256，不能超过 `max_tracks` |
| `min_confidence` | 0.5 | 0–1；整体/玩家可用性及跟踪候选门槛 |
| `max_match_distance` | 48 | 0.01–8192 像素；预测中心匹配距离上限 |
| `ambiguity_margin` | 0.15 | 0–2；两个候选代价之差不超过此值即存在歧义 |

时间和序号限制在可精确表达的 JSON 整数范围内，并为有效期预留空间。来源、地图、类别文本与检测坐标也有明确预算。输入数量检查发生在逐对象校验和成对匹配之前，超出配置时直接拒绝，不截断后假装完整。

## 当前观测与保留假设

`BeliefState` 提供：

- `observation`：本次输入的观测。有效输入保留原对象、检测顺序、坐标与数值；拒绝输入只把状态保守地降为 UNKNOWN/STALE，不使用历史框替换。
- `actionable`、`reason`：该观测是否可交给任务逻辑，以及实际接受/拒绝原因。
- `fields`：玩家框、HP、MP、地图、对话等字段的来源、置信度、测量时刻、有效期及 `observed_this_frame`。
- `tracks`：几何关联产生的对象轨迹与身份不确定性。
- `last_accepted_sequence`、`last_accepted_timestamp_ms`：最后一个实际可用观测的水位。
- `now_ms`、`epoch`：处理时钟与地图/感知来源上下文版本。

可行动输入要求：时间不在未来、年龄在预算内、相对最近处理的有效检测帧序号递增、采集时间不倒退、状态 KNOWN、整体及唯一玩家达到置信度门槛、地图/HP/MP 可读且 HP 大于零。所有条件通过才刷新完整测量和 accepted 水位；当前 HP 为零时不能用保留的旧血量继续行动。

重复帧、未来帧、过期帧、UNKNOWN、低置信度、玩家缺失或歧义、资源缺失等仍生成不可行动的 BeliefState。这些输入保留自身序号、时刻和测量内容，不能冒充旧的有效帧；也不会用一个极大序号阻止之后正常帧恢复。

带合格检测框的 fresh UNKNOWN 帧只进入**非执行性检测假设**：至多一个高置信度人物框可刷新 `player_bbox`，高置信度对象框可刷新轨迹及几何速度；整体 `observation` 仍是 UNKNOWN，地图、HP、MP 不补值，`actionable=false`，`last_accepted_sequence` 不前移。检测假设使用自己的内部序号水位，重复/倒序帧不能再次推进；未知地图 context 与随后完整地图 context 分离并重建，不能把跨图身份接起来。没有合格框的 UNKNOWN 仍不刷新历史测量，也不消耗该水位。

字段到期后清空值与置信度，状态标为 STALE；轨迹到期后移除。到期依据测量时刻，重复读取不会延长有效期。只读检测假设用于录像和窗口校准；任务层不能因为人物或怪物框存在就跳过地图、资源、焦点与权限门禁。

库存、技能、Buff 当前没有观测来源，因此对应字段保持 UNKNOWN、值与来源为空。不能因为模拟器内部有这些数据，就将它们塞入估计状态。任务也不能把保留字段当作本帧测量。

处理时钟倒退会使该估计器持续不可行动，并清空保留假设；恢复需要开始新的估计器/会话。正常 UNKNOWN 连续次数的停止预算仍由监督与任务层负责，估计器不伪造恢复成功。

## 对象关联

对象只在同一类别内关联。匹配代价由预测中心距离与框的 IoU 组成；预测使用已有两次测量得到的速度，只用于关联，不作为新检测输出。候选按几何与稳定 ID 排序，输入检测顺序变化不改变唯一匹配结果。

只有相互最近、两侧替代候选间隔都足够大、且本帧不存在高度重合框的匹配，才沿用已有 ID。交叉/重合等歧义会产生新的不确定轨迹并保留候选旧 ID；该身份歧义会继续保留，不因后续一个清晰框就宣称已确定原对象身份。

`TrackBelief` 中：

| 字段 | 含义 |
|---|---|
| `track_id` | 确定性的几何关联 ID，带上下文 epoch；不是游戏内部实体 ID |
| `association` | `new`、`matched`、`reacquired`、`ambiguous` 或 `unobserved` |
| `identity_uncertain` / `candidate_ids` | 身份存在歧义及其候选；不是经过校准的概率 |
| `bbox` | 最后一次实际测得的框；丢帧期间不替换为预测框 |
| `observed_this_frame` | 是否由本次接受的画面直接测得 |
| `observed_at_ms` / `expires_at_ms` | 测量时刻与失效时刻 |
| `velocity_px_per_second` | 两次框中心变化的估计速度；不代表真实运动模型 |

短暂遮挡后可以在预算内重新关联；到期后重新出现会分配新 ID。没有检测到目标只产生未观测假设，不宣称怪物死亡或掉落已拾取。地图或感知来源发生已接受的变化时，清空旧轨迹/字段并增加 epoch，防止跨图沿用对象身份。

跟踪内存由 `max_tracks` 限制，当前测量优先保留，其余按最近测量排序。关联计算有检测数量和轨迹数量的双重上限。没有保存无限历史列表或原始图像。低置信度的非玩家框保留在原始观测中以确保复现，但不进入关联；任务层仍应独立检查目标的置信度。

## 证据与验证

`to_dict()` 输出 JSON 可序列化的新容器，包含所有测量、时序、身份与拒绝信息；改写输出字典不会修改估计器或原观测。同一配置和输入序列生成相同证据。运行集成必须在每个初始、正常、暂停和终止帧上恰好更新一次，并保存 schema 与配置；重推理时从初始帧重建估计器。

专项测试覆盖真实像素观测透传、配置边界、输入资源预算、未来/重复/过期帧不污染水位、时间倒退、保留字段到期、移动与速度、遮挡重关联、身份交叉歧义、类别隔离、地图/来源切换、容器不可变和跟踪内存上限。

```bash
python -m pytest tests/test_state_estimation.py -q
```

这层目前是同一个感知来源下的保守测量保持与几何关联。多传感器融合、复杂状态转移、真实游戏 HUD、实例身份特征与概率校准仍需要独立实现和实测；不能把它们写成已经完成的完整 WorldState/BeliefState 系统。
