# 生命周期数据集 v2 与分组评估

本功能为合成训练补齐任务开始、战斗、拾取、清场、对话和切图画面。它生成**由实际模拟动作产生的完整轨迹**，并在验证时重新执行这些动作，核对图片、标签、阶段和覆盖统计。场景生成器可以使用训练真值规划动作；它不调用运行时 VisualAgent，不是 YOLO 完成任务的成功率测量。

现有 `profile="legacy"` 是默认值，继续生成 schema v1，原始数据和报告不迁移、不覆盖。v0.1 的 18 会话 ×16 帧、seed=20261124 数据重新生成后，manifest SHA-256 仍为：

`ecc0ff6c4e9bbebc9915f0313bb3b227a99ecc321051ee2ae2e8a0eff096ad67`

## 1. 使用方式

```python
from pathlib import Path
from maple_vision.dataset import generate_dataset, validate_dataset

result = generate_dataset(
    Path("artifacts/lifecycle-dataset"),
    sessions=18,
    frames_per_session=48,
    seed=20261125,
    profile="lifecycle",
)
validate_dataset(result["output"])
```

现有调用方无需更改；`sprite_pack` 仍为独立的可选关键字参数。Python API 的历史 `frames_per_session=8` 默认值保留，选择 lifecycle 时必须显式给出至少 12 帧。CLI 对 lifecycle 默认选择 48 帧，仍建议实验显式写出配置：

```bash
python -m maple_vision dataset create artifacts/lifecycle-dataset --profile lifecycle --sessions 18 --frames 48 --seed 20261125
python -m maple_vision dataset check artifacts/lifecycle-dataset
python -m maple_vision vision benchmark artifacts/lifecycle-dataset --output artifacts/lifecycle-pixel-test.json --split test
```

评估 ONNX 时成对增加 `--model MODEL.onnx --manifest MANIFEST.json`。训练和导出仍使用现有 `model train/export`；YOLO 图片、标签、类别顺序和 `data.yaml` 不变。素材包仍通过 `--asset-registry` 与 `--sprite-map` 接入。相关合同见 [TRAINING.md](TRAINING.md)、[SPRITE_PACK.md](SPRITE_PACK.md) 和 [EVALUATION.md](EVALUATION.md)。

## 2. 实际生成什么

每个独立会话固定一个职业、随机种子和背景样式。路径为 **meadow → cavern → meadow**：两图分别战斗、收集全部掉落、到达右边缘、爬上平台、访问 NPC，然后使用真实传送门。初始位置 x=24，首次进入 cavern 后为 x=35；没有修改模拟器私有状态来伪造这些位置。

阶段按每帧实际状态判断，优先级为下表从上到下；只有适用条件全部排除后才进入后续阶段。

| phase | 实际定义 |
|---|---|
| `transition` | 本次模拟 step 产生 `map_changed`；保存的是传送后的目的地图画面 |
| `spawn` | 序号 0 的初始帧，或首次进入该地图后的下一步；不把所有靠左画面称为出生 |
| `dialogue` | NPC 对话面板实际打开 |
| `combat` | 当前地图仍有存活怪物；包含接近和等待冷却，未宣称每帧都命中 |
| `loot` | 怪物已清空，但还有未收集掉落 |
| `cleared` | 怪物与掉落均清空，且没有更优先的阶段条件 |

每个会话必须选择两图各六种阶段，共至少 12 帧。combat 的必选帧还必须包含实际 `attack_hit`。往返路径确实在两张目的地图上产生 transition，因此没有把未发生的事件强行分配给某张地图。缺少任何要求阶段时生成失败，暂存目录会清理，不发布半成品。

样本预算足够时优先保留两图的高平台、右边缘和非落地画面；18 帧及以上覆盖这些当前轨迹中的场景。其余预算在地图/阶段分组内以固定种子抽取，按时间排序输出。请求超过完整路径长度的帧数时，以实际 idle 步延长已清空场景；不会重复序号或虚构新阶段。

“清空”指战斗区域无怪物、无掉落，画面仍有角色、地形、NPC 等。`not_grounded` 包含攀爬，不等同于自由落体。`high_platform` 只在角色已落地且脚底高于地面时成立；不能用“y 较小”把腾空帧冒充站上高台。

会话整体分配给 train/val/test。会话数足够时按职业分层，令各 split 包含三个职业；每个会话本身经过两张地图。分区仍拒绝完全相同图片跨 split 出现，但这不是未见地图实验：两张内置地图仍出现在所有分区。

## 3. v2 数据合同

根 manifest 保留既有 `origin="synthetic-fixture-v1"`、类别顺序和会话分区定义，并新增：

| 字段 | 含义 |
|---|---|
| `schema_version: 2` | 生命周期数据版本；与模拟器保存版本无关 |
| `profile: "lifecycle"` | 显式选择场景生成方式 |
| `generator_version: "lifecycle-v1"` | 固定本版轨迹与元数据语义 |
| `frames_per_session` | 每个会话实际选取的帧数 |
| `sessions[].trajectory` | `step_ms=50` 与完整 `SimAction` 字段字典列表 |
| `samples[].map_id` | 当前帧实际地图，session 的 `map_id` 仍表示初始地图 |
| `samples[].phase` | 按上表从状态和事件计算的阶段 |
| `samples[].coverage` | 位置、状态、可见类别计数与覆盖标签 |
| `coverage` | 全数据集按 split/map 汇总的覆盖报告 |

动作轨迹直接放在 manifest 中，不新增未声明的外部录像依赖。逐帧 coverage 包含 `player_x`、`player_y`、`position_bin`、`grounded`、`monsters_alive`、`drops_remaining`、`dialogue_visible`、`map_transition`、`visible_class_counts`、`tags`。其中位置与计数是数据制作/验收元数据，**不会加入 `ImageFrame` 或传给检测器**；可见类别计数来自最终渲染实例标签，隐藏对象数量和可见对象数量分别保存。

位置分组使用角色完整 sprite 左上角的世界 x；本版全宽视口无相机偏移，不使用模型预测框决定所属组：

| position_bin | x 范围 |
|---|---|
| `left_edge` | `[0,40)` |
| `left` | `[40,120)` |
| `middle` | `[120,200)` |
| `right` | `[200,280)` |
| `right_edge` | `[280,307)`；实际角色最右位置为 306 |

标签包括边缘、高平台、未落地、怪物/掉落存在、已清场、对话可见、地图切换、实际命中和实际收集。标签可以重叠；一帧对话画面也可能同时满足已清场。

`coverage.by_split[SPLIT]` 与其 `maps[MAP]` 均提供 `frames`、`phase_counts`、`class_counts`、`player_position_bins`、`tag_counts`。全局定义记录分组坐标、要求阶段和标签重叠事实。既有返回摘要中的 `split_maps` 在 v2 按实际帧地图统计，不再将整段会话都归为初始地图。

## 4. 校验不只检查摘要和哈希

既有图片/标签/config 哈希、YOLO 坐标范围、类别、尺寸、路径、会话隔离与文件清单检查继续执行。v2 额外严格检查字段集合、种子/样式/职业身份、分区安排和轨迹预算，然后每个会话只重放一次：

1. 从已声明种子、职业、初始地图重建独立模拟器。
2. 依次执行动作，在所选序号重新计算地图、阶段、位置和状态；即使未选取某个中间帧，也执行其动作。
3. 对所选帧重新渲染，比较 RGB 像素和最终可见实例标签；自定义 SpritePack 使用数据包内已校验的源素材。
4. 要求实际两图清理、对话和往返成立，并检查所选帧满足阶段覆盖；从通过核验的样本重新计数，与声明的 coverage 摘要比较。

因此，伪造 phase、位置分组、布尔值类型或汇总计数会失败；改图片/标签后重算 SHA-256 仍会因动作重放不一致而失败。哈希与重放仍不是外部授权、来源真实性或真实游戏规则的证明。若攻击者重新生成了一整套内部一致的合成数据，平台不会把它当作由第三方签名的数据。

资源边界为：3–1000 个会话、每会话 12–1000 个所选帧、总计最多 100000 帧；单会话最多 4096 步，总计最多 1000000 步；manifest 最多 128 MiB。验证对动作做线性重放，仅渲染选中的帧，不为每张图片从头重放整个会话。

## 5. 分组评估

`benchmark_dataset()` 仍先完整验证数据，再给检测器每张图一次实际推理；最多三次 warmup 不计入延迟。`metrics` 保留原聚合算法和全部预测，不因数据标签决定保留或替换某个模型输出。legacy v1 仍返回原 schema v1 报告，不伪造不存在的生命周期组。

v2 报告使用 `schema_version=2`，额外提供 `grouped_metrics`：

- `by_phase`：六个阶段。
- `by_map`、`by_map_phase`：实际地图及其阶段组合。
- `by_player_position`：五个 x 分组，包括原数据缺少的左侧起点。
- `by_coverage_tag`：高平台、对话等可以重叠的覆盖条件。

每组含图像数、独立会话数、TP/FP/FN、precision/recall/F1、逐类 AP 与宏平均。phase/map/position 各自形成完整分区，计数可以回加聚合结果；覆盖标签互相重叠，不能相加。AP 在每组内重新按置信度排名，组 AP 也不能加和。无该类真值时 AP/recall 为 `null`，但预测出的误检仍计 FP；清场阶段不会把“没有怪物真值”解释成模型永远正确。

AP 仍是**指定 confidence 截断后、单一 IoU 的 101 点插值精度包络**。IoU=0.5 时报告 AP50；`map50_95=null`，没有冒充完整 COCO mAP50–95。单会话相邻帧有相关性，分组结果是描述性统计，不自动提供独立样本置信区间或真实域准确率。

## 6. 本轮数据与验证证据

本轮不可变实验输入：18 个会话 ×48 帧，seed=20261125，864 张图片、7098 个可见标注。三个 split 均包含三个职业、两张地图、六个阶段和全部七个检测类别。manifest SHA-256：

`670f6dae181f0f4b3107ba6a47cb733f19cfbc216c4e8e8c23f96b756601b3ff`

| split | 帧数 | spawn | combat | loot | cleared | dialogue | transition | x<40 | 高平台帧 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 480 | 30 | 110 | 110 | 170 | 40 | 20 | 57 | 24 |
| val | 192 | 12 | 44 | 44 | 68 | 16 | 8 | 23 | 11 |
| test | 192 | 12 | 44 | 44 | 68 | 16 | 8 | 24 | 9 |

数据位于本次本地实验的 `artifacts/v02-experiment/dataset`；大量可重建图像不因此自动加入源码 Git。模型训练、选模和最终 test 结果由独立实验报告记录，本文不提前给出模型效果结论。

针对性验证命令：

```bash
python -m pytest -q tests/test_lifecycle_dataset.py tests/test_dataset.py tests/test_evaluation.py
python -m ruff check src/maple_vision/dataset.py src/maple_vision/evaluation.py tests/test_lifecycle_dataset.py
python -m mypy src/maple_vision/dataset.py src/maple_vision/evaluation.py
```

本轮上述组合为 67 项通过；包含真实轨迹语义、会话隔离、重复生成、v1 固定哈希、缺阶段/超预算失败、重哈希伪造、素材源删除后独立验证、清场误检计数、分组与聚合一致性。独立审查补充了“用真实但从未命中的战斗帧替换全部必选战斗帧”的反例：即使图片、标签和摘要全部一致，也必须因缺少所选命中证据而拒绝。最终发布仍应以准确提交的完整 CI 和验收记录为准。
