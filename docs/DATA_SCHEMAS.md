# 当前数据 Schema

本文描述实际保存的数据，先列当前增量格式，再保留原有兼容格式。`MASTER_SPEC.md` 中更细的技能、掉落、地图连接等表属于长期规划；本版本没有创建几十张专用业务表。
Schema 版本、文件哈希和来源断言分别解决兼容性、完整性和可追溯性；哈希本身不能证明内容真实或使用许可有效。

## v0.5 结构化运行证据

显式 structured 模式的 manifest、summary、trace 行均为 schema2；旧模式保持 schema1。`runtime-config.json` 是完整 `runtime_pipeline.RuntimeConfig.manifest()`，同时绑定 manifest.runtime 和 summary.runtime_evidence.settings。三个副本必须一致，文件读取在哈希前受 64 KiB 上限约束。schema1 不允许夹带新模式字段或边文件。

每行另有 runtime schema1：原始观测以外单独保存 belief、decision、policy_evaluated、dispatch、receipt 和动作回执之后的 state。decision 包含候选、计划、意图和效果证据；receipt 绑定真正应用的动作。输入冻结产生的 idle 不能改写成所请求攻击已经执行。

summary.runtime_evidence.terminal 保存最后 belief、decision、policy_evaluated、interruption 和 state。策略终止记录实际调用结果；监督停止或预算用尽记录 interruption，不补造没有执行过的决定。实际 reinfer 重建全部 N+1 帧并比较上述证据；仅动作 replay 不声称已经复现感知或策略。

状态字段与轨迹独立记录来源、观测时间、置信度、有效期及 observed_this_frame；旧框和旧资源不会重新进入当前可行动 observation。身份关联是几何假设，不能当作已核实怪物身份。详情见 [STATE_ESTIMATION](STATE_ESTIMATION.md) 与 [TASK_RUNTIME](TASK_RUNTIME.md)。

## v0.2–v0.4 格式索引

| 数据 | 格式 | 关键边界 |
|---|---|---|
| 外部地图包 | 原始 schema1 JSON；内嵌 manifest schema1 | 2–8 张固定尺寸合成地图、完整循环路线、规范化内容及 SHA-256 |
| 外部地图模拟器状态 | save schema3 | 嵌入完整 map_pack；可同时包含原 v2 的 class_profile |
| 外部地图 lifecycle 数据集 | manifest schema3，generator lifecycle-map-pack-v1 | 嵌入完整地图；按包内路线核验六阶段、动作、像素和标签 |
| 外部地图运行证据 | 原 bundle schema1 加可选地图字段/文件 | map-pack.json、状态内 map_pack、summary.map_pack 必须一致；感知 settings 绑定地图 SHA-256 |
| 本地知识同步计划 | 独立 schema1 JSON | 源原字节、完整基线、增量和预期结果；原子应用，非授权签名 |
| lifecycle 数据集 | manifest schema2，generator lifecycle-v1 | 会话动作轨迹、阶段/位置/标签覆盖；校验实际画面和可见标注 |
| lifecycle 评估 | report schema2 | 同批预测上的阶段、地图、位置与重叠标签分组 |
| 地形几何评估 | report schema3，terrain_geometry schema1 | 保留聚合/分组，追加匹配后边缘误差及全宽真值召回；模型原生身份与运行选项 |
| 地形结构检查 | 独立 schema1 JSON | 实际检测头、DFL、stride、候选及四边范围、来源哈希；不是训练准确率 |
| 感知重执行证据 | 运行 summary.perception_evidence schema1 | settings、终止观测/监督/决策；真实 N+1 帧重新执行 |
| 模型地形模式 | pipeline/source_id `onnx-all-objects+fixture-hud-v1` | settings 额外明确 terrain_source=model；默认 fixture 字段不变 |

具体严格字段分别以 [知识同步](KNOWLEDGE_SYNC.md)、[生命周期](DATASET_LIFECYCLE.md)、[地形评估](TERRAIN_EVALUATION.md)、[模型地形](MODEL_TERRAIN.md) 和 [重新推理](MODEL_VALIDATION.md) 为准。各文档报告版本不能当作模型准确率、规则版本或运行成功标记。下述旧格式继续兼容，不因新增可选字段改写历史记录。

## 外部地图包 v1

原始地图文件是 UTF-8 JSON 数据，最大 128 KiB。`load_map_pack` 验证普通文件及字节预算；`MapPack.from_dict` 验证字段并构造不可变对象。两者都不执行地图代码、不下载素材、不访问 `source_uri`。根字段必须完整且不能添加未知字段：

| 字段 | 约束 |
|---|---|
| `schema_version` | 严格整数 1；布尔值不视为整数 |
| `id`、`version` | id 为最多 40 字符的小写标识；version 为有界 major.minor.patch 数字版本 |
| `scope` | 固定 `synthetic_fixture` |
| `source_uri`、`permission`、`license` | 必填来源与许可断言；permission 为 self_authored/public_license/explicit_permission |
| `width`、`height`、`ground` | 固定严格整数 320、192、168 |
| `route` | 2–8 个唯一地图 ID，恰好包含包中所有地图；首项为入口 |
| `maps` | 2–8 个地图对象；存储顺序规范化为 route 顺序 |

地图 id 使用 `[a-z][a-z0-9_-]{0,39}`，不能与内置 meadow/cavern 重名。每张地图必须提供以下字段：

| 地图字段 | 约束 |
|---|---|
| `id`、`theme` | id 包内唯一；theme 仅 meadow/cavern，决定已有背景样式 |
| `marker` | 三个 0–255 整数，颜色在包内唯一且不能使用保留对象/HUD 色；进入感知的静态地图颜色表 |
| `spawn_x` | 0–306 整数；角色初始 y 固定 146，传送到该图时使用同一出生点 |
| `platforms` | 2–8 个 `{x1,x2,y}`；首项必须为全宽地面 `{x1:0,x2:320,y:168}`，其它平台位于画面内且不能在同高度相交或相接 |
| `portals` | 恰好一个 `{id,x,y,w,h,destination}`；尺寸 20×36，y=132；destination 必须等于路线下一项，末图返回入口 |
| `npcs` | 恰好一个 `{id,x,y,w,h}`；尺寸 14×24，y=144；本图 NPC 与 portal 的 id 不重复 |
| `ladders` | 1–4 个 `{x,y,w,h}`；宽 10，x=2–308，连接地面与一块支持平台；居中的角色仍完整处于世界内 |
| `monster_x` | 1–6 个地面怪物的整数横坐标，间距至少 24；坐标预算包含确定性 ±3 像素出生扰动 |

原始 JSON 拒绝重复键；几何数值不接受 bool、浮点数、NaN 或无限值。元数据通过结构校验不意味着地图一定可解，也不意味着来源声明已经得到外部验证。

运行、存档和数据集嵌入的 **地图 manifest** 与原始输入文件不同：

```python
{
    "schema_version": 1,
    "content": pack.to_dict(),
    "content_sha256": pack.content_sha256,
}
```

`MapPack.manifest()` 生成此结构，`MapPack.from_manifest()` 重新验证 content 并核对 content_sha256，其值为 64 位小写十六进制字符串。`pack.to_dict()` 才是可直接作为原始地图文件的完整内容。哈希以排序对象键、紧凑分隔符、不转义非 ASCII 字符的 UTF-8 JSON 计算；地图数组先按 route 规范化，其它具有语义的列表保持顺序。文件缩进或对象键顺序变化不改变规范化内容身份，路线、几何、颜色、来源或许可变化都会改变身份。

### 外部地图模拟器状态 schema3

有 map_pack 时，模拟器 save 的 schema_version 为 3，原有 seed、job、step_ms、sequence、map_id、initial_map_id、rng_state、next_drop_id、player、maps、events、dialogue 和 ruleset 均保留，增加完整 `map_pack` manifest。`maps` 的键必须恰好等于包内地图集合；当前图和初始图必须属于该集合。

职业配置与地图可以同时注入，此时 schema3 还包含 `class_profile.spec` / `class_profile.sha256`。恢复时先验证地图和职业配置，再校验动态状态、资源范围、平台/梯子空间关系；无需原始地图 JSON 或知识数据库。未注入地图仍沿用默认 schema1 或仅职业配置的 schema2，这两个旧版本不能夹带 map_pack。

### 外部地图 lifecycle 数据集 schema3

外部地图仅用于 lifecycle profile，保留 schema2 的会话动作轨迹、样本元数据和覆盖统计，并增加完整 `map_pack`。固定 `schema_version=3`、`profile=lifecycle`、`generator_version=lifecycle-map-pack-v1`；origin 仍为 `synthetic-fixture-v1`。样本类别、RGB PNG、YOLO 标签及 data.yaml 的合同不变，也可以同时携带已注册 sprite_pack。

每个 session 从入口开始沿完整循环路线执行，最终回到入口；每图需要 spawn、combat、loot、cleared、dialogue、transition 六阶段。最低帧数为 `6 × 地图数`，每会话动作上限 4096，所有会话动作总数上限 1000000。session 仍作为 train/val/test 的划分单位，同一地图可以出现在不同会话分区中，这不是留出地图的泛化实验。

验证器从 manifest 恢复同一包，独立执行记录动作并重新渲染，核对每帧时间、地图、阶段、资源/位置事实、RGB 像素和可见实例标签，同时要求每图发生对话、命中且最终清场收集完成。缺失阶段、资源耗尽或超预算不会被记为成功覆盖；生成器清理失败的暂存目录。训练工具复制完整 manifest 及图片标签后再次验证，外部源地图删除后私有输入快照仍可独立验证。

评估报告的 `by_map` / `by_map_phase` 使用已验证包的 route，`dataset` 摘要保留地图内容身份。普通 lifecycle 分组报告仍是 report schema2；显式地形几何报告仍是 report schema3。数据集 schema3 与评估报告 schema3 是各自独立的格式版本，不能混为同一种含义。

## SQLite v1 与 v2

| API / 命令 | 数据库版本 | 语义 |
|---|---:|---|
| `knowledge.py`；`db init/check` | 1 | P0 兼容层，仅旧 `source_records`；拒绝未知版本 |
| `catalog.py`；`knowledge init/check/import/query` | 2 | `knowledge init` 显式把兼容 v0/v1 升至 v2；升级后使用 catalog API |

v2 有两个表和一个检索索引：

| 对象 | 实际字段或索引 |
|---|---|
| `source_records` | `id`、`source_uri`、`game_version`、`payload_hash`、`verification_status`；id 增加 NOT NULL 和非空约束 |
| `catalog_records` | `record_id`、`kind`、`entity_id`、`game_version`、`source_uri`、`collected_at`、`permission`、`license`、`scope`、`payload_json`、`payload_sha256`、`declared_status`、`verification_status` |
| `catalog_entity` | `(kind, entity_id, game_version)` |

`kind` 仅允许 classes、skills、monsters、maps、items、npcs、quests、mechanics；细项存入对象型 JSON payload。
旧 source_records 保留作历史记录，不会凭缺失的旧 payload 自动生成 catalog_records。
迁移使用 SAVEPOINT 保留调用方事务所有权，拒绝空旧 ID、未知结构及依赖旧表的扩展对象，失败完整回滚。

导入 JSON 为 `{"schema_version":1,"records":[...]}`。每条记录必须明确提供：

```json
{
  "kind": "classes",
  "entity_id": "fixture-assassin",
  "game_version": "synthetic-fixture-v1",
  "source_uri": "urn:maple-vision:authored-fixture",
  "collected_at": "2026-11-24T00:00:00Z",
  "permission": "self_authored",
  "license": "Project-authored fixture",
  "verification_status": "PROVISIONAL",
  "scope": "synthetic_fixture",
  "payload": {"purpose": "schema example, not game rules"}
}
```

permission 允许 self_authored/public_license/explicit_permission；scope 允许 synthetic_fixture/game_reference。
状态为 VERIFIED/PROVISIONAL/UNKNOWN/CONFLICTED；时间必须带时区，JSON 拒绝重复键和非有限数字。
payload_sha256 是规范化 payload 的内容哈希；record_id 覆盖原始导入记录的规范化字段，重复导入幂等。
相同 kind/entity_id/game_version 的不同 payload 全部保留并标记冲突；默认查询只返回 VERIFIED，显式检查模式才返回其它状态。

知识快照目录包含独立 `knowledge.sqlite` 和 `manifest.json`，后者记录 manifest_version、文件名、哈希和汇总。
验证拒绝额外 WAL/SHM/journal 侧文件，防止主文件哈希通过但读取未登记页面；运行读取用 SQLite mode=ro 和 query_only。

## Asset Registry

图片 bytes 保存在 `objects/<sha256>.<png|jpg|webp>`；元数据保存在 `records/<asset_id>.json`。
AssetEntry 字段为 asset_id、entity_id、game_version、source_uri、license、permission、sha256、relative_path、width、height、media_type。
asset_id 是元数据哈希，sha256 是原始图片 bytes 哈希。验证重新解码图片并检查尺寸、类型、路径和两类哈希。
这是文件注册表，没有 `asset_index` SQL 表。`sprite_pack.py` 可将已验证 AssetEntry 映射到渲染类别，详见 [SPRITE_PACK.md](SPRITE_PACK.md)。

## 图像、观测与模拟器

| 数据 | 版本 / 关键字段 | 边界 |
|---|---|---|
| `ImageFrame` | sequence、timestamp_ms、width、height、pixels | RGB24；长度严格为 width×height×3，不含标签 |
| `Detection` | class_id、bbox、confidence | xyxy、非负原点、正面积、有限值；置信度 [0,1] |
| `VisualObservation` | 帧身份、status、confidence、detections、hp_ratio、mp_ratio、map_id、dialogue_visible、source_id | 不可读资源为 null，空白/不确定场景 UNKNOWN |
| `SimAction` | kind、direction、skill、target | 只进入本地模拟器；方向 -1/0/1，动作与参数组合校验 |
| `SimSnapshot` | 玩家、实体、平台、门、NPC、梯子、事件、对话及场景身份 | 仅渲染/数据生成/独立验证使用 |
| 模拟器 save | 默认 schema1；仅注入职业配置时为 schema2；注入地图时为 schema3 | restore 校验结构、数值、空间关系及版本；schema2 内嵌 class_profile，schema3 内嵌 map_pack 并允许同时内嵌 class_profile |

`profiles.load_class_profile()` 只接受同一 job/version 的 VERIFIED、无冲突 synthetic_fixture 记录。payload 必须完整包含 ClassSpec 的 12 个字段：id、max_hp、max_mp、attack_damage、attack_range、attack_mp_cost、attack_cooldown_ms、ammo_cost、can_jump_attack、heal_amount、walk_speed、jump_speed。
LoadedClassProfile 包含 class_spec、profile_sha256、provenance_sha256 和全部来源记录；run summary 保存这些信息，模拟器保存格式只内嵌 spec 与其哈希。其它知识类型不会自动改变世界规则。

P0 的 opaque `Frame` / `WorldState` 保留原语义，不能混同为新的 RGB 帧或完整玩法世界状态。

## 合成数据集 v1

`manifest.json` 含 schema_version、origin、seed、classes、split_unit、sessions、samples、data_yaml_sha256、notes。
origin 为 synthetic-fixture-v1，split_unit 为 session，类别顺序固定为 player/monster/loot/portal/npc/platform/ladder。
每个 session 记录 id、split、seed、style_seed、job、map_id；每个 sample 记录所属会话和分区、帧序号/时间、宽高、图像/标签路径与 SHA-256、对象数。

图像为 RGB PNG；YOLO 行为 `class cx cy w h`，后四项按图像尺寸归一化。
`data.yaml` 使用 YAML 兼容的 JSON，固定 train/val/test 路径和类别列表；不允许被篡改后指向其它目录。
使用可选 sprite pack 时额外保存 `visual_origin=synthetic_geometry_with_authorized_assets` 与 sprite_pack 元数据，并把原始图片 bytes 放入 sprite_objects；origin 仍为 synthetic-fixture-v1，因为地图和物理仍是合成 fixture。元数据绑定原始资产、来源、解码后 RGBA 及 pack 的哈希，验证重新解码拷贝的源文件。未提供 pack 时不增加这些字段，默认数据集格式和内容哈希保持不变。

数据版本内容由 manifest 哈希标识；同名 schema_version 只代表格式兼容，不代表两个训练集内容相同。
会话隔离与跨分区重复图片检查不能证明覆盖全部任务状态或真实域；覆盖摘要和独立基准仍必需。

## 模型与评估

ONNX manifest schema_version=1，字段为 model_sha256、class_names、input_width、input_height、output_format、scope。
output_format 仅 raw_xywh（`[1,4+C,N]`）或 end2end_xyxy（`[1,N,6]`）；不猜测其它输出格式。scope 为 synthetic_fixture 或 authorized_recordings，由提供者声明。
训练报告保存数据与配置哈希、权重哈希、训练参数、依赖、CPU 线程、耗时与 validation 指标；导出报告保存 ONNX 输入/输出形状和 smoke 推理结果。

评估 JSON 独立记录数据 manifest 哈希、检测器/模型信息、阈值、micro 和逐类指标、AP 定义、热身与时延、运行环境、限制。
AP50 不等于 COCO mAP50–95；无 GT 类的 AP 为 null。详见 [EVALUATION.md](EVALUATION.md)。

## 本地录像与 OCR

录像导入 manifest 为 kind=offline_recording、scope=user_supplied_recording，记录来源 URI/许可断言、源文件或选定 PNG 的哈希、名义 FPS、时间基准、帧尺寸、序号、时间与每帧哈希；session_id 覆盖 manifest 内容。
MP4 经本地可选解码器转换，PNG 序列需要显式 FPS。时间戳来自声明的名义帧率，并非原始媒体逐帧 PTS 的精确还原；解码停止和帧数截断有独立标记。
OCRResult 包含帧身份、language、source_id、crop、words 和 text；OCRWord 含 text、原图坐标 bbox 与 confidence。OCR 置信度不等同经过校准的识别正确概率。

## 运行证据包 v1

| 文件 | 内容 |
|---|---|
| `initial-state.json` / `final-state.json` | 完整模拟器保存状态 |
| `trace.jsonl` | 每步帧哈希、观测、监督决策、任务决策、实际动作、执行事件、可见效果、结果状态哈希 |
| `input-bindings.json`（可选） | 用户确认的角色配置名及 1–128 条技能/键位；仅用于离线模拟技能解析 |
| `frames/*.png` / `final.png` | 间隔采样画面与最终画面 |
| `preview.gif` | 最多 600 个采样帧组成的预览，不能替代完整动作记录 |
| `summary.json` | 配置、状态/原因、步骤、模拟/实际时长、行为/事件统计、独立目标检查、模型及运行身份 |
| `manifest.json` | kind=offline-run-bundle、文件到 SHA-256 的映射、recording 与 replay_scope |
| `sprite-pack.json` / `sprite_objects/*`（可选） | 实际使用的素材 pack 元数据与原始图片；全部纳入 bundle 文件哈希，summary 中的 pack 必须一致 |
| `map-pack.json`（可选） | 完整 MapPack manifest；与状态及 summary 中嵌入的地图必须一致，文件自身纳入 bundle SHA-256 索引 |

suite.json 记录多职业/种子和故障结果；soak 另写 episodes.jsonl、checkpoint.json、soak.json。
启用素材 pack 的回放先验证包内源图片、AssetEntry、RGBA 和 pack 哈希，再重建同一个渲染器；原 registry 删除后仍可回放。
回放重新计算模拟器状态与像素，不重新执行原模型，明确返回 `perception_reproduced=false`。源文件/包版本身份用于识别环境变化，不是防篡改签名。

外部地图运行的 summary 额外保存 `map_pack`，`summary.perception_evidence.settings.map_pack_sha256` 绑定实际地图内容。suite/soak 摘要同样携带地图 manifest，内部感知设置也包含同一内容摘要；默认内置地图路径不写这些新增字段。

`verify_run` 必须核对包内 map-pack.json、initial state 和 summary 的完整地图身份，再使用恢复后的包重放所有动作和采样像素；地图证据缺失、混用或未登记会被拒绝。`verify_perception` 在此基础上重建相同地图标记表与路线，检查绑定地图后的感知 settings 并重新执行 N+1 帧感知、监督与策略。两种验证都无需再次提供外部地图文件；数据内容哈希仍不能代替模型在这些新地图上的准确率或闭环准入证据。

选择角色键位配置时，summary.input_profile 与 `input-bindings.json` 必须逐字段相同。每个受键位约束的技能步骤增加 `execution.input_dispatch`，记录 profile、技能语义、原始/归一化键、是否绑定及原因；非技能步骤为 `null`。重放器根据调度授权动作和配置重新计算该字段，再决定实际应用原动作或 idle。此字段是离线模拟输入证据，不是操作系统或正式客户端输入回执。
