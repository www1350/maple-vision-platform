# 当前数据 Schema

本文描述 v0.1 代码实际保存的数据。`MASTER_SPEC.md` 中更细的技能、掉落、地图连接等表属于长期规划；本版本没有创建几十张专用业务表。
Schema 版本、文件哈希和来源断言分别解决兼容性、完整性和可追溯性；哈希本身不能证明内容真实或使用许可有效。

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
| 模拟器 save | 默认 schema_version=1；注入职业配置时为 v2，并增加 class_profile.spec/sha256 | restore 校验结构、数值、空间关系及版本；v2 内嵌参数保证脱离数据库也能回放 |

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
| `frames/*.png` / `final.png` | 间隔采样画面与最终画面 |
| `preview.gif` | 最多 600 个采样帧组成的预览，不能替代完整动作记录 |
| `summary.json` | 配置、状态/原因、步骤、模拟/实际时长、行为/事件统计、独立目标检查、模型及运行身份 |
| `manifest.json` | kind=offline-run-bundle、文件到 SHA-256 的映射、recording 与 replay_scope |
| `sprite-pack.json` / `sprite_objects/*`（可选） | 实际使用的素材 pack 元数据与原始图片；全部纳入 bundle 文件哈希，summary 中的 pack 必须一致 |

suite.json 记录多职业/种子和故障结果；soak 另写 episodes.jsonl、checkpoint.json、soak.json。
启用素材 pack 的回放先验证包内源图片、AssetEntry、RGBA 和 pack 哈希，再重建同一个渲染器；原 registry 删除后仍可回放。
回放重新计算模拟器状态与像素，不重新执行原模型，明确返回 `perception_reproduced=false`。源文件/包版本身份用于识别环境变化，不是防篡改签名。
