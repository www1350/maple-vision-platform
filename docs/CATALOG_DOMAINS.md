# 版本化知识领域投影

`catalog_domains.py` 对应原始总纲第 03 章。它从一个已经形成的 v2 知识快照，
生成独立、可复验的 SQLite 领域数据库。职业、技能等级、技能效果、怪物掉落、地图连接、
装备与任务要求都有独立的类型列和关系约束；领域查询无需解析通用 `payload_json`。

本模块不更改现有 `catalog.py`、同步计划、职业参数加载或 Runtime。旧数据库保持原样；
原始资料不足以映射时保留证据和缺失原因，不猜测游戏数值。不访问小册子或其他网站。

## API 与产物

```python
from pathlib import Path
from contextlib import closing
from maple_vision.catalog_domains import (
    build_domain_snapshot,
    verify_domain_snapshot,
    open_domain_readonly,
    query_domain,
)

source = Path("artifacts/knowledge-snapshot")  # catalog.create_snapshot() 的输出
output = Path("artifacts/domain-snapshot")  # 必须尚不存在

manifest = build_domain_snapshot(source, output)
assert verify_domain_snapshot(output) == manifest

rows = query_domain(
    output,
    "skill_levels",
    "fixture-v1",
    scope="synthetic_fixture",
    entity_id="throw",
    limit=100,
)
with closing(open_domain_readonly(output)) as connection:
    foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
```

CLI 已接入 `knowledge project-domains SOURCE OUTPUT`、`knowledge verify-domains OUTPUT` 和 `knowledge query-domain OUTPUT TABLE GAME_VERSION --scope SCOPE`。完整自写例子见 `fixtures/knowledge-domain-demo.json`；普通旧 fixture 不会自动变成领域导出。

产物包含 `domain.sqlite`、`manifest.json`，以及 `source/knowledge.sqlite` 和
`source/manifest.json`。归档的源数据库和源 manifest 保持逐字节不变，后续验证不依赖原目录。
所有四个文件均是新产物；不会迁移、覆盖或改写历史快照。

`manifest.json` 使用 schema 1，固定 kind 为 `domain-catalog-snapshot`。
`files` 绑定三个其他文件的 SHA-256；`snapshot_id` 是去掉自身后整个规范 JSON 的 SHA-256。
规范 JSON 使用项目已有的排序键、UTF-8、有限数值格式。数据库为 SQLite `STRICT` 表，
`PRAGMA user_version=1`，不与原 v2 catalog 的数据库版本混用。

## 输入合同与证据状态

外层来源记录沿用 [CATALOG.md](CATALOG.md) 的全部字段，包含 `scope`、`game_version`、
原始来源、采集时间、授权声明、声明状态和有效状态。仅在来源记录的有效状态为 `VERIFIED`，
且 payload 明确声明 `domain_schema_version: 1` 时，才尝试投影。

最小可映射 payload 示例：

```json
{
  "domain_schema_version": 1,
  "name": "Self-authored learner",
  "tier": 0,
  "advancements": []
}
```

这个 payload 需要放入 `kind=classes` 的完整外层来源记录。`[]` 表示该导出明确提供的空集合；
省略一个必需集合会产生缺失问题。它是示例领域数据格式，不是实际国服资料。

处理规则如下：

| 源记录情况 | 结果 |
|---|---|
| `UNKNOWN`、`PROVISIONAL`、`CONFLICTED` | 保留原证据，写入 `unverified_or_conflicted` 问题，不产生业务领域行 |
| 没有 `domain_schema_version` | `unmapped_payload`；不把旧任意 JSON 自动推断成新 schema |
| 明确声明其他整数 schema | `unsupported_domain_schema` |
| 缺少必需字段、子字段或某个技能等级 | `missing_fields`；撤销整条实体及其已写子表，不保留半个技能 |
| 已识别 schema 中字段类型错误、未知字段、重复关系、无效数值或循环依赖 | 整次构建失败，不发布产物 |
| 指向缺失、未投影、冲突、其他版本或其他 scope 的实体 | 同版本同 scope 外键不成立，整次构建失败 |
| 多条已验证来源具有同实体、版本、scope 和相同 payload | 一个领域实体，多条 `entity_sources` 来源链接 |

允许为 `null` 的数值仍以 SQL `NULL` 保存，含义是该字段未知，不能读成零。
原 v2 来源没有置信度字段，因此 `source_records.confidence` 始终为 `NULL`；
本模块不会把 `VERIFIED` 换算为 1.0 置信度。

`VERIFIED` 沿用来源的声明与冲突检查结果。格式校验、关系成立、文件哈希相等均不证明
导入者确有版权许可，也不证明内容符合实际游戏规则。合成 fixture 的已验证状态只作用于
明确标出的合成数据范围。

## 领域 payload schema 1

下表列出除公共 `domain_schema_version` 之外的所有必需字段。根对象还可包含可选
`assets` 与 `evidence` 数组；其余字段拒绝。`?` 表示允许显式 `null`，不表示可以省略键。

| kind | 必需根字段 | 主要规范化关系 |
|---|---|---|
| `classes` | `name`、`tier`、`advancements[]` | `classes`、`advancements` |
| `skills` | `name`、`class_id`、`max_level`、`levels[]` | `skills`、`skill_levels`、`skill_effects` |
| `monsters` | `name`、`level`、`max_hp?`、`max_mp?`、`experience?`、`drops[]` | `monsters`、`monster_drops` |
| `maps` | `name`、`width`、`height`、`connections[]`、`platforms[]`、`portals[]` | `maps`、`map_connections`、`map_platforms`、`map_portals` |
| `items` | `name`、`category`、`stack_limit`、`equipment?`、`consumable?` | `items`、`equipment`、`consumables` |
| `npcs` | `name`、`map_id`、`x`、`y` | `npcs` |
| `quests` | `name`、`npc_id?`、`min_level`、`requirements[]` | `quests`、`quest_requirements` |
| `mechanics` | `name`、`rule_type`、`unit`、`numeric_value?`、`text_value?` | `mechanics_rules` |

子对象也采用固定字段集合：

| 子对象 | 必需字段 |
|---|---|
| advancement | `to_class_id`、`required_level` |
| skill level | `level`、`hp_cost?`、`mp_cost?`、`ammo_item_id?`、`ammo_cost?`、`cooldown_ms?`、`cast_ms?`、`effects[]` |
| skill effect | `effect_id`、`kind`、`target`、`magnitude?`、`duration_ms?`、`stat?` |
| monster drop | `item_id`、`probability?`、`quantity_min`、`quantity_max` |
| map connection | `connection_id`、`destination_map_id`、`travel_kind` |
| platform | `platform_id`、`x1`、`x2`、`y` |
| portal | `portal_id`、`destination_map_id`、`destination_portal_id?`、`x`、`y` |
| equipment | `slot`、`required_level`、`required_class_id?`、`attack?`、`defense?` |
| consumable | `hp_restore?`、`mp_restore?`、`duration_ms?` |
| quest requirement | `requirement_id`、`kind`、`target_id?`、`quantity` |
| asset declaration | `asset_id`、`role` |
| validation evidence declaration | `evidence_id`、`method`、`artifact_sha256`、`source_uri`、`declared_status` |

技能 `level` 必须覆盖 `1..max_level`，不推导缺失等级；等级上限为 200。正弹药消耗必须
给出同版本弹药 item 引用。效果种类为 `damage/heal/buff/debuff/summon/movement/resource/
damage_over_time`，target 为 `self/enemy/ally/area`。这里只存字段，不解释或执行技能。

掉落概率在 `[0,1]` 内或明确未知；数量范围至少为 1 且下界不高于上界。地图尺寸至多
1,000,000；平台须满足 `0 <= x1 < x2 <= width` 且 y 在地图内。传送门目的地图必须存在，
指定目的门时还要存在该门。NPC 坐标根据其引用的地图检查。地图连接可形成循环；
职业转职及任务前置关系拒绝循环。结构关系成立不证明该地图可以导航或任务可完成。

item category 为 `equipment/consumable/ammunition/material/currency/quest/other`，
必须与 equipment/consumable 子对象的有无一致。任务 requirement kind 为
`item/monster/quest/class/level`；只有 level 要求的 `target_id` 必须为 null，其他种类
必须指定对应实体。任务要求分别映射到独立的 item/monster/quest/class 外键列。

mechanics 的 numeric/text 值最多有一个非 null；两者均 null 表示规则值未知。
这里没有执行表达式、脚本或 SQL 的字段。

`asset_index` 仅记录显式声明的 asset ID 和用途；不会抓取、解码或核实对应图片。
`validation_evidence` 也仅保存来源提供的证据定位、哈希与状态声明，不把该 URI 当作已验证文件。
manifest 固定说明 `asset_bytes_verified=false`、`evidence_artifacts_verified=false`、
`runtime_rules_compiled=false` 和 `real_game_accuracy=not_established`。

## 来源、关系与映射分母

每张领域表均带 `scope`、`game_version`、所属根 `entity_id`。
主键及外键包含相同的 scope 和 version，跨版本名称碰巧相同也不能满足引用。
子表通过附加 `level`、`effect_id`、`portal_id` 等组成复合主键。

辅助表为 `game_versions`、`knowledge_snapshots`、`source_records`、`entity_sources`、
`projection_issues`。原 v2 的全部 `catalog_records` 都进入 `source_records`；P0 legacy
源行保留在归档的原数据库，不冒充包含领域 payload 的证据行。本模块没有产生同步作业，
因此不创建虚构 `sync_jobs` 记录；网络同步和持久调度仍是后续工作。

报告分别列出来源证据行数、已映射/未映射来源行数、实体版本总数、已投影/未映射实体版本数、
问题分类和各业务表行数。一条实体可以有多个相同 payload 的来源，也可以同时有已验证来源
和待核实来源；两个分母因此不同。`all_source_records_mapped` 只说明这个输入的全部证据记录
是否满足投影合同，不表示“游戏数据完整”。零输入会标记 `mapping_status=empty_source`，
该布尔值为 false。允许 null 的字段即使成功投影仍保持未知。

## 独立复验、只读与原子发布

验证首先检查严格 manifest、固定路径、文件类型和字节预算，再对读入的同一份源数据库
字节复用 catalog 的 schema、记录哈希、状态和冲突检查。之后从归档的原始 payload
重新生成所有领域表，核对实际 SQLite schema、每张表的全部有序行及报告。
单独修改 `domain.sqlite` 并重新计算所有外层哈希不能通过这个语义复验。
完整替换全部来源与结果并重新构造一致快照，属于另一份新快照；哈希不提供签名认证。

`open_domain_readonly()` 在通过验证的数据库字节上创建内存连接，并设置 `query_only`
与读操作 authorizer；写入、ATTACH 及关闭 query_only 的 SQL 请求拒绝。调用方负责关闭连接。
验证后不重新打开原路径，因此并发替换磁盘文件不会改变已经返回的查询快照。
这属于数据库接口约束，不宣称是进程权限沙箱。

`query_domain()` 的表名来自固定允许列表；scope、version、entity 和 limit 均使用 SQL 参数。
一次最多返回 1000 行。大量连续查询可复用上面的已验证连接，避免每次重建投影。

输入源数据库上限 128 MiB，源 manifest 64 KiB，源证据与 legacy 行合计最多 100,000；
领域库上限 256 MiB，累计投影行最多 250,000，单个嵌套集合最多 1000。
拒绝非有限数值、bool 冒充整数、符号链接、额外文件和 SQLite WAL/SHM/journal 旁文件。

构建先在内存事务与临时目录内完成，成功复验后独占保留目标目录，最后发布 manifest 作为
完成标记。已有目标不会覆盖；发布失败删除本次部分输出；源快照始终只读。

## 专项验证与剩余范围

自写的完整合成导出 fixture 位于 `tests/test_catalog_domains.py` 的 `authored_export()`；
覆盖 14 个实体、八种来源 kind、全部业务关系。测试检查缺失子字段撤回、冲突、外键、
跨版本/scope、循环、几何、查询注入、只读语义、源/结果篡改和原子失败。

```bash
python -m pytest tests/test_catalog_domains.py -q
python -m ruff check src/maple_vision/catalog_domains.py tests/test_catalog_domains.py
python -m mypy src/maple_vision/catalog_domains.py
```

本包完成的是离线领域投影接口及其可复验性。小册子实际资料、获准接口、同步调度、
完整职业机制、资产文件验证、知识审阅冲突解决、Runtime 能力编译和真实领域正确率仍未完成。
真实数据到达时应转换为这个明确的导出合同并独立审阅，不能仅因导入成功就提升其验证状态。
