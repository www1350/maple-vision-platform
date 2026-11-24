# 离线知识目录与素材注册表

## 实现范围

`catalog.py` 提供可运行的离线知识导入、查询、冲突保留和 SQLite 快照。`asset_registry.py` 接收用户已经有权使用的本地图片，验证图片内容并保存内容寻址对象及来源元数据。两者均没有网络抓取、游戏资源提取或正式客户端接口。

`fixtures/knowledge-demo.json` 是本项目自写的 **synthetic_fixture** 数据，版本为 `fixture-v1`。其中所有技能、怪物、地图、道具数值只用于测试，不代表国服怀旧服规则。“VERIFIED” 在该范围内表示确认过本测试数据，不会把它升级为真实游戏数据。

小册子来源 `https://mxdc.dvg.cn/` 的本次访问超时，未取得可验证的数据或访问授权信息。因此没有把猜测的资料填入游戏知识库。后续取得允许使用的导出文件后，可用相同导入格式接入，并保留原始来源、日期、授权说明与核实状态；运行时始终读取本地目录或快照。

## 知识导入格式

根对象只允许 `schema_version: 1` 和 `records`。每条记录必须明确提供以下全部字段；不接受未知字段、重复 JSON 键、NaN 或 Infinity。

| 字段 | 规则 |
|---|---|
| kind | classes、skills、monsters、maps、items、npcs、quests、mechanics |
| entity_id / game_version | 非空实体 ID 与版本；同 ID 的不同版本互不覆盖 |
| source_uri | 绝对 http(s)、urn 或 file URI；不可内含账号凭据 |
| collected_at | 带时区的 ISO 8601 时间 |
| permission | self_authored、public_license 或 explicit_permission |
| license | 非空许可或授权说明；保留导入者提供的原始断言 |
| scope | synthetic_fixture 或 game_reference |
| verification_status | VERIFIED、PROVISIONAL、UNKNOWN 或 CONFLICTED |
| payload | JSON 对象；具体业务字段可按 kind 扩展 |

同一批次在写入前全部校验；数据库写入使用一个 SAVEPOINT。输入错误或数据库故障会回滚整个批次。重复导入相同数据幂等；不同来源会保留独立记录；同实体、同版本出现不同 payload 时，全部候选变为 CONFLICTED，原来源和 declared_status 保留。不会自动挑选“最新的一条”解决冲突。

payload 与记录都使用 UTF-8、排序键、无额外空格的规范 JSON 计算 SHA-256。这是项目定义的确定性格式，不声称符合外部 JSON 签名规范。记录哈希用于发现修改，**不是来源真实性的密码学证明**。

默认 `query_catalog()` 仅返回 VERIFIED 且没有冲突的记录。检查未知和冲突数据时显式传 `allow_unverified=True`，调用方仍必须检查 status、scope 与 game_version。业务运行不得把空结果当作默认真实规则。

`profiles.load_class_profile()` 已将职业类目录接入独立模拟器：仅加载三个现有职业、完整 `ClassSpec`、全部 VERIFIED 且无冲突的 synthetic_fixture 候选。返回参数和来源哈希，模拟器保存时内嵌参数，重放不再依赖数据库。其他知识种类不会自动执行或改变世界规则。详细合同见 [CLASS_PROFILES.md](CLASS_PROFILES.md)。

## API 与迁移

```python
from maple_vision.catalog import (
    initialize_catalog,
    import_catalog,
    query_catalog,
    catalog_summary,
    create_snapshot,
    verify_snapshot,
)

initialize_catalog("knowledge.sqlite")
result = import_catalog("knowledge.sqlite", "fixtures/knowledge-demo.json")
records = query_catalog("knowledge.sqlite", "classes", "assassin", "fixture-v1")
manifest = create_snapshot("knowledge.sqlite", "snapshot-001")
verify_snapshot("snapshot-001")
```

新目录为 schema v2。`initialize_catalog` 可显式迁移原 P0 v1 文件，保留 `source_records` 旧记录，并补齐主键 NOT NULL/非空约束。已有 NULL/空 ID 时拒绝迁移，保持原数据库不变。P0 旧记录缺少 payload 与完整来源元数据，因此只作为 legacy 记录保留，不伪装成可查询的新目录记录。

如果 v1 文件还扩展了引用 `source_records` 的外键、视图、触发器或额外索引，自动迁移会拒绝并保持原结构。这样避免 SQLite 重命名旧表时把扩展引用改成随后删除的临时表，或连同旧表删除扩展对象。这类扩展需要单独编写显式迁移，不能默默丢失。

P0 的 `knowledge.py`、`db init` / `db check` 和测试合同保持 v1 兼容；不要用旧 `db` 命令读已经升级的 v2 文件。目录功能使用新 `knowledge` 命令或上述 Python API。未初始化路径、未知版本和不兼容 schema 都明确失败，不静默创建或修复数据。

快照通过 SQLite backup API 获取一致数据库，转换为独立的 DELETE journal 模式，附带文件 SHA-256 和可复核统计摘要。目标目录必须不存在；manifest 最后发布，作为完成标记。运行时通过 `open_catalog_readonly` 使用 `mode=ro` 与 `query_only`，不依赖跨平台文件权限位。快照验证拒绝额外 WAL、SHM 或 journal 文件，并把通过哈希检查的同一份数据库字节加载到只读内存连接，检查 schema、记录哈希、冲突状态和统计摘要；不会另开文件时读到未参与哈希的 WAL 页。带错误哈希或额外日志的快照不能作为运行依赖。

## 素材注册表

```python
from maple_vision.asset_registry import import_asset, verify_asset, list_assets

entry = import_asset(
    "registry",
    "my-sprite.png",
    entity_id="fixture.player",
    game_version="fixture-v1",
    source_uri="urn:my-project:player",
    license="CC0-1.0",
    permission="self_authored",
)
verify_asset("registry", entry.asset_id)
```

图片支持单帧 PNG、JPEG、WEBP，最大 32 MiB / 16 百万像素；先验证解码，再写入注册表。图片原始字节保存在 `objects/<sha256>.<ext>`，来源和尺寸等元数据保存在 `records/<asset_id>.json`。同一图片的不同来源记录共用图片对象，但来源不互相覆盖。

原子 hard-link 发布避免覆盖已存在对象；重复导入会检查原字节是否一致。验证拒绝内容修改、元数据修改、非内容寻址路径与注册表内部符号链接。授权字段只记录导入者提供的说明，不能替代真实授权证明。注册表不执行其中的任何内容。
