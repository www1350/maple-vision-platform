# 版本化素材索引与动画内容包

本工作包补充原始总纲第 04 章的 SQLite 素材索引，以及第 05 章的动画帧、时长和锚点。`asset_index.py` 与 `animation_pack.py` 提供独立内容接口，`scene_animation.py` 把显式类别绑定接入渲染、数据集、训练副本和可复验运行。省略动画输入时原有行为保持兼容。

仓库的 `examples/animations/fixture` 是代码自绘的几何人物，共四张 24×24 透明 PNG、三个动作片段。它们不是冒险岛素材，不包含游戏文件、下载图片或描摹资源，也不代表真实游戏帧率、动作规则或真实素材许可验收。示例登记为 `synthetic_fixture`，使用 `self_authored` 来源声明与 `CC0-1.0` 许可声明。

## 1. 内容与信任边界

原始图片仍按内容 SHA-256 存在 `Asset Registry` 文件系统中，元数据以 `AssetEntry.asset_id` 绑定实体、版本、来源、许可、图片哈希、尺寸和格式。新的 SQLite 文件是查询索引。写入索引前、查询返回时和全量验证时，都会核对注册表记录、实际图片哈希与解码结果。

哈希用于验证内容一致性。来源 URI、许可及授权字段仍是导入者的声明，不是著作权、官方来源或使用权的密码学证明。模块不会访问这些 URI，也不会下载或解包素材。

动画包导出后包含实际原图、完整 `AssetEntry`、RGBA 哈希、透明边界和帧定义。核验只读取包内数据，删除原始图片、映射文件和注册表后仍可使用。导出的包不包含 SQLite 索引；索引的内容权威仍是导入注册表。

## 2. SQLite 索引 API

```python
from pathlib import Path
from maple_vision.asset_index import index_assets, query_assets, verify_asset_index

result = index_assets(
    Path("assets.sqlite"),
    Path("registry"),
    (first.asset_id, second.asset_id),
)
entries = query_assets(
    Path("assets.sqlite"),
    Path("registry"),
    entity_id="fixture.player",
    game_version="synthetic-animation-v1",
    limit=256,
)
all_verified = verify_asset_index(Path("assets.sqlite"), Path("registry"))
```

`index_assets` 返回不可变 `IndexResult(imported, unchanged, total)`。同一批中的重复 ID 会被拒绝；不同调用中重复导入相同记录不产生新行。新数据库先完整创建再发布，已有数据库使用事务处理整批导入。源数据验证失败不会创建数据库，已有内容也不会被覆盖。

`query_assets` 按 **精确实体 ID + 精确游戏版本** 查询，不选择“最近版本”作为替代。结果按 `asset_id` 排序；将上一页最后一个 ID 传入 `after_asset_id` 可继续分页。空结果明确表示该版本没有匹配条目。每个返回条目都重新验证注册表，查询不等于整库核验；`verify_asset_index` 会重核整个索引。

SQLite 使用专用 `application_id`、`user_version=1`、`asset_entries` 表和 `asset_entity_version` 索引。验证会核对规范 SQL、检查额外表/视图/触发器、字段预算以及数据库完整性，拒绝错误版本、错误表结构、被改写的索引行和源数据。该专用索引不启用 WAL；带 `-wal`、`-shm` 或 `-journal` 旁文件的数据库会被拒绝，应在写入连接完成并关闭后再查询。不会静默迁移或覆盖其它数据库。

## 3. 输入动画映射

`load_animation_pack(registry, mapping)` 接收 JSON 文件，顶层字段必须为：

```json
{
  "schema_version": 1,
  "scope": "synthetic_fixture",
  "clips": [
    {
      "clip_id": "fixture.player.idle",
      "entity_id": "fixture.player",
      "game_version": "synthetic-animation-v1",
      "class_id": "player",
      "action_state": "idle",
      "playback": "loop",
      "frames": [
        {
          "asset_id": "REPLACE_WITH_IMPORTED_64_CHARACTER_LOWERCASE_SHA256",
          "duration_ms": 100,
          "anchor": [12, 23]
        }
      ]
    }
  ]
}
```

示例中的 `asset_id` 占位符必须替换为真实登记 ID；可直接运行本文末尾的构建脚本获得完整可验证样例。

`schema_version` 必须为整数 1，布尔值和浮点数均不接受。重复 JSON 字段、额外字段和未知枚举会被拒绝。`scope` 为 `synthetic_fixture` 或 `authorized_assets`。后一个值表示使用者声明其素材获准使用，本模块不会自动验证法律事实。

每个片段的所有帧必须与声明的 `entity_id` 和 `game_version` 一致。同一版本的实体只能对应一个检测类别；同一实体、版本、动作态不能有两个候选片段。片段 ID 也必须唯一。

当前类别沿用静态素材的七类：`player`、`monster`、`loot`、`portal`、`npc`、`platform`、`ladder`。动作态可使用现有 `SimAction` 名称，以及 `hurt`、`death`、`spawn`、`despawn`。这些是内容关联键：声明一个 `attack` 动画不会自动添加攻击规则或让模拟器执行攻击。

每张来源图必须是单帧 PNG、JPEG 或 WebP。动画容器不能直接充当一帧；应先按帧登记。完全透明的图片被拒绝。帧的可见边界由非零 alpha 像素推导，采用半开区间 `(left, top, right, bottom)`。

## 4. 锚点与确定性时间轴

`anchor=[x,y]` 使用图片的像素边界坐标，必须满足 `0 ≤ x ≤ width`、`0 ≤ y ≤ height`。允许脚底锚点恰好处于图片的底边。它不是碰撞框，也不改变游戏几何。

```python
pack = load_animation_pack(Path("registry"), Path("mapping.json"))
clip = pack.resolve("fixture.player", "synthetic-animation-v1", "idle")
if clip is None:
    raise ValueError("this entity/version has no idle animation")
sample = clip.sample(125)
image = sample.frame.source.image()
anchor = sample.frame.anchor
```

`sample` 接收非负整数毫秒，不读墙钟，也不使用浮点插值。帧区间左闭右开，例如帧时长为 80 和 120 毫秒时：

| 时间 | 循环播放结果 |
|---|---|
| 0–79 ms | 第一帧 |
| 80–199 ms | 第二帧 |
| 200 ms | 下一轮第一帧 |

`playback="once"` 在整个片段结束后保持最后一帧，同时返回 `finished=True`。它不会隐式消失、返回 idle 或改变任务状态。`AnimationSample` 另含 `frame_index`、`cycle_index` 和 `frame_elapsed_ms`。渲染器未来可消费这些结果，并自行定义状态进入时刻、动作选择和缺失动作策略。

## 5. 合成接口和 alpha 边界

```python
from maple_vision.animation_pack import AnimationPlacement, compose_animation

composition = compose_animation(
    pack,
    (
        AnimationPlacement("fixture.player.idle", 125, (48, 70)),
        AnimationPlacement("fixture.player.attack", 90, (96, 70), layer=1, flip_x=True),
    ),
    size=(160, 100),
)
composition.save_png(Path("composition.png"))
```

`position` 指定锚点在画布中的位置。合成按 `layer` 升序，同层按输入顺序绘制，使用 source-over alpha 合成。水平翻转同时将锚点变为 `width-anchor_x`。不进行缩放；屏幕外的部分被裁切，完全离开画布的贴图返回 `visible_bbox=None`。

`composition.placements` 为每次贴图提供片段、帧序号、是否结束、完整画布位置和裁切后的 alpha bbox。**这个 bbox 表示该贴图自身的非透明裁切范围**；后绘制的对象可能把它覆盖，因而它不是全场景遮挡后的实例标注。`composition.visible_bbox` 则来自最终 RGBA 画布的 alpha，若给定不透明背景，它会包含背景区域。

相同包、整数时间、摆放顺序和背景会产生相同像素。同一软件环境的 PNG 输出可重复；跨 Pillow/zlib 版本的编码字节不作为额外保证。原始素材字节始终原样保存。`image()` 返回新图片对象，修改它不会污染包内不可变像素。

## 6. 自包含导出与验证

```python
from maple_vision.animation_pack import export_animation_pack, verify_animation_pack

export_animation_pack(pack, Path("exported-animation"))
restored = verify_animation_pack(Path("exported-animation"))
assert restored.manifest() == pack.manifest()
```

输出目录包含 `animation-pack.json` 和 `animation_objects/<content_sha256>.<extension>`。相同原图内容可以被多个来源记录引用，导出时只复制一次原图文件；各记录的来源声明仍分别保留。

Manifest 保存完整片段定义、每个来源的 `AssetEntry`、内容路径、RGBA SHA-256、alpha bbox，以及整份规范 JSON 的 `pack_sha256`。验证会重新读实际原图、解码 RGBA、计算边界，再核对重建结果。即使修改者重新计算外层 pack hash，错误尺寸、错误原图、虚构 alpha bbox、错误实体/版本和错误路径仍会被拒绝。

输出目录必须不存在。源文件写入和 manifest 写入失败时，会尽力清理本次创建的文件与空目录；不会递归删除出现的无关文件。清理本身若因文件系统故障失败，可留下不能通过核验的残留目录，不会返回成功结果。Manifest 最后写入，调用返回前不应把正在生成的目录用于消费。

未来嵌入其它 bundle 的接口是 `pack.write_sources(existing_output)` 与 `pack.manifest()`；验证可调用 `verify_animation_pack(bundle, manifest)`，无需独立 manifest 文件。`animation_objects` 必须是新目录，核验拒绝其中未列出的文件和子目录。其余 bundle 文件不属于动画包哈希范围。

## 7. 明确的资源边界

| 项目 | 上限 |
|---|---:|
| SQLite 主文件 | 128 MiB |
| SQLite 索引条目 | 4,096 |
| 每批索引／查询页 | 256 |
| 每条来源元数据 | 64 KiB |
| 动画 manifest | 512 KiB |
| 每包片段 | 32 |
| 每片段帧数 | 128 |
| 每包帧引用 | 512 |
| 每包来源记录 | 256 |
| 单张动画来源编码字节 | 8 MiB |
| 所有动画来源编码字节合计 | 32 MiB |
| 单帧宽／高 | 1,024 px |
| 单帧解码像素 | 1,048,576 |
| 所有来源解码像素合计 | 8,388,608 |
| 片段总时长 | 60,000 ms |
| 单次合成画布像素 | 4,194,304 |
| 单次贴图数量 | 128 |
| 画布加贴图的处理像素合计 | 16,777,216 |

来源预算按来源记录计数，多个记录引用同一原图也会保守计入总预算。计数、声明尺寸和文件字节预算在解码前检查，实际读入长度也会再次检查。帧内容哈希计算前使用有限长度读取；SQLite 元数据在被取回 Python 前检查字节上限。路径必须为内容寻址路径，符号链接、目录逃逸、未列出文件和超预算内容不会被作为来源使用。

## 8. 可复现的自绘样例

在安装项目依赖后，从仓库根目录执行：

```bash
python examples/animations/build_fixture.py /tmp/animation-demo-new
python -m pytest tests/test_asset_index.py tests/test_animation_pack.py
```

脚本用矩形和像素点绘制人物，登记四张图、核验 SQLite 索引、构建 idle/move/attack 三个片段，随后导出动画包。脚本会先删除临时注册表、原始图和索引，再从导出包重新核验并生成 `preview.png`。预览图是便于查看的派生图片，不在动画源 manifest 的哈希范围内。

已提交的同类样例位于 `examples/animations/fixture`。模型对新外观的适配、真实授权素材和真实域效果仍需要各自验收。核心 clip 模块的测试不能代替下面的跨层渲染与证据测试。

## 场景、数据集与运行集成

`SceneAnimation` 将 1–7 个渲染类别映射为明确的 clip。`bind_scene_animation(pack, {"player": "clip.id"})` 校验类别与片段声明一致；`export_scene_animation` 原子写入 `scene-animation.json` 和源图片，`load_scene_animation` 核验全部身份和内容。场景清单最多 1MiB，加载先限额再解析。

```bash
python -m maple_vision animations scene examples/animations/fixture bindings.json artifacts/scene
python -m maple_vision dataset create artifacts/animated-data --profile lifecycle --sessions 3 --frames 12 --animation-scene artifacts/scene
python -m maple_vision dataset check artifacts/animated-data
python -m maple_vision lab run --runtime structured --animation-scene artifacts/scene --output artifacts/animated-run
python -m maple_vision lab reinfer artifacts/animated-run
```

`bindings.json` 需写入包中真实的类别与 clip_id。这些命令验证格式和执行链；样例画风不一定被当前颜色基线识别，任务可能正确暂停或失败。闭环验收使用另行自绘、保留原调色板和实体可见边界的两帧动画，不声称任意素材兼容。

渲染按 `snapshot.timestamp_ms` 的绝对时间选择帧，不依据未实现的攻击动画状态推断时钟。原生尺寸图片以 anchor 附着逻辑实体的底边中心；镜头坐标变换与透明度合成后再计算标签。全不透明像素遮掉下层实例，部分透明像素保留双方贡献，透明洞不遮挡；HUD 和对话框另行擦除标签。锚点不改变碰撞几何。

动画输入优先于同类别静态 SpritePack，没有绑定的类别继续原绘制路径。`animation=None` 的旧存档、像素和数据生成继续兼容。lifecycle 动画数据用 schema4，既能嵌入外部地图也能使用内置两图；重放校验每张图片与标签，训练私有快照复制完整动画源。旧 schema 拒绝夹带未声明的动画证据，包括悬空符号链接。

run/suite/soak 的动画元数据进入摘要；完整 source 和 sidecar 随运行包保存。动作重放重新渲染，reinfer 另外重建所有 N+1 帧的观测和任务，均无需原始素材目录。只重算文件哈希不足以掩盖 sidecar、summary、源像素或时间轴之间的矛盾。[独立跨层回归](../tests/test_content_integration_review.py)
