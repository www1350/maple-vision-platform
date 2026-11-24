# 外部地图包：格式、运行与独立复验

v0.4 将独立模拟器的地图数据接入可校验的本地 JSON 包。一个包可以包含 **2–8 张地图**，按明确的循环路线连接；模拟、渲染、生命周期数据生成、视觉任务和重放使用同一份内容身份。仓库提供三个自绘场景的完整例子：[three-map-fixture.json](../examples/maps/three-map-fixture.json)。

本格式使用既有合成物理规则：世界与视口均为 **320×192**，地面高度为 **168**。它是固定规则下的外部内容接口。`maps check` 校验结构、范围和身份；一个包通过校验，并不代表当前视觉策略一定能完成其中所有任务。实际任务、故障结果和数据覆盖需要另外运行验证。

## 1. 先运行仓库示例

以下命令在仓库根目录、已经安装项目的 Python 3.12 环境中执行。每次生成使用新输出目录，避免覆盖已有证据。

```bash
python -m maple_vision maps check examples/maps/three-map-fixture.json

python -m maple_vision lab run --output artifacts/map-assassin --job assassin --seed 0 --max-steps 1500 --map-pack examples/maps/three-map-fixture.json
python -m maple_vision lab replay artifacts/map-assassin
python -m maple_vision lab reinfer artifacts/map-assassin

python -m maple_vision vision observe artifacts/map-assassin/final.png --map-pack examples/maps/three-map-fixture.json
```

示例路线是 `grove → quarry → ruins → grove`，三张地图各有三个地面怪物、一个 NPC、一个门和一条梯子。视觉任务依次完成每张地图的战斗、拾取与对话，在最后一张地图结束；数据生成器还会通过最后一个门回到入口，以覆盖最后一张地图的切换阶段。

运行结果应同时检查 `status` 与 `independent_goal_check`。`completed` 是策略结束状态，后者由独立真值核验实际清场、收集、对话和存活情况。失败任务也可以被正确重放或重新感知，不能把 `verified=true` 单独解读成任务成功。

不传模型时使用自绘调色板感知基线。`vision observe` 只把地图包的**地图 ID 与 RGB 标记对应关系**交给感知模块；图像必须具有匹配的样例 HUD，地图 JSON 不能补出缺失或未知的画面信息。

### 多职业与故障矩阵

```bash
python -m maple_vision lab run --output artifacts/map-knight --job dragon_knight --seed 1 --max-steps 1500 --map-pack examples/maps/three-map-fixture.json
python -m maple_vision lab run --output artifacts/map-cleric --job cleric --seed 2 --max-steps 1500 --map-pack examples/maps/three-map-fixture.json
python -m maple_vision lab suite --output artifacts/map-suite --seeds 3 --map-pack examples/maps/three-map-fixture.json
python -m maple_vision lab soak --output artifacts/map-soak --seconds 60 --map-pack examples/maps/three-map-fixture.json
```

`suite` 为三个已有职业运行指定数量种子的正常任务，并运行既有故障场景。`soak` 按请求墙钟持续执行任务并记录资源采样。两者都绑定地图包内容身份，并保留实际失败任务的原始证据。较短的资源运行不能计作 24 小时验收；任意有效包也没有由地图数量直接推出的成功率保证。手动 `lab run` 的步数预算由 `--max-steps` 指定，默认仍为 1000；`suite/soak` 对外部包使用 `max(1000, 500 × 地图数)`，预算耗尽必须如实记失败。

## 2. 生成并核对地图包数据集

外部地图包只接入 `lifecycle` 数据模式。显式传入 `--profile lifecycle`；省略它仍会选择兼容的 `legacy` 默认模式，并因组合不支持而拒绝。

```bash
python -m maple_vision dataset create artifacts/map-data --profile lifecycle --sessions 9 --frames 48 --seed 2026112504 --map-pack examples/maps/three-map-fixture.json
python -m maple_vision dataset check artifacts/map-data
python -m maple_vision vision benchmark artifacts/map-data --split val --output artifacts/map-pixel-val.json
```

每会话最少帧数为 `6 × 地图数`，以容纳每图的出生、战斗、拾取、清场、对话、切图六阶段；这只是最低输入预算。生成器实际执行有界动作轨迹、选择覆盖样本，再经校验后发布完整目录。无法完成轨迹或覆盖要求时明确失败，不通过修改阶段标签伪造覆盖。每会话仍最多 1000 帧，总数据最多 100000 帧。

训练数据构造允许读取模拟器真值来组织场景和生成可见标签；运行中的感知和策略不读取这些真值。`dataset check` 从嵌入地图内容恢复会话、重放动作，核对画面、标签、阶段及覆盖。`vision benchmark` 从数据集 manifest 取得地图分组，不再假定只有 `meadow/cavern`。

train/val/test 继续按会话划分；同一包中的地图可以出现在各分区。会话隔离、三个新地图和像素基线结果，不构成 YOLO 对未见地图或真实场景的准确率证据。本轮不对 v0.3 已拒绝候选追加训练、test 评分或推广验证。模型研究的原冻结结论保持独立。

## 3. JSON v1 的完整字段

完整可运行文件见[示例](../examples/maps/three-map-fixture.json)，不要将单张地图对象直接作为整个包加载。所有层级都拒绝未知字段；数值字段要求真正的 JSON 整数，`true`、浮点数、NaN 和无穷不能冒充合法坐标或版本。

### 包级字段

| 字段 | 约束和含义 |
|---|---|
| `schema_version` | 整数 `1`，表示地图文件格式版本 |
| `id` | 长度 1–40；以小写字母开头，其余为小写字母、数字、下划线或连字符 |
| `version` | 最多 40 字符的数字 `major.minor.patch`，表示作者声明的内容版本 |
| `scope` | 固定 `synthetic_fixture` |
| `source_uri` | 来源声明；接受已有目录规则支持的绝对 HTTP(S)、URN 或 file URI；不会请求或读取该 URI |
| `permission` | `self_authored`、`public_license` 或 `explicit_permission` |
| `license` | 非空许可说明；遵守现有来源字段长度和控制字符限制 |
| `width / height / ground` | 分别固定为整数 `320 / 192 / 168` |
| `route` | 2–8 个不重复地图 ID，每张地图恰好出现一次，首项为入口 |
| `maps` | 2–8 个地图定义；ID 集合必须与 `route` 完全相同 |

原始文件最多 **128 KiB**，只允许普通本地文件，拒绝符号链接。重复 JSON key、非法 UTF-8、超出解析深度和超预算内容都会失败。字段中没有模块导入、表达式、可执行脚本或外部素材文件引用。

来源、权限和许可字段是提供者的声明，校验这些字段不等于核实权属。仓库示例的来源为 `urn:maple-vision:self-authored:three-map-fixture`，权限为 `self_authored`，新地图数据的许可声明为 `CC0-1.0`；这些是独立合成内容，不是游戏地图导出或已核实的国服数据。

### 单张地图字段

| 字段 | 约束和含义 |
|---|---|
| `id` | 与包 ID 使用相同标识符规则；`meadow/cavern` 保留给旧内置地图 |
| `theme` | `meadow` 或 `cavern`，选择既有背景配色，不改变地图身份 |
| `marker` | 三个 0–255 整数组成的 RGB；包内唯一，不得使用对象、HUD 或旧地图的保留颜色 |
| `spawn_x` | 玩家左上角 x，范围 0–306；y 固定为 146，身体 14×22 |
| `platforms` | 2–8 个 `{x1, x2, y}`；第一项必须是 `{0, 320, 168}` 全幅地板 |
| `portals` | 恰好一个 `{id, x, y, w, h, destination}` |
| `npcs` | 恰好一个 `{id, x, y, w, h}` |
| `ladders` | 1–4 个 `{x, y, w, h}` |
| `monster_x` | 1–6 个地面怪物的名义 x；整数 3–299，任意两个至少相隔 24 像素 |

坐标均为世界像素；实体位置是其左上角。除第一项地板外，平台 y 为 72–167，宽至少 16，x 范围在 0–320 内；同一高度的平台不能相交或相接。碰撞几何和图片外观保持分离。

门固定为宽 20、高 36、y=132，x 为 0–300；NPC 固定为宽 14、高 24、y=144，x 为 0–306。两者 ID 在同张地图中必须不同。门目的地必须是路线中的下一张地图，最后一张回到第一张，因此不存在未解析目的地或互不连通的循环。

梯子固定宽 10，其顶端 y 必须与一个非地板平台一致，整个宽度受该平台支撑，底端连接地面。梯子 x 必须在 **2–308**：攀爬时宽 14 的玩家会对齐梯子中心，两侧各留 2 像素才能保证玩家仍在 0–306 的保存状态范围内。重复梯子拒绝。

怪物仍使用当前地面身体、血量和战斗规则，实际出生 x 会施加由会话种子确定的 `−3…+3` 像素抖动；范围和间隔校验确保这个抖动不会把怪物身体挤出视口或让初始实体完全重叠。地图包不声明怪物种类、平台上的怪物、技能或掉落表。

## 4. Python API 与信息边界

```python
from pathlib import Path

from maple_vision.lab import RunConfig, run_episode
from maple_vision.map_pack import MapPack, load_map_pack
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator
from maple_vision.vision import PixelPerception

pack = load_map_pack(Path("examples/maps/three-map-fixture.json"))
simulator = Simulator(seed=0, job="assassin", map_id=pack.entry_map, map_pack=pack)
renderer = SceneRenderer(map_pack=pack)
perception = PixelPerception(map_markers=pack.markers)

frame = renderer.render(simulator.snapshot())
observation = perception.observe(frame)
assert observation.map_id == pack.entry_map

embedded = pack.manifest()
assert MapPack.from_manifest(embedded) == pack
restored = Simulator.restore(simulator.save())
assert restored.save() == simulator.save()

result = run_episode(
    Path("artifacts/map-python-run"),
    RunConfig(seed=0, job="assassin", max_steps=1500),
    map_pack=pack,
)
```

`MapPack` 和 `MapDefinition` 是不可变数据类，集合使用 tuple，`markers` 是只读映射。`get(map_id)` 返回地图定义，未知 ID 抛 `ValueError`；`to_dict()` 和 `manifest()` 返回独立的 JSON 可序列化副本，修改副本不改变包。`from_dict()` 对已经解析的原始对象仍执行完整验证，不能靠直接构造数据类绕过约束。

| 消费者 | 可以得到的地图信息 |
|---|---|
| `Simulator(..., map_pack=pack)` | 静态地图几何、出生配置、连接和完整动态状态；执行固定规则 |
| `SceneRenderer(..., map_pack=pack)` | 地图主题、标记及待渲染 snapshot；输出纯 RGB `ImageFrame` |
| `PixelPerception(map_markers=pack.markers)` | 静态标记对应关系和 RGB 图像；没有包的碰撞几何或动态状态 |
| `HybridPerception(..., map_markers=pack.markers)` | 同一静态标记映射；对象仍按显式模型/地形来源获得 |
| `VisualAgent(job, maps=pack.map_ids)` | 有序目标地图 ID 和观测；不获取地图坐标、NPC/怪物真值 |
| 数据生成器、独立验证器 | 可以读取真值生成训练场景或核验任务；不把答案回传给运行策略 |

需要直接构造模拟器时显式传 `map_id=pack.entry_map`。构造器的默认 `map_id="meadow"` 继续服务旧调用；它不把与外部包不一致的默认值悄悄解释成另一张地图。外部包渲染要求 320×192，旧内置地图的现有视口行为保持兼容。

## 5. 内容身份、保存和迁移

`content_sha256` 是规范化内容的 SHA-256：UTF-8、排序对象 key、无多余空格、无非有限数。地图数组规范化为 `route` 顺序，因此格式化 JSON、调整对象 key 顺序或调换原始 `maps` 数组顺序不改变身份。`route` 顺序决定入口和任务次序，属于身份；坐标、实体顺序、来源或许可变动也会改变身份。

当前示例的规范化内容 SHA-256 为：

```text
e4bed8c565452dc79d29f51d2a78564546be3070168a882fa2b8215b1152e2b9
```

这个值与原 JSON 文件的字节哈希不是同一个概念。内容哈希用于完整性绑定，没有签名效力；能重新生成所有哈希的人仍可制作另一份合法包，不能把它当成可信发布者证明。

`manifest()` 的完整嵌入格式为 `{schema_version: 1, content: <完整规范化内容>, content_sha256: <哈希>}`。`from_manifest()` 同时检查外层字段、版本、完整内部格式和内容哈希。

| 载体 | 使用外部地图包时 | 不传 `map_pack` 时 |
|---|---|---|
| 原始地图 JSON | 地图格式 v1 | 无此输入 |
| 模拟器保存状态 | schema3，嵌入完整 `map_pack`；可以同时嵌入已有职业参数 | 保留普通 schema1、职业参数 schema2 |
| 生命周期数据集 | manifest schema3，`generator_version=lifecycle-map-pack-v1`，嵌入完整包 | 保留 legacy schema1 和 lifecycle schema2 |
| 任务证据目录 | 现有运行 schema1 增加地图字段及 `map-pack.json`；初末状态为 schema3 | 保留原运行格式与内置地图路径 |

运行目录中的 `map-pack.json`、summary、初始保存状态和文件清单必须一致；感知设置另绑定 `map_pack_sha256`。完整包也写入最终保存状态。`lab replay` 与 `lab reinfer` 从证据目录恢复地图，不接受另一个 `--map-pack` 替换它；删除或移动最初的地图 JSON 后，原任务仍有独立复验所需的地图内容。模型或素材若被使用，继续遵守各自的既有身份和复制规则。

`lab replay` 重放记录动作并核对状态、事件和图像，返回的 `perception_reproduced` 仍为 false。`lab reinfer` 真正重跑感知、监督与策略，包括终止帧，并要求相同地图和感知身份。修改一份地图字段、只重算外围文件哈希、或将地图身份伪装到旧 schema 中，都不能使互相矛盾的证据通过验证。

外部包版本不自动覆盖旧状态中的嵌入内容。修改地图后要生成新任务和新数据；旧证据保持原身份，不通过迁移旧报告来冒充新地图验收。

## 6. 当前边界与后续扩展

本版支持的是有限合成地图内容。大世界和滚动地图、可变尺寸、每图多门分支、平台怪物、地图编辑器、自定义背景资源、动画、新职业、通用任务 DSL 仍需单独设计合同与测试。静态 SpritePack 是已有的独立接口，可在支持它的 `dataset create` 与 `lab run` 上组合使用；它不会修改地图碰撞，也不保证调色板感知适应任意素材。

结构校验无法证明任务可行性：密集遮挡、平台布局、资源消耗、既有策略能力和步数预算都可能使合法包的任务失败。当前策略没有根据梯子检测执行通用导航；生命周期生成器可以在训练场景中主动攀爬，不能据此宣称视觉梯子导航已完成。新增内容应按“校验格式 → 生成和检查数据 → 单次任务及独立复验 → 多职业/故障矩阵 → 固定环境资源运行”的顺序保留实际证据。

实现入口为 [map_pack.py](../src/maple_vision/map_pack.py)，格式边界测试为 [test_map_pack.py](../tests/test_map_pack.py)，模拟集成为 [test_map_simulation.py](../tests/test_map_simulation.py)，数据集集成为 [test_map_dataset.py](../tests/test_map_dataset.py)。发布结果由当前版本验收文档统一记录，本文只说明接口、约束和复现方法。
