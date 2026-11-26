# API 合同

本页先说明 v0.5 状态/任务与 v0.4 外部地图接口，再列出 v0.2/v0.3 增量接口、继续兼容的 v0.1 接口与 P0 历史接口。历史代码块列出对应版本的调用方式；当前新增参数以前面的增量合同为准。数据字段及格式见 [DATA_SCHEMAS.md](DATA_SCHEMAS.md)，实现边界见 [ARCHITECTURE.md](ARCHITECTURE.md)。
新 RGB 视觉链路与 P0 NoOp 链路分开：`scene.ImageFrame` 不是 `contracts.Frame`，`simulation.Simulator.step(action)` 也不是旧占位协议的 `step() -> Frame`。调用方应显式选择对应 API。

## v0.5 时序状态与任务接口

`runtime_pipeline.RuntimeConfig` 是新配置，不是旧 P0 的同名配置类；其完整设置和内容哈希随运行保存。

```python
from maple_vision.runtime_pipeline import RuntimeConfig
from maple_vision.state_estimation import TemporalStateEstimator
from maple_vision.task_runtime import TaskRuntime

belief = TemporalStateEstimator().update(observation, now_ms)
runtime = TaskRuntime(job="assassin", maps=("meadow", "cavern"))
decision = runtime.decide(belief, now_ms)
# 每个非空 intent 必须先 dispatch；仅执行 accepted 返回的 action。
authorization = runtime.scheduler.dispatch(decision.intent, now_ms)
# 执行后仅传入真实 accepted/reason/time，不把 snapshot/events 交给 runtime。
runtime.record_execution(decision.intent.intent_id, accepted, reason, now_ms)
```

上例描述边界，调用方必须处理 `intent is None` 的等待、暂停和终止，以及 dispatch 拒绝。完整可运行入口为 `lab.run_episode(..., runtime=RuntimeConfig())`；`run_suite` 与 `run_soak` 同样接受该可选参数。`None` 保留旧默认，`RunConfig` 本身不迁移。CLI run/suite/soak 使用 `--runtime structured`；replay/reinfer 从证据恢复，禁止猜测模式。

`TaskRuntime.interrupt(reason, now_ms, terminal=False)` 取消在途计划；非终止暂停可由新观测恢复，终止状态锁定。`snapshot()` 只导出当前状态，不补做策略决策。动作效果使用 SUCCESS/LIKELY_SUCCESS/UNKNOWN/FAILED/TIMEOUT；当前合成观测没有库存、Buff 或击杀事件证据，相关效果保持 UNKNOWN，不以接受回执或框消失代替确认。具体字段和限制见 [状态合同](STATE_ESTIMATION.md)、[任务合同](TASK_RUNTIME.md)。

## v0.4 外部地图接口

```python
# maple_vision.map_pack：输入为本地数据，不执行脚本，也不读取来源 URI
load_map_pack(path: Path) -> MapPack
MapPack.from_dict(value: object) -> MapPack
MapPack.from_manifest(value: object) -> MapPack
pack.get(map_id: str) -> MapDefinition
pack.to_dict() -> dict[str, object]
pack.manifest() -> dict[str, object]
# 只读属性：map_ids / route、entry_map、markers、content_sha256、width / height / ground

# 外部地图仍使用固定的 320×192 合成世界
Simulator(seed=0, job="assassin", map_id="meadow", step_ms=50,
          *, class_spec=None, map_pack=None)
SceneRenderer(width=320, height=192, style_seed=0,
              *, sprite_pack=None, map_pack=None)
PixelPerception(*, map_markers=None)
HybridPerception(detector, *, terrain_source="fixture", map_markers=None)
VisualAgent(job="assassin", maps=("meadow", "cavern"), *, class_spec=None)

# 数据集与闭环入口均可显式绑定同一个不可变 MapPack
generate_dataset(output, sessions=9, frames_per_session=8, seed=0,
                 *, sprite_pack=None, profile="legacy", map_pack=None) -> dict
run_episode(output=None, config=None, detector=None, *, class_profile=None,
            sprite_pack=None, map_pack=None, terrain_source="fixture") -> dict
run_suite(output, seeds=3, detector=None, *, terrain_source="fixture",
          map_pack=None) -> dict
run_soak(output, seconds=60, detector=None, *, resource_interval_seconds=1.0,
         max_resource_samples=1024, terrain_source="fixture", map_pack=None) -> dict
```

`load_map_pack` 接受不超过 128 KiB 的普通 UTF-8 JSON 文件，拒绝符号链接、重复键、未知字段和非法数值。`from_dict` 验证原始地图内容；`from_manifest` 接受 `{"schema_version":1,"content":...,"content_sha256":...}`，验证内容后重新计算摘要。`manifest()` 返回可独立恢复的完整内容，不能把它误当作 `load_map_pack` 所需的原始输入格式。哈希基于规范化 UTF-8 JSON，来源和许可字段也属于身份的一部分；它证明内容一致，不验证声明者的实际权利。

地图包 v1 固定宽 320、高 192、地面 y=168，含 2–8 张地图。`route` 恰好遍历每张地图一次，每张地图的唯一传送门指向路线下一张，最后一张返回入口。`maps` 的保存顺序规范化为路线顺序。`MapDefinition` 包含 id、theme、marker、spawn_x、platforms、portals、npcs、ladders、monster_x；结构与数量约束见 [数据格式](DATA_SCHEMAS.md#外部地图包-v1)。内置 meadow/cavern 不会被外部同名数据替换。

`Simulator` 的默认 map_id 仍是 `meadow`。因此绑定外部包时应显式调用 `Simulator(map_id=pack.entry_map, map_pack=pack)`；构造器也允许选择包内其它地图作为该实例的初始图。`reset()` 保留包和初始地图，传送门使用目标地图的 spawn_x。`simulator.map_pack` 和 `simulator.map_ids` 是只读静态数据属性。外部地图渲染器要求世界及视口均为 320×192，不能借助修改 width/height 引入未实现的宽世界探索能力。

外部地图的 NPC 与传送门交互还要求角色和目标在垂直方向具有正面积交叠；传送门保留横向距离及角色着地检查。角色站在目标正上方的远处高台，或仅与目标边缘接触，不会触发交互。内置地图的历史动作行为保持原样。

依赖边界保持明确：模拟器和渲染器接收完整包；`PixelPerception` / `HybridPerception` 只接收 `pack.markers` 的只读颜色映射；`VisualAgent` 只接收 `pack.map_ids` 的有序地图 ID，不接收平台坐标、出生点、怪物位置或 simulator。运行入口负责注入这些静态信息，策略仍只通过 RGB 产生的 `VisualObservation` 决策。地图标记解析属于合成 HUD，不是通用 OCR。

外部数据集必须选择 `profile="lifecycle"`，使用 manifest schema3、`generator_version="lifecycle-map-pack-v1"`。每会话至少 `6 × 地图数` 帧，每张地图必须具有 spawn/combat/loot/cleared/dialogue/transition 六阶段，并保留实际命中的战斗帧。生成轨迹完成整条路线后返回入口；每会话最多 4096 个动作、整个数据集最多 1000000 个动作。结构合法不保证轨迹可解；资源、覆盖或步数不足会失败并清理暂存输出。校验器从嵌入包独立重放动作、像素、标签和阶段，不能仅通过重新计算哈希伪造覆盖。评估中的地图分组来自已验证包，按会话拆分不等于按地图留出泛化验证。

运行包保存 `map-pack.json`，initial/final state 使用 schema3，summary 可选 `map_pack`，感知 settings 增加 `map_pack_sha256`。动作回放 `verify_run(output)` 与感知重执行 `verify_perception(output, detector=None, *, terrain_source="fixture")` 都从包内证据恢复地图，无需再提供原始 JSON。前者验证环境与像素，后者还检查地图绑定后的感知 settings 并真实重执行 N+1 帧感知与策略。suite/soak 使用同一地图身份及按路线设置的任务预算；运行完成仍须通过独立目标检查。

`map_pack=None` 保留内置地图、旧保存版本和旧数据格式。新增地图不会改变既有模型的准入结论，也不会自动证明其在新地图上的识别能力。

## v0.2/v0.3 增量接口

```python
# 本地知识增量计划；不会访问网络来源
plan_sync(database, source_json, output_plan) -> dict
apply_sync(database, plan_path) -> dict

# 默认继续 legacy；lifecycle 必须满足自己的帧数/覆盖合同
generate_dataset(output, sessions=9, frames_per_session=8, seed=0,
                 *, sprite_pack=None, profile="legacy") -> dict

# 地形预检只接受 train/val，iterations=0 表示不做计时
audit_terrain_architecture(dataset, model_config, output, *, split="val",
                          image_size=320, warmup=3, iterations=20, seed=42) -> dict
benchmark_dataset(dataset, output, detector=None, split="test",
                  confidence_threshold=0.25, iou_threshold=0.5,
                  *, terrain_metrics=False) -> dict
evaluate_terrain_geometry(samples, image_sizes, *,
                          confidence_threshold=0.25, iou_threshold=0.5) -> dict

# 缺少模型时只允许默认 fixture；模型对象模式不补颜色地形
HybridPerception(detector, *, terrain_source="fixture")
PixelPerception().observe_hud(frame) -> VisualObservation
run_episode(output=None, config=None, detector=None, *, class_profile=None,
            sprite_pack=None, terrain_source="fixture") -> dict
run_suite(output, seeds=3, detector=None, *, terrain_source="fixture") -> dict
run_soak(output, seconds=60, detector=None, *, resource_interval_seconds=1.0,
         max_resource_samples=1024, terrain_source="fixture") -> dict
verify_perception(output, detector=None, *, terrain_source="fixture") -> dict
```

`terrain_source` 严格为 `fixture` 或 `model`。默认保留原模式；`model` 模式把七类对象全部交给检测器，HP/MP、地图与对话标记仍是 fixture 像素解析。该模式的完整模型 manifest 必须包含七类，非法/越界输出拒绝，不转为像素结果。`observe_hud` 不检测对象，其自身结果为 UNKNOWN；组合后 KNOWN 也不等于全部地形均已检出。[模式合同](MODEL_TERRAIN.md)

`verify_perception` 真实重新执行 N+1 帧感知、监督、策略和动作；原有 `verify_run` 仍只核对动作及环境。新模式必须显式指定同一 `terrain_source`，错误模式在首次检测前拒绝；默认旧证据格式保持兼容。[验证合同](MODEL_VALIDATION.md)

原生 `OnnxDetector.loaded_model_sha256` 是传给 ORT 的实际字节身份；只读 `runtime_settings` 返回实际 session providers、intra/inter-op 线程和执行模式。`terrain_metrics=True` 的 schema3 报告绑定这些运行选项与完整 manifest，并在评分前后核对身份。地形几何分层复用同批预测，不再次调用检测器。[地形指标](TERRAIN_EVALUATION.md)

知识计划的基线检查与导入在一个写事务中执行；schema、并发、严格幂等、冲突及体积边界见 [KNOWLEDGE_SYNC.md](KNOWLEDGE_SYNC.md)。lifecycle 格式和实际轨迹覆盖见 [DATASET_LIFECYCLE.md](DATASET_LIFECYCLE.md)。结构检查与检测评分输出为新的 JSON 文件；其通过不自动授予模型推广或真实域准入。

## v0.1 本地数据接口

```python
# maple_vision.catalog
migrate_catalog(connection: sqlite3.Connection) -> None
initialize_catalog(path: str | Path) -> None
open_catalog_readonly(path: str | Path) -> sqlite3.Connection
load_import(source: str | Path) -> tuple[CatalogRecord, ...]
import_catalog(path: str | Path, source: str | Path) -> ImportResult
query_catalog(path, kind, entity_id, game_version, *, allow_unverified=False)
catalog_summary(path: str | Path) -> dict[str, object]
create_snapshot(path, destination_dir) -> dict[str, object]
verify_snapshot(destination_dir) -> dict[str, object]

# maple_vision.asset_registry
import_asset(registry_dir, source_path, *, entity_id, game_version,
             source_uri, license, permission) -> AssetEntry
verify_asset(registry_dir, asset_id) -> AssetEntry
list_assets(registry_dir) -> tuple[AssetEntry, ...]
```

catalog 必须先初始化至 v2，import 不会因路径拼错而创建新库；整批校验后事务导入，相同记录幂等，冲突记录保留。
默认 query 只返回 VERIFIED，`allow_unverified=True` 仅用于检查。来源和许可是调用方可审阅的断言，系统不会自动把公开网址等同于使用许可。
`open_catalog_readonly()` 的连接由调用方关闭；快照输出目录必须不存在，父目录必须已存在。
素材注册保留原始图片 bytes 并核对解码结果；`load_sprite_pack(registry: Path, mapping: Path) -> SpritePack` 可把类别到 asset_id 的映射加载为验证过的静态 RGBA 素材。
`load_class_profile(database, job, game_version) -> LoadedClassProfile` 只加载指定版本、VERIFIED 且无冲突的完整合成 ClassSpec，保留配置哈希和来源；不会隐式降级到未知或冲突数据。

## v0.1 模拟器、图像与感知

```python
# maple_vision.simulation / sim_types
Simulator(seed=0, job="assassin", map_id="meadow", step_ms=50, *, class_spec=None)
simulator.reset(seed: int | None = None) -> None
simulator.step(action: SimAction | None = None) -> StepOutcome
simulator.snapshot() -> SimSnapshot
simulator.save() -> dict[str, Any]
Simulator.restore(state: object) -> Simulator
SimAction(kind="idle", direction=0, skill="basic", target="")

# maple_vision.scene
ImageFrame(sequence, timestamp_ms, width, height, pixels)
ImageFrame.from_file(path, sequence=0, timestamp_ms=0) -> ImageFrame
frame.to_image() -> PIL.Image.Image
frame.png_bytes() -> bytes
SceneRenderer(width=320, height=192, style_seed=0, *, sprite_pack=None)
renderer.render(snapshot: SimSnapshot) -> ImageFrame
renderer.render_sample(snapshot: SimSnapshot) -> RenderedSample
renderer.annotations(snapshot: SimSnapshot) -> tuple[Detection, ...]

# maple_vision.vision / onnx_detector / agent
PixelPerception().detect(frame: ImageFrame) -> tuple[Detection, ...]
PixelPerception().observe(frame: ImageFrame) -> VisualObservation
OnnxDetector(model: Path, manifest: Path,
             *, confidence_threshold=0.25, iou_threshold=0.45)
detector.detect(frame: ImageFrame) -> tuple[Detection, ...]
HybridPerception(detector).observe(frame: ImageFrame) -> VisualObservation
```

内置 job 为 assassin/dragon_knight/cleric；map_id 为 meadow/cavern；step_ms 是 10–100 的整数。
动作 kind 支持 idle/move/jump/attack/jump_attack/loot/potion_hp/potion_mp/buff/heal/interact/portal/climb；参数组合在 SimAction 构造时校验。
StepOutcome 返回 snapshot、accepted、reason、events。动作拒绝或角色死亡时 step 仍推进一个固定时步；accepted 不等同任务成功。

snapshot 和渲染标签只允许进入数据制作或独立验证；策略只拿 `VisualObservation`。
`HybridPerception` 使用 ONNX 对象框与合成场景的可见 HUD、地图及对话标记，不能当成真实游戏 UI/OCR 实现。
ONNX manifest 校验哈希、类别、输入尺寸和输出格式。输入为 float32 NCHW RGB，letterbox 后解码回原始像素；不存在自动下载模型。

## v0.1 策略、监督和运行包

```python
Supervisor(max_unknown_steps=12, max_age_ms=250)
supervisor.inspect(observation, now_ms: int) -> GuardDecision
VisualAgent(job="assassin", maps=("meadow", "cavern"), *, class_spec=None)
policy.decide(observation: VisualObservation) -> AgentDecision
verify_visual_effect(before, after, action: SimAction) -> str

RunConfig(seed=0, job="assassin", max_steps=1000,
          fault="none", record_stride=4)
run_episode(output: Path | None = None,
            config: RunConfig | None = None,
            detector: Detector | None = None,
            *, class_profile: LoadedClassProfile | None = None,
            sprite_pack: SpritePack | None = None) -> dict[str, Any]
verify_run(output: Path) -> dict[str, Any]
run_suite(output: Path, seeds=3) -> dict[str, Any]
run_soak(output: Path, seconds=60) -> dict[str, Any]
chat_draft(incoming: str, reply: str) -> dict[str, object]
```

GuardDecision 的 state 为 running/paused/stopped，AgentDecision 包含 action/task/state/reason。
监督检查帧身份、时间、可信度、玩家唯一性和资源；连续异常计数有上限，时钟倒退和死亡可立即停止。
可见效果为 observed/pending/unknown/not_applicable，不能据此证明隐藏伤害或掉落归属。

fault 仅支持 none/brief_dropout/persistent_dropout/duplicate_frames/freeze_input/loot_false_negative。`loot_false_negative` 只在合成画面中遮掉掉落实体像素，模拟“真值有掉落但感知看不见”的受控反例。
运行任务状态为 completed/failed/stopped；completed 还必须满足独立 oracle，策略自称完成但真值不符会降为 failed。
`output=None` 只返回摘要；指定输出则保存采样图片、完整动作轨迹、状态、摘要和文件 manifest；提供 sprite_pack 时同时保存其原始图片和元数据，回放时独立验证并恢复相同渲染。
`verify_run()` 做动作/状态/像素回放，返回 `perception_reproduced=False`；不重新执行学习模型。

`run_episode(..., input_profile=InputBindingProfile(...), input_confirmation=...)` 可选地把用户确认的键位表及确定性人工确认凭证绑定到任务。确认凭证存在时必须与 profile 完全一致；运行包单列完整凭证边车，summary 只保留可检索摘要，动作 replay 和完整 reinfer 均重新验证两者。当前约束内置普通攻击、龙骑士模拟范围攻击和外部职业 `cast`；未绑定的受约束技能转换为 idle 并保留拒绝证据。其他动作不经过此表。配置、sidecar、逐步 dispatch、动作 replay 和完整 reinfer 必须一致。该参数只路由独立模拟器动作，不提供官方客户端控制后端。

`lab run --input-bindings FILE` 读取同一 schema 的有界本地 JSON 后传入上述参数。文件必须是普通非符号链接、UTF-8、最大 64 KiB；CLI 不从角色名猜测浏览器 localStorage，也不读取官方客户端配置。
v0.1 的 suite 与 soak 固定使用像素基线；当前版本可通过前述增量接口显式传入 detector 和 terrain_source。没有 detector 时仍使用像素基线。

soak 的 seconds 是墙上时间预算，范围 1–604800；会完成当前 episode 后退出，可能多运行一个 episode 的时长。模拟时长单列，不能替代 24 小时真实进程运行。
chat_draft 只封装 incoming 和 reply 为待人工确认的本地草稿，返回 `sent=False`；不生成语言模型回复、不发送消息。

## v0.1 数据集与模型工具

```python
generate_dataset(output, sessions=9, frames_per_session=8, seed=0, *, sprite_pack=None)
validate_dataset(output) -> dict[str, object]
train_yolo(dataset: Path, output: Path, model_config: Path,
           epochs=3, image_size=320, batch=8, seed=0)
export_yolo(weights: Path, output: Path, image_size=320,
            scope="synthetic_fixture")
benchmark_dataset(dataset: Path, output: Path, detector=None, split="test",
                  confidence_threshold=0.25, iou_threshold=0.5)
evaluate_predictions(samples: Sequence[EvaluationSample], *,
                     class_names=CLASS_NAMES,
                     confidence_threshold=0.25, iou_threshold=0.5)
```

以上函数返回 JSON 可序列化摘要；EvaluationSample 内含 ground_truth 与 predictions 的 Detection tuple。
生成数据集会先暂存、验证再发布，已有输出不覆盖；会话数 3–1000、每会话帧数 1–1000、总帧数上限 100000。
训练和导出是可选重依赖命令；Python API 的 image_size 默认 320，CLI 的明确默认是 160，运行报告记录实际参数。
AP 定义、统计空值、时延范围和模型域限制见 [EVALUATION.md](EVALUATION.md)。训练完成和导出成功不等同于通过闭环准入。

## 本地录像与 OCR 接口

```python
TesseractOCR(binary=None, language="eng", *, tessdata_dir=None,
             timeout_seconds=5, psm=6)
ocr.recognize(frame: ImageFrame, crop=None) -> OCRResult
import_recording(source: Path, output: Path, *, source_uri, permission, license,
                 fps=None, max_frames=300, presentation_time=False,
                 start_frame=0) -> dict[str, object]
validate_recording(recording: Path) -> dict[str, object]
iter_recording(recording: Path) -> Iterator[ImageFrame]
observe_recording(recording: Path, output: Path, detector=None) -> dict[str, object]
summarize_recording_windows(recordings, output) -> dict[str, object]
verify_recording_windows(recordings, report) -> dict[str, object]
```

OCR 使用已安装的本地 Tesseract 和语言包，不下载；crop 是原图内正面积 xyxy 整数 tuple，结果框映射回原图坐标。
录像输入为本地 PNG 目录或 MP4；只观察和记录，不接执行器。名义 FPS 时间轴不等同精确媒体 PTS，导入的来源断言也不等同独立真实域标注。

## 当前 CLI 表面

| 命令 | 主要参数 / 作用 |
|---|---|
| `demo`、`db init/check` | 原 P0 兼容入口；`db` 使用 v1 |
| `knowledge init/check DB` | 初始化/显式升级/检查 v2 |
| `knowledge import DB SOURCE` | 导入本地知识 JSON |
| `knowledge query DB KIND ID VERSION` | 可选 `--allow-unverified` |
| `knowledge snapshot DB OUTPUT` / `knowledge verify-snapshot DIR` | 生成和验证独立快照 |
| `assets import REGISTRY SOURCE` | 必填 `--entity-id --game-version --source-uri --license --permission` |
| `assets list REGISTRY` / `assets verify REGISTRY ID` | 列表和内容验证 |
| `maps check FILE` | 校验本地原始地图 JSON，输出规范化身份及完整包；不验证任务可解性 |
| `dataset create PATH` / `dataset check PATH` | create 可选 `--sessions --frames --seed --profile`，成对的 `--asset-registry --sprite-map`；外部地图使用 `--profile lifecycle --map-pack FILE` |
| `vision ocr IMAGE` | `--language --binary --tessdata-dir --timeout --crop x1 y1 x2 y2`；需要本地 Tesseract |
| `recording import SOURCE OUTPUT` | 必填来源/许可断言，PNG 目录需要 `--fps`，可选 `--max-frames` |
| `recording check DIR` / `recording observe DIR --output DIR` | 完整性验证或离线逐帧观察；observe 可选模型对 |
| `vision observe IMAGE` | 可选成对的 `--model --manifest`、`--terrain-source`；`--map-pack FILE` 仅向感知提供静态地图标记 |
| `vision benchmark DATASET --output FILE` | 可选模型对及 `--split train|val|test` |
| `model train DATASET --output DIR` | `--config --epochs --image-size --batch --seed` |
| `model export WEIGHTS --output DIR` | `--image-size --scope` |
| `lab run --output DIR` | `--job --seed --max-steps --record-stride --fault`；可选模型对、`--terrain-source --map-pack`、`--knowledge-db --game-version` 及成对的 `--asset-registry --sprite-map` |
| `lab suite --output DIR` | `--seeds`；可选模型对、`--terrain-source --map-pack` |
| `lab replay DIR` | 校验记录包并回放动作，外部地图从运行包恢复 |
| `lab reinfer DIR` | 真实重执行感知及策略；可选同一模型对和 `--terrain-source`，地图从运行包恢复 |
| `lab soak --output DIR` | `--seconds` 为实际墙上时长；可选模型对、`--terrain-source --map-pack`、`--resource-interval --max-resource-samples` |
| `chat-draft --incoming TEXT --reply TEXT` | 输出本地草稿 JSON |

`python -m maple_vision` 与安装后的 `maple-vision` 等价。正常 stdout 为一份 JSON 摘要；训练日志写 stderr。
已处理的参数、配置、文件和数据库错误返回 2；lab 的失败/停止或 suite/soak 未通过返回 1；其余正常返回 0。
套件中的预期故障停止可以构成测试通过，但不能算作任务成功。

## P0 历史兼容接口

以下保留 P0 接口的历史说明；其中的阶段性“尚未实现”仅描述 P0。v0.1 实际能力和边界以前面的当前接口为准。

### 1. 已实现的数据模型

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

### 2. 端口与依赖方向

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

### 3. 管线与时间

- 原 `run_once(source, perception)` 保留原返回语义，只执行观测。
- 新 `Pipeline.tick(now_ms=None)` 返回 TickResult；source 结束时返回 None，不再调用后续模块。
- `now_ms=None` 使用当前帧时间，供确定性演示；提供 now_ms 时必须与帧处于同一时钟域。
- `FreshnessEstimator(max_age_ms=250)` 在观测年龄大于上限时输出 STALE；等于上限仍有效。未来时间戳会被拒绝。
- 管线在控制器之前校验帧、观测、世界状态、意图的序号和时间关联。回执与验证结果也必须关联同一意图。
- 处理时间倒退会被拒绝，防止过期数据重新变新。
- NoOp 意图有效区间为 `[created_at_ms, deadline_ms)`；在截止时间执行会被拒绝，验证为 TIMEOUT。
- NoOp 意图 ID 在一次运行内唯一，跨运行用 JSONL 的 run_id 区分。

### 4. CLI 与配置

```text
maple-vision demo [--config FILE] [--frames N] [--output FILE]
maple-vision db init PATH
maple-vision db check PATH
```

`python -m maple_vision` 提供相同命令。成功退出码 0，参数、配置、文件或数据库失败退出码 2。成功时 stdout 是一份 JSON 摘要，错误在 stderr；argparse 的 help 文本除外。

TOML 根字段只允许 `schema_version` 与 `runtime`，要求 schema_version=1。runtime 只允许 mode、frames、frame_interval_ms、max_observation_age_ms。整数约束分别为 frames:1–10000、间隔:1–60000ms、最大观测年龄:0–60000ms；bool 不接受为整数。

JSONL 每行含 schema_version、run_id 和 tick，不保存原始帧。现阶段是 trace writer，并非完整 Recorder、Replay 解码器或 Failure Bundle 系统。

### 5. 数据库迁移合同

```python
migrate(connection: sqlite3.Connection) -> None
initialize(path: str | Path) -> None
open_readonly(path: str | Path) -> sqlite3.Connection
```

迁移版本 0→1 或验证既有 v1。SAVEPOINT 将 DDL 和 user_version 更新组成原子操作；外层已有事务时，迁移不替调用方 commit/rollback。调用方可以继续回滚自己的事务及迁移内容。

已有 `source_records` 采用保守的规范化 DDL 比较，允许空白及关键词大小写差异，保留原项目合法 v1 Schema；不自动接受任意“看起来类似”的表。未知版本、结构冲突返回 ValueError，原数据不修改。

open_readonly 不创建数据库，使用 SQLite URI mode=ro 和 query_only；调用方负责 close。P0 保留原 v1 中 `id TEXT PRIMARY KEY` 的行为；P1 建立写入功能前，需要显式迁移补齐 NOT NULL 等数据约束，不能假定当前 Schema 已满足完整知识库需求。

### 6. 兼容性和下一阶段

保留 Frame 的前三个构造参数、Observation 的前三个构造参数、run_once、migrate 和 Simulator 原接口。新增字段有默认值。新输入校验明确拒绝原先不应该有效的负时间、无穷值和错误枚举。

P0 尚未冻结未来地图、技能、玩法及观察融合的数据结构；它们应在后续阶段采用显式版本迁移扩展，不能为了维持文档中的理想结构而伪造实现。
