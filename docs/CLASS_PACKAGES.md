# 外部职业能力包、编译器与有限动作 DSL

`class_package.py` 实现原始总纲第 10 章的一组可执行边界：自包含职业包、选定等级的技能注册表、能力编译结果和有限复合动作程序。独立模拟器从同一包执行技能，不依赖职业名分别编写一套策略。

示例全部由本项目自绘规则、自行编写，声明 `scope=synthetic_fixture`、`permission=self_authored`、`license=CC0-1.0`。它们用于验证机制和工程边界，不代表国服、怀旧服或其他具体版本的真实技能数值。

## 当前闭环与三个差异示例

| 文件 | 职业 ID | 实际机制 |
|---|---|---|
| `examples/classes/archer.json` | `archer` | 两级远程箭技能，选定 2 级；方向与射程、MP 和弹药成本；地面箭/空中箭分别限制；jump → wait → 空中箭；限时攻击 Buff |
| `examples/classes/bandit.json` | `bandit` | 近战 slash → wait 150ms → finisher；两种不同伤害技能共享冷却组；终结技另消耗 HP；编译出累计资源和冷却偏移 |
| `examples/classes/blood_knight.json` | `blood_knight` | 普通近战；消耗 12HP 的双向范围攻击，最多 3 个目标；至少保留 1HP；注册治疗技能与 HP 上限 |

技能效果首批支持瞬时伤害、即时治疗和有期限的攻击增量 Buff。弹道飞行时间、持续伤害、毒雾、控制、召唤实体、复杂增益叠层和真实职业进阶关系仍未实现。有限动作 DSL 也不是通用脚本语言。

旧 `ClassSpec`、`SUPPORTED_JOBS` 和 `CLASS_SPECS` 仍只接受原先三个固定职业。没有把 `archer` 或 `bandit` 加入旧常量并复制旧策略；新职业必须显式提供经过校验的外部包。

## 公共入口

```python
from maple_vision.class_package import load_class_package, compile_class_package
from maple_vision.simulation import Simulator

package = load_class_package("examples/classes/bandit.json")
capabilities = compile_class_package(package)
program = capabilities.program_for("combat")
assert program is not None

steps = program.instantiate(direction=1)
assert [step.action.skill for step in steps] == ["slash", "finisher"]
assert [step.not_before_ms for step in steps] == [0, 150]

simulator = Simulator(job=package.class_id, class_package=package)
saved = simulator.save()
assert saved["schema_version"] == 4
assert Simulator.restore(saved).save() == saved
```

编译结果供任务规划器读取，是静态能力知识；不包含真实库存、位置、怪物生命、当前冷却或模拟器实例。调用方将 `CompiledStep` 转换为既有 `ActionStep/ActionPlan`，由统一调度器逐步授权。编译器不自行执行程序、等待时间或发送动作。

| API | 结果 |
|---|---|
| `load_class_package(path)` | 从本地普通文件读取最多 64KiB 的严格 JSON，返回冻结 `ClassPackage` |
| `ClassPackage.from_dict(value)` | 校验完整字段、所有等级、引用、程序及角色语义 |
| `ClassPackage.manifest()` / `from_manifest()` | 完整内容、内容 SHA256、schema 与编译器版本；可独立恢复 |
| `compile_class_package(package)` | 选定等级、解析引用、编译时间偏移和静态预算，返回 `CompiledCapabilities` |
| `caps.skills` / `caps.programs` | 只读映射，条目为冻结对象 |
| `caps.program_for(role)` | 返回绑定程序或 `None`；未知 role 被拒绝 |
| `program.instantiate(direction)` | 将 `target` 方向替换为 ±1，生成有限 `CompiledStep` 元组 |
| `caps.class_spec` | 冻结的 `CapabilityClassSpec` 数值视图；用于既有物理、渲染和静态策略参数 |
| `simulator.class_package` / `.capabilities` | 只读静态包及编译结果；不能把模拟器其他动态字段传给策略 |

`CapabilityClassSpec` 与旧类分别定义。资源上限和移动参数来自包，基础战斗伤害/范围/资源/冷却摘要来自绑定的 combat 程序；其存在不修改旧 `ClassSpec` 的枚举或校验。完整语义以注册技能和编译程序为准，不能用一个汇总伤害字段替代各步骤的技能效果。

## 包格式与完整性

源文件精确包含：

```json
{
  "schema_version": 1,
  "id": "synthetic-bandit",
  "version": "1.0.0",
  "scope": "synthetic_fixture",
  "class_id": "bandit",
  "source_uri": "urn:maple-vision:self-authored:bandit:1",
  "permission": "self_authored",
  "license": "CC0-1.0",
  "resources": {},
  "skills": [],
  "programs": [],
  "bindings": {"combat": "dagger_combo", "area": null, "airborne": null, "heal": null, "buff": null}
}
```

上面仅展示字段结构，空的 resources/skills/programs 不能通过校验；完整有效内容见三个示例文件。未知字段、缺失字段、重复 JSON 键、非有限数、布尔伪装整数、非法版本或未声明语法都会被拒绝。

ID 使用最多 40 个字符的小写字母、数字、下划线或连字符，必须以字母开头。`version` 为有界 `major.minor.patch`。来源/许可采用现有 provenance 校验；`source_uri` 仅是来源声明，加载和编译不会访问该 URI。源文件必须为本地普通文件，不接受符号链接。

每包最多 16 个技能、16 个程序；每技能 1–16 个不同等级，选定等级必须存在。技能和程序声明的存储顺序会规范化，等级按数值排序；动作程序内部的步骤顺序保留语义。

`manifest()` 包含：

```text
schema_version = 1
compiler_version = 1
content = 完整规范化源内容
content_sha256 = 规范 JSON 内容的 SHA256
```

哈希用于内容完整性和运行证据绑定，不是对作者身份或内容正确性的签名。如果有人另写一份结构有效的包并重算哈希，那是另一份包；系统不能只凭内容哈希宣称已经认证其来源。运行证据需要额外保持包哈希、运行配置、意图和实际结果的一致性。

## 技能等级表与资源

`resources` 精确包含 `max_hp`、`max_mp`、`initial_ammo`、`walk_speed`、`jump_speed`。

| 字段 | 范围 |
|---|---|
| `max_hp` | 1–100000 |
| `max_mp` | 0–100000 |
| `initial_ammo` | 0–1000000 |
| `walk_speed` | 1–200 |
| `jump_speed` | 1–400 |

技能定义精确包含 `id`、`kind`、`shape`、`posture`、`cooldown_group`、`selected_level`、`levels`。`kind` 为 `damage/heal/buff`，形状为 `directional/radial/self`，姿态为 `ground/air/any`。damage 需要非自体形状；heal/buff 只能是 self，不能带射程、目标数或弹药成本。

每一个等级都必须有以下字段，不允许通过缺省值隐藏规则：

| 字段 | 含义/范围 |
|---|---|
| `level` | 1–100；同技能内唯一 |
| `power` | 1–100000；伤害、治疗量或攻击增量，由 kind 决定 |
| `range_px` | 0–320；damage 必须大于零 |
| `vertical_range_px` | 0–192；damage 必须大于零 |
| `max_targets` | 0–8；damage 必须大于零 |
| `hp_cost` | 0–99999，且小于职业 HP 上限 |
| `mp_cost` | 0–100000，不能超过职业 MP 上限 |
| `ammo_cost` | 0–200，不能超过初始弹药预算 |
| `cooldown_ms` | 10–60000；作用于声明的共享冷却组 |
| `duration_ms` | 只有 Buff 可以为正，最大 60000；其他技能必须为零 |

所有等级都接受语义校验，未选择的等级也不能夹带非法成本或字段。编译器只把 `selected_level` 对应的数值展开为 `CompiledSkill`，同时保留原包中的完整等级表和内容哈希。程序累计 HP/MP/ammo 成本也不得超过职业可承受的静态容量，HP 累计成本必须保留至少 1HP。

这些是静态可行性约束。实际施放时还必须检查当前资源；一个结构有效的程序不保证角色在每次执行时都有足够资源。

## 有限复合动作 DSL

DSL 仅支持四种精确形式：

```json
{"op": "cast", "skill": "slash", "direction": "target"}
{"op": "move", "direction": 1}
{"op": "jump", "direction": 0}
{"op": "wait", "duration_ms": 150}
```

方向仅能为 `-1/0/1/"target"`，move 不能为零。没有循环、递归、条件表达式、代码字符串、子程序调用、文件路径、网络或动态求值。单程序最多 16 个源指令、8 个可执行动作；wait 为 1–5000ms，总等待不超过 5000ms；不能以无后续动作的 wait 结束。每个能力程序至少调用一个注册技能。

`wait` 不生成执行器动作，而是累加后续动作的 `not_before_ms`。move/jump/cast 是一个模拟器步长内的一次动作，不是持续按住输入。多个动作即使最早偏移相同，也必须由单一调度器按顺序、使用新观测分别执行。

刀飞的编译结果为：

| 步骤 | 动作 | 最早偏移 | 资源 |
|---:|---|---:|---|
| 0 | `cast slash` | 0ms | 2MP |
| 1 | `cast finisher` | 150ms | 3HP + 2MP |

两个技能共享 weapon 冷却组。第一步冷却 150ms，因此第二步等待小于 150ms 的包会在编译时被拒绝；第二步冷却 300ms，因此理想调度下该组从程序起点计算的 ready 偏移为 450ms。

`CompiledProgram` 显式提供 `range_px`、初始 `posture`、累计 `hp_cost/mp_cost/ammo_cost`、`duration_ms`、`reuse_after_ms`、`cooldown_ready_offsets`、`skill_ids` 和 `max_targets`。

**时间元数据是静态最早偏移，不能冒充实际执行时钟。** `duration_ms` 是最后动作的最早偏移，计划期限必须严格大于起点加此偏移，并为实际帧步长/延迟预留预算。`reuse_after_ms` 是按这些理想偏移推算的最晚冷却结束；若后续动作延迟，真实冷却也会后移。策略需按每个 cast 的实际尝试时刻维护保守冷却组时钟，不能在部分执行后把静态计划当作全部施放成功。

角色绑定包括 `combat/area/airborne/heal/buff`。combat 必填，其余可为 null；引用必须存在，角色效果必须相符，area 必须包含多目标伤害，默认 combat 不能要求从空中开始。对地形可达性、跳跃落点和所有后续姿态的保证仍由当前观测与执行器检查，编译器不读取地图真值推演成功。

## 独立模拟器的实际语义

外部包通过 `Simulator(..., class_package=package)` 显式开启，`job` 必须等于包的 class_id，与旧 `class_spec` 注入互斥。只新增 `SimAction(kind="cast", skill="技能ID", direction=...)` 入口；旧 attack/jump_attack 的 basic/area 枚举保持不变。

包路径中技能只能经过 cast。旧 attack、jump_attack、buff、heal 返回 `capability_cast_required`，无法绕过注册表获取未声明能力。未提供包时 cast 返回 `capability_package_required`。移动、跳跃、梯子、药品、拾取、NPC 和 portal 沿用公共环境动作。

施放依次校验技能存在、地面/空中限制、冷却组、当前 HP/MP/ammo、治疗是否已满以及 Buff 是否已经生效。所有条件通过后才扣资源、更新冷却和施加效果；拒绝不会扣除这些资源或改变技能姿态。固定步长的物理与接触伤害仍照常推进，因此“拒绝资源原子性”不等于“整个世界停止变化”。

伤害根据形状、朝向、水平距离、垂直带宽和目标数选择目标。目标按距离和稳定 ID 排序；范围攻击可以覆盖两侧。有效 Buff 的攻击增量参与注册伤害计算，到期后移除。治疗封顶至 max_hp；HP 成本在扣费前要求 `current_hp > hp_cost`，不能用技能成本直接耗尽最后一点 HP。

普通动作的物理/渲染层仍使用既有数值属性。新增职业没有读取真实怪物 HP 来帮助策略选技能；这些状态只存在模拟器内部和独立测试 oracle 中。

## 存档、恢复与历史兼容

只有外部包路径使用新的模拟器存档 schema 4：

- `class_package`：完整包 manifest、来源声明、所有等级、程序、绑定及哈希。
- `capability_state.cooldowns`：恰好包含编译注册表中的每个冷却组及 ready_ms。
- `capability_state.buffs`：当前有效 Buff 技能 ID 及 expires_ms；攻击增量从已验证技能派生，不接受存档随意指定 bonus。
- 若同时使用外部地图，继续嵌入完整 `map_pack`。

恢复时重新校验并编译包，检查 job、内容哈希、编译器版本、冷却组集合、数值类型、时间界限及 Buff 类型/期限。包路径不允许注入旧 `buff_until_ms/attack_ready_ms/skill_ready_ms` 状态来绕过新注册表。删除最初的包文件后，存档仍能独立恢复，继续产生相同动作结果和随机掉落。

`_Player` 未新增字段；`class_package=None` 不增加存档键，不改变原资源默认值和 schema 1/2/3。旧 profile、地图包及默认三职业继续沿原路径执行。测试固定了能力包实现前提交 `9aaba7c55ed999eb7288e4cc22b47d0692bb3548` 的原 simulation.py：五种旧路径各 107 帧，分别比较规范存档轨迹和渲染像素的累计 SHA256。基准由原提交代码独立执行生成，记录于 `tests/fixtures/capability-legacy-baseline.json`，没有用新增实现倒推期望值。

## 验证入口与证明范围

```bash
python -m pytest tests/test_class_package.py tests/test_capability_simulation.py -q
python -m pytest tests/test_simulation.py tests/test_profiles.py tests/test_map_pack.py -q
```

专项测试包含包/manifest/编译结果确定性、选定等级、有限 DSL 与资源/冷却约束、真实方向/范围/弹药效果、不同技能复合执行、HP 保留 1、地空限制、拒绝原子性、Buff 到期、治疗封顶、禁止旧技能绕过、删除源文件后的双图/三图恢复、哈希/身份/能力状态伪造反例、容器隔离及历史字节兼容。

这些测试证明外部数据确实驱动了新技能和模拟器效果；完整运行命令、任务规划接入、证据侧文件、动作重放及重推理属于跨层集成验收，应以相应运行证据为准。示例包和新机制不改变已有 YOLO 模型的质量结论，也不代表真实游戏接入或长期稳定性已经得到验证。

## 统一 TaskRuntime 的能力包路径

```python
from maple_vision.class_package import load_class_package, compile_class_package
from maple_vision.task_runtime import TaskRuntime

package = load_class_package("examples/classes/bandit.json")
runtime = TaskRuntime(
    job=package.class_id,
    capabilities=compile_class_package(package),
    maps=("meadow", "cavern"),
)
```

构造器要求 job 与 `caps.class_id` 一致，包能力与旧 `class_spec` 注入互斥，并复编译原包核对编译缓存，拒绝篡改后的静态能力视图。任一程序最晚动作偏移达到配置的计划时限时，构造直接失败；不能靠运行时悄悄截断程序来满足预算。

外部包使用同一个候选生成器、任务仲裁、ActionPlanner、ActionScheduler 和效果验证器。Planner 根据 `combat/area/airborne/heal/buff` 角色选择注册程序，按声明的 range 接近可见目标，将每个 `CompiledStep` 转换为真实 `ActionStep`。范围角色需要当前看到多个合格目标；空中角色按声明的 jump/cast 顺序执行。没有按 archer、bandit、blood_knight 分别复制策略，也不调用旧 `VisualAgent`。

每个 cast intent 使用固定的 `combat/resource/skill` 锁，逐步经过调度器登记和授权。组合中的后续步骤到达最早执行时刻后重新检查当前观测的资源、姿态、可见目标范围及方向；不满足时取消剩余计划并重新规划。因实际回执延迟而尚未冷却时，可在原计划期限内等待；到期取消。紧急生存任务会抢占并取消剩余组合，未执行步骤不会被记为技能尝试或消耗。

本地能力状态只记录**尝试与保守估计**：

- 冷却组按每个已真实 dispatch、收到绑定回执的 cast 尝试时刻更新。若回执晚于 dispatch，以更晚时刻计算冷却，避免用理想时间线提前重用技能。
- HP/MP 从当前 100px 合成 HUD 的半像素误差区间计算整数下界。新的观测捕获时间覆盖尝试之前，累计未观测消耗不退还；相同捕获时刻的新序号不能重复分配同一份资源。只在更新的实际资源测量到达后重新建立预算，不凭接受回执增加 HP/MP。
- 弹药采用初始包预算减去尝试消耗的保守上限，不宣称识别到真实库存或拾取补充。
- Buff 只记录声明期限内的尝试窗口，避免重复施放；不会将其标成已观察到的 Buff 状态。
- 暂停、来源切换和计划取消会切断组合和旧效果验证，但不退还已发生的尝试冷却或待核实资源支出。

cast 的伤害、Buff 和击杀效果保持 UNKNOWN；当前已知的治疗 cast 可以在同源、时序有效的新画面确实显示 HP 增加时给出窄范围 `observed_hp_increase`，仍不代表整个任务成功。输入被故障注入替换为 idle 时，真实接受回执不会让请求技能被确认为成功，战斗无进展预算会终止运行。

只有能力包路径的 `RuntimeDecision.evidence` 和 `snapshot()` 增加 `capabilities` 字段，包含 class_id、包内容哈希、冷却组时钟、逐技能尝试、Buff 尝试期限和资源估计。`capabilities=None` 不增加这些字段，不改变旧候选、动作、理由、授权、回执或快照。专项测试固定了修改前源文件 SHA256 `73389c477ee4bb41edfa035a0b466d087d0444f20f992765ef54da6d9aec8fae` 下三个原职业的完整两图决策/授权/回执/快照轨迹哈希，检查新 None 路径逐字节保持。

运行时专项入口：

```bash
python -m pytest tests/test_capability_runtime.py -q
```

测试覆盖三个外部职业在两图/三图的完整像素任务和独立真值目标，另以真实像素的密集目标及低 HP 场景验证范围/治疗程序选择；还包含真实多步调度、延迟回执冷却、未更新画面的资源预算、急救抢占、暂停/UNKNOWN/来源切换/目标变化取消、冻结输入的有界失败，以及删除源包后的运行重放和重推理。独立真值仅在模拟器及测试 oracle 使用，不进入候选、Planner 或执行调度。
