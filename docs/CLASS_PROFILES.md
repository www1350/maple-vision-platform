# 从本地知识目录加载职业规则

这个接口连接 SQLite 知识目录与独立模拟器，不需要网络。它只接收自建样例范围 `synthetic_fixture` 中、已明确核实的三个现有职业参数。它不会把 `game_reference` 记录解释为已经验证的模拟器规则。

## 完整数据合同

记录使用知识目录标准格式，`kind="classes"`，`entity_id` 为 `assassin`、`dragon_knight` 或 `cleric`。payload 必须提供下面全部字段，不能靠默认值掩盖缺失字段，也不接受附加字段。

| payload 字段 | 类型与范围 | 含义 |
|---|---|---|
| `id` | 三个现有职业之一，必须与 entity_id/运行 job 一致 | 职业 ID |
| `max_hp` | 整数 1–100000 | 最大生命及初始生命 |
| `max_mp` | 整数 0–100000 | 最大 MP 及初始 MP |
| `attack_damage` | 整数 1–100000 | 基础攻击伤害 |
| `attack_range` | 整数 1–320 | 普通攻击距离，像素 |
| `attack_mp_cost` | 整数 0–max_mp | 普通攻击 MP 消耗 |
| `attack_cooldown_ms` | 整数 10–60000 | 攻击冷却，毫秒 |
| `ammo_cost` | 整数 0–200 | 每次攻击的弹药消耗 |
| `can_jump_attack` | 严格布尔值 | 是否支持起跳攻击组合 |
| `heal_amount` | 整数 0–max_hp | 自我治疗量；0 表示没有该能力 |
| `walk_speed` | 整数 1–200 | 水平移动速度，像素/秒 |
| `jump_speed` | 整数 1–400 | 起跳向上速度，像素/秒 |

布尔值不会被当成整数，浮点数、NaN、Infinity 和越界值全部拒绝。地形、重力、怪物、通用 Buff、药品恢复量、治疗 MP 消耗和龙骑范围技能半径等仍是 `synthetic-fixture-v1` 的共同规则；本接口尚未将它们参数化。

`fixtures/knowledge-demo.json` 的三个职业使用上述完整合同，其数值与内置默认职业一致。其他种类的示例条目仍是目录演示数据，不会自动成为运行规则。

## 启动与接口

```bash
python -m maple_vision knowledge init artifacts/knowledge.sqlite
python -m maple_vision knowledge import artifacts/knowledge.sqlite fixtures/knowledge-demo.json
python -m maple_vision lab run --output artifacts/profile-run --job assassin --knowledge-db artifacts/knowledge.sqlite --game-version fixture-v1
python -m maple_vision lab replay artifacts/profile-run
```

所有输出使用新路径；父目录 `artifacts` 须已存在。版本必须显式选择，不自动猜测当前国服版本。

Python 接口：

```python
from dataclasses import asdict
from maple_vision.profiles import load_class_profile, parse_loaded_class_profile
from maple_vision.simulation import Simulator

profile = load_class_profile("knowledge.sqlite", "assassin", "fixture-v1")
sim = Simulator(seed=7, job="assassin", class_spec=profile.class_spec)

assert sim.class_spec == profile.class_spec
metadata = asdict(profile)
assert parse_loaded_class_profile(metadata) == profile
restored = Simulator.restore(sim.save())
assert restored.class_spec == sim.class_spec
```

加载器读取同一职业/版本的全部候选，要求全部 `VERIFIED`、没有冲突且 scope 为 `synthetic_fixture`。同一 payload 的多个已验证来源全部保留；UNKNOWN、PROVISIONAL、CONFLICTED 或其他 scope 的候选会导致启动失败，不能按最新时间、来源顺序或任意一条自动消解。

返回值 `LoadedClassProfile` 是不可变对象，包含：

- `class_spec`：已验证的静态职业参数。
- `profile_sha256`：完整职业 payload 的规范 JSON 哈希。
- `records`：全部来源记录，包含原 URI、版本、日期、许可与验证状态。
- `provenance_sha256`：这些来源记录的规范 JSON 哈希。

`LoadedClassProfile` 的构造也检查这些字段之间的对应关系。`parse_loaded_class_profile()` 能独立复核导出的 metadata，不需要再访问数据库。哈希用于检查内容一致性，不等于第三方来源的数字签名。

## 保存、恢复与兼容性

没有传入 `class_spec` 的运行仍输出原 schema v1；已有默认状态与动作轨迹的规范字节哈希保持不变。

显式注入参数的运行输出 schema v2，增加：

```json
{
  "class_profile": {
    "spec": {"id": "assassin", "...": "完整 ClassSpec 字段"},
    "sha256": "规范 JSON 的 SHA256"
  }
}
```

这里只展示结构，不能把省略字段的例子当成可导入配置。恢复时重新检查字段集合、数值边界、job 对应及哈希，然后使用内嵌参数重建物理、资源与战斗。v2 不能缺少职业参数；v1 不能偷偷附加职业参数；参数和哈希不一致时拒绝。

数据库只在启动时读取。保存后即使移动或删除数据库，也能重放同一动作序列。完整来源 metadata 另外进入运行报告；独立验证同时检查该 metadata 与模拟器嵌入参数一致。`reset()` 保持注入规则，新的 `Simulator()` 仍使用原内置默认值，不修改全局 `CLASS_SPECS`。

视觉策略接收的是相同的不可变 `ClassSpec` 静态知识；这不包含玩家位置、怪物血量、冷却剩余等动态真值。动态状态仍来自画面观测。

## 已验证的行为

`tests/test_profiles.py` 实际验证：扩大标飞攻击范围并提高伤害后能命中并消灭默认范围外目标；MP、弹药、移动速度、跳跃速度、治疗和资源上限使用注入参数；原全局默认值保持不变；删除数据库后保存恢复与后续动作逐步相同；配置篡改、来源篡改、缺失字段、未知/冲突资料与错误 scope 被拒绝。另有默认 v1 状态和轨迹的历史哈希回归。
