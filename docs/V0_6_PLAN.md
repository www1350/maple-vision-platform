# v0.6：领域数据、动画与可编译职业内容

本轮继续原始 v8 第 3–6、10–13 和 15 章，基线为 v0.5 合并提交 `24f73dbf832c81b1bcee71093e60e7cd030b533c`。原始 22 章正文和归档源码保持字节不变。v0.5 已由 [PR #7](https://github.com/www1350/maple-vision-platform/pull/7) 完成准确提交检查并合并，本轮不重复其归档流程。

## 交付目标与依赖

| 交付 | 实际链路 | 独立验证 |
|---|---|---|
| 领域数据库 | 已验证的本地 catalog snapshot → 25 张类型化领域表 → 只读查询 | 从随包原始快照重新投影 schema 和每行数据；外层哈希重算不能掩盖矛盾 |
| 资源索引 | 现有 Asset Registry → 独立 SQLite 索引 → entity/version 查询 | 重新核对来源登记与源字节；拒绝错表、隐藏 trigger、WAL 和路径问题 |
| 动画内容 | 注册帧 → clip/时长/锚点 → 绑定渲染类别 → 场景、数据集、运行证据 | 独立像素和时间边界、alpha 遮挡、HUD 裁切、删去原素材后逐帧恢复 |
| 职业编译 | ClassPackage → 选定等级技能表 → 有限动作程序 → 统一任务调度 → Simulator | 三种机制不同的职业；资源、冷却组、姿态、取消、抢占、真实回执与旧行为基准 |
| 分层报告 | 原始运行包 → 真值/观测/belief/任务/动作/验证/监督 + 全帧 RGB | 独立重放实际 applied_action，重新投影各层，保留真实终止原因；重推理须实际执行 |

数据与运行内容都声明为 `synthetic_fixture`。导入的数据来源声明不会自动变成真实游戏资料认证；没有原始依据的字段保留空值或明确问题记录。

## 外部职业的执行方式

`ClassPackage` 包含资源上限、移动参数、技能各等级、选定等级、程序及角色绑定。程序只能使用 cast、move、jump 和有界 wait，没有脚本求值或网络。编译器核查引用、姿态、资源和共享冷却，生成静态能力视图。

`TaskRuntime` 仍使用原有候选仲裁、任务生命周期、Planner、Scheduler、Intent 和效果验证。职业包改变可用能力和程序，不复制每职业 Runtime。实际调度每步检查新的观测，冷却由实际施放时刻计算；等待或暂停不能提前刷新冷却，资源按 HUD 可见误差下界和未覆盖的施放扣减预算处理。隐藏 HP、怪物、背包和冷却状态不进入策略。

三份样例分别覆盖弓手弹药与空中射击、刀飞共享冷却组合技、血骑 HP 消耗范围攻击与治疗。它们是机制样例，原始标飞、龙骑、牧师保持兼容。

```bash
python -m maple_vision classes check examples/classes/bandit.json
python -m maple_vision lab run --class-pack examples/classes/bandit.json --output artifacts/bandit
python -m maple_vision lab reinfer artifacts/bandit
python -m maple_vision lab suite --class-pack examples/classes/archer.json --class-pack examples/classes/bandit.json --class-pack examples/classes/blood_knight.json --output artifacts/classes-suite
```

`--class-pack` 自动选择包中的职业和 structured 模式；显式指定不一致职业、legacy 模式或旧知识库职业参数会拒绝。API 使用 `CapabilityRunConfig`，旧 `RunConfig` 仍只允许原三职业。完整包进入模拟器 schema4 存档和 schema2 运行证据，独立恢复不依赖原文件。

## 动画与资料入口

```bash
python -m maple_vision knowledge init artifacts/domains-source.sqlite
python -m maple_vision knowledge import artifacts/domains-source.sqlite fixtures/knowledge-domain-demo.json
python -m maple_vision knowledge snapshot artifacts/domains-source.sqlite artifacts/domains-source
python -m maple_vision knowledge project-domains artifacts/domains-source artifacts/domains
python -m maple_vision knowledge query-domain artifacts/domains skill_levels fixture-v1 --scope synthetic_fixture --entity-id throw
python -m maple_vision animations check examples/animations/fixture
```

动画绑定文件是 `{"player":"实际 clip_id"}` 形式的 JSON。`animations scene BUNDLE BINDINGS OUTPUT` 输出可移植场景，`--animation-scene` 可用于 lifecycle 数据和 run/suite/soak。逐帧按绝对模拟时间选择 clip，原生像素按锚点附着实体底边中心。这一轮不声称已根据动作识别自动选择 idle/attack/hurt 动画。

注册素材不保证被既有颜色基线或模型识别。验收中的闭环动画明确使用自绘且保留颜色边界的几何帧，任意游戏画风需要各自的检测器和证据。schema4 数据包含完整动画源；训练私有副本复制同一内容，旧数据生成保持兼容。

## 分层失败与回放报告

```bash
python -m maple_vision lab export-bundle artifacts/bandit artifacts/bandit-bundle
python -m maple_vision lab verify-bundle artifacts/bandit-bundle --reinfer
```

成功或失败运行都可以导出。默认检查动作及证据层，`--reinfer` 才重新执行感知和任务；模型运行需要原模型身份和参数。可选 `--video` 生成展示用 MP4，其有损像素不是权威真值。报告保留完整 N+1 帧、实际终止状态和未验证项，不把预算终止补写成多一次策略决定。

## 验收与继续执行

独立审查先构造反例，再做完整回归、正式任务矩阵、短时墙钟测试和干净 wheel 安装。统一复现入口：

```bash
python scripts/verify_content_platform.py --output artifacts/v06-acceptance
MAPLE_TRAINING_SMOKE=1 python -m pytest -q
python -m build --wheel --no-isolation
python scripts/wheel_smoke.py --wheel dist/maple_vision_platform-0.6.0-py3-none-any.whl
```

实际成绩在发布验收记录中填写，不能由此执行方案推断已通过。最终准确 head 的 Ubuntu、Windows、原生测试和安装检查通过后才合并。随后继续两层地图图结构、可验证导航、更多技能效果、任务恢复及管理工作台；真实数据、独立真实录像、模型族比较和 86400 秒运行均保留各自门禁。
