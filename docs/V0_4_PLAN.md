# v0.4 外部地图包执行方案

本轮把地图内容从两张内置样例中解耦：新增一个有界、不可变的 JSON 地图包，让同一份内容进入模拟器、场景渲染、生命周期数据、视觉任务和自包含验证。基线是已经合并的 v0.3，提交 `493b7d24688c6aad22d7583f11e8098a28965567`，历史 [PR #4](https://github.com/www1350/maple-vision-platform/pull/4) 的最终归档 CI 已通过。

本轮属于工程扩展，下面的验收条件在早期接口探测后整理，**不是模型研究预注册**。不训练新候选、不评分 v0.3 留出 test，也不重开已经拒绝的候选推广分支。旧模型在新地图上的准确率仍未验收。

## 1. 交付合同

| 子系统 | 本轮实现 | 独立检查 |
|---|---|---|
| 内容加载 | 严格 schema1，完整来源声明和 SHA-256；2–8 张地图 | 非法类型、重复键、错连接、越界、超限、哈希错误均拒绝 |
| 模拟器 | 显式地图包、出生位置、怪物位置、平台、门、NPC、梯子 | 实际碰撞/攀爬/交互，存档恢复和循环路线 |
| 渲染/感知 | 地图主题及唯一 HUD 标记；视觉只拿 RGB 与静态标记表 | 固定 320×192，标签保持真实可见范围，模型地形无像素补回 |
| 任务 | VisualAgent 只接收静态路线 ID，按顺序清理、拾取、对话、切图 | oracle 从声明地图及怪物数量推导目标，不写死两图六怪 |
| 数据 | lifecycle schema3，完整嵌入地图，逐会话动作/像素/标签复核 | 每图六阶段、分区互斥、内容篡改和源文件删除 |
| 运行证据 | schema3 存档、地图 sidecar、摘要/感知身份绑定 | 动作 replay 与 N+1 帧实际 reinfer；精确类型及一致性检查 |
| 发布 | 版本 0.4.0，旧接口兼容、干净安装、三组远端 CI | 准确提交检查通过后合并；最终归档 head 也必须检查 |

完整 schema、错误边界及具体 API 见 [MAP_PACKS.md](MAP_PACKS.md)。本轮地图始终是自建的 `synthetic_fixture`；JSON 不执行代码、不下载 URI、不加载新职业实现。已有 SpritePack 和职业参数可同时注入，但不会因为地图加载成功而自动取得适配后的视觉准确率。

## 2. 内容边界

v1 外部地图包固定世界及视口为 320×192，地面 y=168；每图 2–8 个平台、1 个门、1 个 NPC、1–4 条梯子、1–6 个地面怪物。路线必须包含全部地图且每图恰好一次，唯一出口指向下一图，末图回到入口。运行任务到末图对话即结束；数据生成额外走完回到入口的循环，用于切图阶段覆盖。

示例 `examples/maps/three-map-fixture.json` 包含 `grove → quarry → ruins → grove`，每图 3 个怪物，共 9 个；样例内容声明 CC0-1.0。其规范化内容 SHA-256 为 `e4bed8c565452dc79d29f51d2a78564546be3070168a882fa2b8215b1152e2b9`。哈希核对内容，不验证声明者身份或法律权利。

结构校验不等于任务可解：视野遮挡、位置组合、有限药品/弹药和策略预算仍可能造成失败。数据轨迹最多 4096 步，失败不发布半成品；run 使用显式 max_steps，suite/soak 的外部地图每轮预算为 `max(1000, 500×地图数)`。当前策略不提供任意地图导航、梯子驱动规划、分支任务图、技能动画包或新职业代码插件。

## 3. 兼容和身份

未传 `map_pack` 时，内置两图、schema1/2 存档、legacy v1 / lifecycle v2 数据继续走原有路径。三职业旧状态与 RGB 黄金摘要独立从基线提交生成；禁止更新预期摘要来掩盖兼容性改变。

外部存档升级为 schema3，地图与可选职业参数同时内嵌。运行摘要/外层证据 manifest 保持 schema1，新增 map_pack 元数据、`map-pack.json` 与感知 settings 中的地图摘要。验证器核对三份地图、声明种子的初始世界、采样画面、全部动作事件和终止结果；重新计算外层哈希不能掩盖它们之间的矛盾。全部证据一起重写不具有原作者身份保证，哈希不是签名。

数据集采用 schema3，其地图在 manifest 中完整嵌入；分组评估沿实际地图 ID 输出。每会话至少 `6×地图数` 帧，本轮使用 `9×地图数=27` 帧并生成 18 会话。训练私有副本原有完整 manifest 复制机制继续保留地图身份，训练入口无需修改。

## 4. 实施顺序与失败处理

1. 冻结不可变内容 schema 与跨模块 API，先实现严格加载器和三图样例。
2. 接入实例级模拟地图、独立渲染与静态标记解析，保留旧默认路径。
3. 接通生命周期数据及分组评估，证明源文件删除后数据仍可复验。
4. 接通任务路线、动态目标数、运行证据与实际重新感知；加入来源不一致和重算哈希反例。
5. 做跨模块审查，保留实际发现：梯子两端会越界、垂直不相交的传送/NPC、布尔与整数相等、哈希前无界读、初始世界与声明种子不符。
6. 修复后运行全树格式、静态检查、完整测试、原生 smoke 和隔离 wheel；工程一 epoch 测试不属于研究候选训练。
7. 运行三职业任务、13 场景、源路径删除后的 replay/reinfer、18×27 数据及 pixel val 分组、60 秒短时运行。保留真实失败，不重跑替换失败证据。
8. 代码、方案、示例和报告进入同一 PR；核对准确 head 的 Ubuntu、Windows、native CI，再合并。最终归档回执写在 PR，避免新增记录自身哈希的循环提交。

## 5. 从零复现命令

先按 [快速开始](QUICKSTART.md) 安装环境，在仓库根目录执行。输出必须使用全新路径：

```bash
python -m maple_vision maps check examples/maps/three-map-fixture.json
python -m maple_vision lab run --map-pack examples/maps/three-map-fixture.json --job assassin --output artifacts/v04-run
python -m maple_vision lab replay artifacts/v04-run
python -m maple_vision lab reinfer artifacts/v04-run
python -m maple_vision dataset create artifacts/v04-data --profile lifecycle --map-pack examples/maps/three-map-fixture.json --sessions 18 --frames 27 --seed 20261125
python -m maple_vision dataset check artifacts/v04-data
python -m maple_vision vision benchmark artifacts/v04-data --split val --output artifacts/v04-pixel-val.json
```

完整本轮独立验收可以执行：

```bash
python scripts/verify_external_maps.py --output artifacts/v04-acceptance
```

该脚本使用像素基线，默认包含实际 60 秒短时运行。报告包含三个职业的完整运行摘要、动作重放及实际重新感知结果、13 场景、数据与分组指标、资源采样和源码身份。加载器实际读取的临时路径在开始测试前删除；原始用户文件不删除。原始运行包、失败包、数据、逐步轨迹留在指定目录。

工程命令为：

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -q
python -m build --wheel --no-isolation
python scripts/wheel_smoke.py --wheel dist/maple_vision_platform-0.4.0-py3-none-any.whl
```

设置 `MAPLE_TRAINING_SMOKE=1` 后的完整测试或 CI 原生作业，另验证真实一 epoch 训练、导出、ORT、录像和 OCR。准确结果以 [RELEASE_0_4_ACCEPTANCE.md](RELEASE_0_4_ACCEPTANCE.md) 为准，执行命令存在不表示它已通过。

## 6. 未扩展的结论

本轮没有学习模型的新地图测试、未见地图留出研究或真实域准确率结果。同一地图包在各数据 split 出现，会话隔离不能作为未见地图泛化证明。60 秒实际运行不能计作 24 小时。真实录像标注、网络知识来源、动画、更多职业、时序状态、任务 DSL 和完整 v8 继续列为后续工作。v0.3 已拒候选的结论与历史报告保持不变。
