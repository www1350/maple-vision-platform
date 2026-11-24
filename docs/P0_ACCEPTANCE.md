# P0 验收报告

- 日期：2026-11-24（北京时间）
- 方案：v8.0；工程阶段：P0
- 起始提交：`961abeaa299ca36be02308964b9d8cf068fd857c`
- 交付位置：[PR #1](https://github.com/www1350/maple-vision-platform/pull/1)
- 代码验收提交：`06deaa52dc9f938efb4f7741d21601fdab7e301a`
- 当前结论：**P0 验收通过**，本地和 Windows/Linux CI 均已完成。

## 已交付

1. 完整 `Frame → Observation → WorldState → Intent → NoOpController → Verification` 管线，保留 `run_once` 旧 API。
2. 冻结数据模型和类型协议，检测无效 ID、负时间、非法置信度、过期与不匹配数据。
3. TOML 配置、`maple-vision`/`python -m maple_vision` 命令入口、版本化 JSONL 和 JSON 摘要。
4. 事务原子的 SQLite v1 迁移、原数据保留、未来/冲突 Schema 拒绝、只读检查。
5. 知识源、素材导入、Detector 和 Simulator 抽象协议。
6. 固定开发工具版本、Linux/Windows CI、独立 wheel 安装验收、README/API/ADR/P0 任务合同。

## 实际本地验证

环境：Linux x86_64，CPython 3.12.14；具体开发工具版本见 `requirements-dev.txt`。

| 命令/检查 | 实测结果 |
|---|---|
| `python -m ruff format --check .` | PASS |
| `python -m ruff check .` | PASS |
| `python -m mypy` | PASS，10 个源码文件 |
| `python -m pytest -q --junitxml=artifacts/pytest.xml` | PASS，70 项测试 |
| `python -m build --wheel --no-isolation` | PASS，生成 0.0.1 wheel |
| 新虚拟环境安装 wheel，并在仓库外运行 | PASS，导入路径来自该环境 site-packages |
| 三帧 CLI demo / JSONL | PASS，三个 NOOP 回执和三个 NOT_APPLICABLE 验证 |
| CLI db init / check | PASS，v1 Schema、0 条示例数据 |

没有把模拟器终版的识别率、恢复率、24 小时目标套用到 P0，也未运行或声称完成这些测试。

## 独立审查与修复

独立审查发现：仅检查帧时间还不足以阻止 `now_ms` 回退，使旧观测可能重新变新。已加入处理时钟单调检查，并用显式回退、切回隐式时钟两个回归场景验证；错误在调用控制器前拒绝。

另外独立确认了：回执恰好达到截止时间是 TIMEOUT；日志不可覆盖；SQLite URI 只读限制不会因关闭 query_only 而失效；外层事务回滚可以撤销迁移 DDL。

## 远端验收

CI 对本 PR 提交运行 Ubuntu 和 Windows、Python 3.12，分别完成格式、lint、类型、pytest、wheel 构建与仓库外安装检查。结果以 [Actions](https://github.com/www1350/maple-vision-platform/actions) 的具体提交 SHA 为准，不能沿用最初四项测试的旧成功记录。

本次代码验收：[Foundation CI #4](https://github.com/www1350/maple-vision-platform/actions/runs/45367311402)，状态为 success。

| 环境 | pytest 实测 | 格式 / lint / 类型 | wheel 构建与独立安装 |
|---|---|---|---|
| Ubuntu / Python 3.12 | 70 passed，0.90s | 全部 PASS | PASS |
| Windows / Python 3.12 | 70 passed，2.02s | 全部 PASS | PASS |

两个平台均成功上传 `p0-evidence` 产物，含 JUnit XML 与 wheel smoke 摘要；产物按 workflow 设置保留 14 天。测试耗时仅描述此次 CI 运行，不作为性能保证。

此处追加的是测试证据与合同状态，未改动已验收的源代码或测试。后续文档提交和合并仍由 CI 检查，不继承过期的检查结论。

## 限制与后续

- P0 的帧是确定性测试数据；没有真实视觉识别、录像解码或游戏模拟器物理。
- NoOp 回执为 SKIPPED，验证为 NOT_APPLICABLE，不能计作游戏任务成功。
- 没有小册子爬虫、静态素材、模型权重或真实知识数据进入仓库。
- P0 保持既有 v1 `id TEXT PRIMARY KEY` 兼容；P1 写入数据前需显式迁移补齐 NOT NULL 及业务完整性约束。
- 只读连接和 NoOp 约束不是系统级沙箱，不宣称能阻止任意替换 Python 代码的行为。
- P0 完成后按 `AGENTS.md` 停止；下一阶段为 P1 本地知识快照和来源规范化。
