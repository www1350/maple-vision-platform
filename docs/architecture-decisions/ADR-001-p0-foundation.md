# ADR-001：P0 使用标准库构建可验收的最小闭环

- 状态：实施
- 日期：2026-11-24
- 适用范围：P0

## 背景

原始 PR 仅实现 Frame→Observation，尚未覆盖方案要求的 WorldState、Intent、Controller 和 Verification。此时引入完整视觉或游戏依赖，会让工程基础验证依赖外部数据、模型和设备。

## 决定

1. P0 使用 dataclass、Protocol、argparse、tomllib、json、sqlite3；运行时没有第三方依赖。Pydantic、OpenCV、ONNX 等按后续阶段需要引入。
2. 保留原 API，新增完整 NoOp 管线。NoOp 的结果为 SKIPPED/NOT_APPLICABLE；不给空实现颁发游戏动作 SUCCESS。
3. 帧时间、观测时间和处理时间使用同一时钟域；检查不一致、倒退和过期。
4. SQLite 迁移保留有效 v1 Schema，通过 SAVEPOINT 尊重外层事务所有权。P1 负责新业务 Schema 与数据约束。
5. 配置使用 TOML，避免 P0 运行时只为配置增加 YAML 依赖；职业/地图内容包依然可在后续阶段使用 YAML。开发验收合同继续保留 YAML。
6. CLI、类型检查、负面用例、Windows/Linux CI 和仓库外 wheel 安装共同提供交付证据。

## 影响

这一阶段可以完整验证工程连通、失败语义、数据库安全和打包。它不能证明视觉模型、玩法机制、控制性能或正式账号安全。后续模块必须逐项获得独立验收。
