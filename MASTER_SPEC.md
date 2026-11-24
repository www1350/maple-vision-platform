# Maple Vision Platform v8.0

完整架构、从零执行步骤与 AI 接手指南见 [docs/FINAL_PLAN.md](docs/FINAL_PLAN.md)；当前 v0.2 实现与验收证据见 [docs/RELEASE_0_2_ACCEPTANCE.md](docs/RELEASE_0_2_ACCEPTANCE.md)，v0.1 历史记录保留在 [docs/RELEASE_ACCEPTANCE.md](docs/RELEASE_ACCEPTANCE.md)。本文件保留总体规格，具体交付范围以版本验收记录为准。

## 项目范围
面向冒险岛风格的离线视觉研究和独立模拟器。正式游戏仅限符合规则的只读观察；不提供正式客户端自动控制、反作弊绕过、封包拦截、内存操纵或输入伪装。独立验证不能证明正式账号安全。

## 总体架构
授权知识与素材 → SQLite 知识快照与 Asset Registry → 地图图层和场景渲染 → Dataset Factory → YOLO Benchmark / ONNX → Observation / WorldState / BeliefState → 模拟器任务引擎 → Verification / Supervisor → Validation Lab → AI Engineering Harness。

## 数据底座
小册子 https://mxdc.dvg.cn/ 等获准来源，提前同步职业、转职、技能、怪物、掉落、地图、道具、NPC、任务。SQLite 只读运行，原始记录、来源、版本、哈希、授权、验证状态均可追溯。VERIFIED、PROVISIONAL、UNKNOWN、CONFLICTED 分开管理；冲突不覆盖。异步知识补充不得阻塞运行。

## 素材与地图
授权动画、图标、背景、地图结构独立存储，按实体 ID 关联。MapleNecrocer 为待评估候选工具，须验证许可及兼容性。分离地图坐标、镜头坐标、远景视差、装饰、地形、动态对象和特效。地图包包含 manifest、geometry、layers、portals、validation。逻辑结构不等同背景图。

## 数据集与视觉
合成数据优先，自动标签，独立真实录像评估。按会话和场景划分，防止相邻帧泄漏。对比通用检测、细粒度分类、两阶段检测分类及 nano/small/medium 模型。评估像素画适用增强、背景捷径、跨地图泛化、技能遮挡、小目标、延迟。禁用不符合目标域的大幅透视和旋转作为初始基线。训练与推理分离，验证后导出 ONNX。OpenCV 负责固定 UI，YOLO 负责动态对象，OCR 负责文字。状态具有时间戳、来源、置信度和有效期。

## 独立模拟器
固定时间步与种子，支持保存、恢复、单步、逻辑和视觉渲染。独立物理、战斗、技能、Buff、资源、背包、掉落、地图、NPC 对话。职业包覆盖标飞、龙骑士、牧师等差异化机制，实际规则须验证。任务生命周期、冲突管理、动作结果验证、有限重试和安全停止仅在模拟器内执行。真实玩家聊天仅提供人工确认的草稿。Ground Truth 仅供独立验证，禁止视觉 Agent 读取。

## 验证
单元、集成、Golden Replay、属性测试、差异测试、故障注入、24 小时长稳。失败包保存录像、观测、世界状态、任务和动作轨迹、Ground Truth、种子、配置、版本锁。未知不等于成功。建议模拟器目标：地图识别 99%、平台定位 98%、核心任务 98%、拾取 95%、生存 99%、恢复 95%；须报告样本量和不确定性，非实测成绩。

## 技术栈
Python 3.12、SQLite、Pydantic、OpenCV、Pillow、Ultralytics YOLO、ONNX Runtime、本地 OCR、Pygame CE、pytest、Hypothesis、Ruff、mypy、JSONL、MP4。单机模块化，不需要 Redis、Kafka 或运行时 LLM。

## 交付阶段
P0 工程合同与 CI；P1 知识同步；P2 素材；P3 地图渲染；P4 数据集；P5 模拟器；P6 模型评估；P7 视觉状态；P8 职业能力；P9 模拟器任务与验证；P10 Supervisor；P11 回归；P12 扩展；P13 发布。每阶段独立测试、审查和验收。P0 已验收；用户于 2026-11-24 授权连续实施、测试并交付后续离线平台。真实素材、真实录像、已核实规则和长稳验收须分别记录，不能用合成测试冒充。
