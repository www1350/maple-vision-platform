# Maple Vision Platform v8.0

完整架构、从零执行步骤与 AI 接手指南见 [docs/FINAL_PLAN.md](docs/FINAL_PLAN.md)。当前版本为 **v0.3 地形研究与模型对象管线，工具工程验收进行中**。唯一候选已经完成训练并在val失败，不推广、不评分test；执行合同见 [docs/V0_3_PLAN.md](docs/V0_3_PLAN.md)，实际结果与待完成的工程门禁见 [docs/RELEASE_0_3_ACCEPTANCE.md](docs/RELEASE_0_3_ACCEPTANCE.md)。v0.2 已验收证据保留在 [docs/RELEASE_0_2_ACCEPTANCE.md](docs/RELEASE_0_2_ACCEPTANCE.md)，v0.1 历史记录保留在 [docs/RELEASE_ACCEPTANCE.md](docs/RELEASE_ACCEPTANCE.md)。本文件保留总体规格，具体交付范围以版本验收记录为准；版本号不代表完整 v8 已完成。

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

当前 v0.3 有界研究只测试一个 `reg_max=16`、stride4/8/16 的候选，并新增实际结构/锚点审查、可选 schema3 地形几何评分及显式 `terrain_source=model` 模式。该模式的七类对象全部来自模型，固定 HUD/地图/对话仍由 fixture 像素解析器读取；默认 v0.2 混合模式继续保留。平台总体召回、全宽地板召回和原图顶边/端点误差分别验收，不能用 IoU 达标代替支撑面几何。

本轮18会话×48帧使用新种子，并与旧数据核对 RGB 重复及会话种子；train/val/test 为10/4/4会话。两张地图仍相同，test 的192帧来自4个相关会话，不是未见地图测试。候选训练前已经冻结[预注册](reports/release-0.3.0/preregistration.json)：单候选40epoch，val失败不评分test、不推广。本次实际40epoch已完成：14项val阈值中13项通过，但平台召回472/576=0.819444未达0.98；即使全宽地板192/192命中，也不能替代总体平台门禁。失败候选保留在 `examples/terrain-candidate`，原 `examples/lifecycle-model` 的已验收混合管线不变。test及该候选的推广suite/reinfer/soak按协议不执行，不追加第二候选。

结构审查增加原生模型构建前的分配预算，复用训练离线运行域。正式40epoch训练与val评分期间，五项冻结源码保持预注册字节，历史实现可从 `ff357c8951da8e84265d31c5a8b86a48934a4f01`读取，1730个输入副本文件一致。之后远端原生CI发现日志handler引用已关闭流，`training.py`另作窄范围工程修复；修复不追溯进入正式实验，模型与原报告保持不变。

修复前完整本地823项与旧wheel保留为历史；首轮远端基础Ubuntu/Windows及wheel通过，但native为152通过、3失败，整轮未通过。修复后完整本地 **834通过、0失败/跳过**，同序155项、日志回归13项及重建wheel均通过；各组覆盖重叠不相加，修复后准确提交的远端CI仍待结果。详情见[发布记录](docs/RELEASE_0_3_ACCEPTANCE.md)，不能用工程修复改变候选拒绝结论。

## 独立模拟器
固定时间步与种子，支持保存、恢复、单步、逻辑和视觉渲染。独立物理、战斗、技能、Buff、资源、背包、掉落、地图、NPC 对话。职业包覆盖标飞、龙骑士、牧师等差异化机制，实际规则须验证。任务生命周期、冲突管理、动作结果验证、有限重试和安全停止仅在模拟器内执行。真实玩家聊天仅提供人工确认的草稿。Ground Truth 仅供独立验证，禁止视觉 Agent 读取。

## 验证
单元、集成、Golden Replay、属性测试、差异测试、故障注入、24 小时长稳。失败包保存录像、观测、世界状态、任务和动作轨迹、Ground Truth、种子、配置、版本锁。未知不等于成功。建议模拟器目标：地图识别 99%、平台定位 98%、核心任务 98%、拾取 95%、生存 99%、恢复 95%；须报告样本量和不确定性，非实测成绩。

## 技术栈
Python 3.12、SQLite、Pydantic、OpenCV、Pillow、Ultralytics YOLO、ONNX Runtime、本地 OCR、Pygame CE、pytest、Hypothesis、Ruff、mypy、JSONL、MP4。单机模块化，不需要 Redis、Kafka 或运行时 LLM。

## 交付阶段
P0 工程合同与 CI；P1 知识同步；P2 素材；P3 地图渲染；P4 数据集；P5 模拟器；P6 模型评估；P7 视觉状态；P8 职业能力；P9 模拟器任务与验证；P10 Supervisor；P11 回归；P12 扩展；P13 发布。每阶段独立测试、审查和验收。P0 已验收；用户于 2026-11-24 授权连续实施、测试并交付后续离线平台。真实素材、真实录像、已核实规则和长稳验收须分别记录，不能用合成测试冒充。

当前已经归档v0.3唯一候选失败与关闭的推广分支，并完成日志修复后完整回归和重建安装验证；下一步是修复后准确提交的远端CI，再进入后续扩展。模型推广已明确未通过，工程发布仍需远端证据。网络小册子适配、真实数据、外部地图/职业包、时序跟踪、任务DSL、梯子驱动导航和24小时长稳保持未完成。
