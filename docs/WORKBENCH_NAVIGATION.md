# 本机工作台与两层地图导航

## 运行

需要 Python 3.12。以下命令在仓库根目录执行。浏览器打开 http://127.0.0.1:8765 。

```bash
python -m pip install -e ".[workbench]"
python -m maple_vision workbench --workspace artifacts/workbench --examples examples
```

| 页面 | 操作和证据 |
|---|---|
| 任务与回放 | 选择职业、地图、感知和故障，启动一次 structured 任务。运行记录保留原始目标结果、终止原因、采样帧和可分页的动作/观测。持续丢帧会产生可以检查的停止记录 |
| 回放复验 | 在「验证方式」中选择动作重放、感知与任务重推理、导出分层包或验证分层包，再选择来源。分层包提供全部 N+1 帧；动作重放和重新感知分别执行 |
| 单步模拟器 | 手动推进、保存、重置、恢复。每一步 50 ms 模拟时间。视觉观测由渲染帧产生，验证真值来自模拟器。修订号拒绝基于旧状态提交的操作 |
| 内容与模型 | 展示本次启动载入的职业包版本与技能、地图路线、感知方式。省略 `--examples` 可直接使用三个内置职业与内置两图。ONNX 混合模型还需要 `pip install -e ".[workbench,inference]"` 和对应模型文件 |

工作台使用单机 SQLite 和每个任务的独立证据目录。每次最多四个排队或运行中的任务，一个工作线程实际执行。排队任务可取消，已运行任务保留其终止结果。重新启动同一目录可以查看已有结果，进程中断时未完成的任务记录为中断。已打开的单步会话在刷新页面后恢复。证据检查以现有 lab 规则为准；原始数据与图形界面分开保存。

## 地图与平台路径

```bash
python -m maple_vision maps route examples/navigation/branch-world.json camp lookout
python -m maple_vision maps platform-route examples/navigation/platform-geometry.json examples/navigation/movement.json --start 24 146 --target platform_2
python examples/navigation/verify_fixture.py
```

第一条命令验证带方向和通行成本的世界图。样例选择 camp → fork → bridge → lookout，总代价 4，孤立节点不会被推断为可达。第二条基于移动能力、初始位置和静态平台做有界搜索。返回路径时可复演每一步；预算耗尽会报告未知。第三条将样例平台路径的每一步送入原有模拟器，独立核对实际位置、速度和支撑状态。

当前地图包的自动任务仍使用其声明的循环路线。世界图分支规划和平台动作验证可以单独调用。动态敌人避让、任意正式游戏地图的可达性和国服物理参数需要额外证据。

## 验证入口

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_workbench.py tests/test_navigation_graph.py -q
python -m pip install playwright==1.62.0
python scripts/verify_workbench_browser.py --browser /path/to/chrome
```

浏览器检查实际启动本机 HTTP 服务，测试故障任务、两种复验、分层包、单步存档和刷新恢复，并检查 390 像素宽的页面。发布结果以 PR 对应提交的 CI 为准。v0.6 的原始实验证据和本轮图形界面验证分别保留。

在工作台打开「地图路线」，选择世界图、起点和终点。界面按有向边的代价计算路线，并列出每一步的地图与出口 ID。射手村东部使用社区地图资料，分支路线样例使用合成世界图。没有已登记的通路时页面明确显示不可达。

「死亡之林」世界图收录两张国服怀旧版地图之间的往返。Ⅰ 的地图编号为 211041100，Ⅱ 为 211041200。每个方向各有一条已登记的连接，代价设为一次换图。出口 ID 为项目自定义标识。此样本只覆盖这两图的相互连接。[Ⅰ 的传送列表](https://mxdzlk.com/map/211041100/?v=CMSC) · [Ⅱ 的传送列表](https://mxdzlk.com/map/211041200/?v=CMSC)

在「角色键位」按区服和角色确认绑定，然后在「单步模拟器」创建该职业的会话。在「键位试放」填写相同角色名，聚焦试放按钮后按键。每次读取当前已保存的绑定；保存空白键位表即解除全部绑定。未绑定、未实现技能均不推进场景。受冷却、姿态或资源限制的模拟动作会产生真实的拒绝回执。会话最多显示最近 50 条按键记录，含按键、解析出的技能、帧号及效果事件。普通输入框中的按键不驱动模拟器。当前仅支持单键。

完整任务也可以在「角色键位配置」填写同一个区服与角色名。浏览器把当时已确认的表作为版本化输入交给离线任务；`input-bindings.json`、summary 与逐步 `execution.input_dispatch` 共同保存配置、解析键、技能和接受/拒绝原因。受支持的普通攻击、龙骑士模拟范围攻击或外部能力包 `cast` 没有绑定时，该步只推进模拟时间，不产生攻击效果，独立目标检查不能把它算作完成。`lab replay` 会从配置重新解析每一步，`lab reinfer` 还会重新执行感知和任务；只修改并重新哈希 trace 不能伪造另一枚键。没有选择角色配置的旧任务保持原行为。

此链只证明“确认键位 → 离线模拟技能”的执行与证据。移动、跳跃、拾取、传送和交互仍是模拟器动作；它不会读取、聚焦或向官方客户端发送任何输入，也不能证明真实键位截图、技能距离或国服机制已经校准。

同一个 profile manifest 也可直接进入可复现的 CLI 运行：

```bash
python -m maple_vision lab run --output artifacts/keyed-forest --job dragon_knight --map-pack examples/maps/dead-forest-drill.json --mission-program examples/missions/dead-forest-hunt.json --runtime structured --input-bindings role-keys.json
```

`role-keys.json` 使用运行包 `input-bindings.json` 的相同 schema。CLI 只接受不超过 64 KiB 的本地普通 UTF-8 JSON 文件，不跟随符号链接；工作台与 CLI 随后走同一个不可变解析器和验证链。
