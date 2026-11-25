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
