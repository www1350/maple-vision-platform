# 本机工作台与两层地图导航

## 运行

需要 Python 3.12。以下命令在仓库根目录执行。浏览器打开 http://127.0.0.1:8765 。

```bash
python -m pip install -e ".[workbench]"
python -m maple_vision workbench --workspace artifacts/workbench --examples examples
```

要逐帧复核已经导入的真实录像，可把会话分别放在一个固定目录的直接子目录中，并把已复验的可见事件 JSON 放在另一个固定目录：

```bash
python -m maple_vision workbench --workspace artifacts/workbench --examples examples \
  --recordings artifacts/recording-sessions \
  --recording-events artifacts/recording-events \
  --recording-coverage artifacts/window-coverage.json
```

| 页面 | 操作和证据 |
|---|---|
| 任务与回放 | 选择职业、地图、感知和故障，启动一次 structured 任务。运行记录保留原始目标结果、终止原因、采样帧和可分页的动作/观测。持续丢帧会产生可以检查的停止记录 |
| 回放复验 | 在「验证方式」中选择动作重放、感知与任务重推理、导出分层包或验证分层包，再选择来源。分层包提供全部 N+1 帧；动作重放和重新感知分别执行 |
| 真实录像复核 | 启动时完整校验固定目录中的录像会话、可见事件和可选覆盖报告。覆盖报告通过后可用一条连续时间轴跨窗口逐帧显示原始导入像素、全局 PTS 派生时间和当前帧事件；也可下拉选择单窗口。上一个/下一个可见事件按钮只在已复验事件的起始帧间跳转，并显示当前覆盖的事件与精确帧区间。浏览器不能提交文件路径，帧读取时重新核对像素哈希 |
| 单步模拟器 | 手动推进、保存、重置、恢复。每一步 50 ms 模拟时间。视觉观测由渲染帧产生，验证真值来自模拟器。修订号拒绝基于旧状态提交的操作 |
| 内容与模型 | 展示本次启动载入的职业包版本与技能、地图路线、感知方式。省略 `--examples` 可直接使用三个内置职业与内置两图。ONNX 混合模型还需要 `pip install -e ".[workbench,inference]"` 和对应模型文件 |

工作台使用单机 SQLite 和每个任务的独立证据目录。每次最多四个排队或运行中的任务，一个工作线程实际执行。排队任务可取消，已运行任务保留其终止结果。重新启动同一目录可以查看已有结果，进程中断时未完成的任务记录为中断。已打开的单步会话在刷新页面后恢复。证据检查以现有 lab 规则为准；原始数据与图形界面分开保存。

录像根目录、事件目录和覆盖报告只在服务启动时指定；目录内不接受符号链接，最多登记 128 个会话、256 份事件报告和 64 份覆盖报告。启动会完整校验 manifest、全部帧哈希、事件到源帧的绑定，以及覆盖报告引用的会话身份、来源、PTS 时基、连续性和解码停止状态。覆盖报告可重复传入；其全部窗口都必须已经在 `--recordings` 根目录登记。界面不返回本地路径。逐帧读取仍核对所选 PNG 的编码哈希、RGB 哈希和尺寸。固定录像不能验证反事实动作或动作成功，可见事件也不识别技能、施放者、目标或键盘输入；合成机制、角色键位和耗镖证据继续单列。

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

在「角色键位」可以选择本地设置截图，对照画面填写技能名与键位。图片只在当前页面预览，切换角色时清除；浏览器在本地计算图片 SHA-256、字节数和尺寸，原图不上传。按区服和角色确认绑定，然后在「单步模拟器」创建该职业的会话。在「键位试放」填写相同角色名，聚焦试放按钮后按键。每次读取当前已保存的绑定；保存空白键位表会解除全部绑定并清除旧确认凭证。未绑定、未实现技能均不推进场景。受冷却、姿态或资源限制的模拟动作会产生真实的拒绝回执。会话最多显示最近 50 条按键记录，含按键、解析出的技能、帧号及效果事件。普通输入框中的按键不驱动模拟器。当前仅支持单键。

完整任务也可以在「角色键位配置」填写同一个区服与角色名。浏览器把当时已确认的表和独立确认凭证一并交给离线任务；缺少或损坏凭证会要求回到角色键位页重新核对。`input-bindings.json`、`input-binding-confirmation.json`、summary 与逐步 `execution.input_dispatch` 共同保存确认来源、收据哈希、配置、解析键、技能和接受/拒绝原因。受支持的普通攻击、龙骑士模拟范围攻击或外部能力包 `cast` 没有绑定时，该步只推进模拟时间，不产生攻击效果，独立目标检查不能把它算作完成。`lab replay` 会复核确定性凭证并从配置重新解析每一步，`lab reinfer` 还会重新执行感知和任务；只修改并重新哈希 trace 或替换另一角色的凭证都不能通过。没有选择角色配置的旧任务保持原行为；直接调用旧 API/CLI 仍可只提供 profile，但不会显示为已有确认凭证。

此链只证明“确认键位 → 离线模拟技能”的执行与证据。移动、跳跃、拾取、传送和交互仍是模拟器动作；它不会读取、聚焦或向官方客户端发送任何输入，也不能证明真实键位截图、技能距离或国服机制已经校准。

同一个 profile manifest 也可直接进入可复现的 CLI 运行：

```bash
python -m maple_vision lab run --output artifacts/keyed-forest --job dragon_knight --map-pack examples/maps/dead-forest-drill.json --mission-program examples/missions/dead-forest-hunt.json --runtime structured --input-bindings role-keys.json
```

`role-keys.json` 使用运行包 `input-bindings.json` 的相同 schema。CLI 只接受不超过 64 KiB 的本地普通 UTF-8 JSON 文件，不跟随符号链接；工作台与 CLI 随后走同一个不可变解析器和验证链。

外部职业也走同一链。`examples/input-bindings/spear-fighter.json` 将合成键 `S` 绑定到合成技能 `triple_thrust`；固定 seed 17 在死亡之林两图各施放一次，每次 3 段，共 2 次绑定接受、6 个伤害事件和 2 次击杀。102 步 action replay 与 103 帧完整 reinfer 均通过，摘要及文件哈希见 `reports/release-0.10.0/keyed-spear-two-map-acceptance.json`。该 fixture 不代表用户真实键位，也不代表国服枪连击参数已校准。

「导出已保存的键位 JSON」使用当前角色最后确认的绑定。下载的 `input-bindings.json` 可直接用于 `lab run --input-bindings`。截图与尚未保存的编辑不进入这个执行配置。

「导出键位确认凭证」另存 `input-binding-confirmation.json`。凭证包含同一份已验证 profile、profile 哈希、人工对照来源，以及所选截图由浏览器上报的本地哈希和尺寸；若玩家直接对照当前游戏窗口，则来源明确记为 `live_window_manual_comparison`。截图字节不保存，服务端也不复算图片哈希。凭证的 `automatic_skill_recognition=false` 与 `candidates_generated=false` 固定说明本阶段是玩家人工核对，不把截图预览冒充技能图标识别，也不证明键已实际按下或客户端接受了输入。

若玩家在同一截图上完成 A–L 九格复核，确认凭证升级为 schema 2，并保存 A/L 原图坐标、固定采样参数、算法标出的深色内容格、玩家纠正后的内容格及增删项。玩家必须点「确认九格复核结果」后才会写入；换图会清空草稿。这里的候选表示键格里可能存在深色内容，不能识别技能，浅色图标可能漏标，键帽文字也可能触发。没有九格复核的旧 schema 1 凭证字节保持兼容。

单步试放会把所选角色配置名与按键回执一起保存在本地会话历史。攻击回执按该步的直接伤害事件显示总命中段数，跨目标相加；零段表示这次已执行的攻击没有直接命中。旧会话中未记录角色名的按键保留原记录。

确认非空角色键位时会先调用本地执行器的配置校验。比如 Ctrl 和 Control 指向同一键，两行同时绑定会提示冲突，并保留上次已确认的配置。中文多绑定配置可用于导出、单步试放和整段任务；清空并确认仍可解除全部绑定，但不会生成确认凭证。

工作台可载入 CLI 或工作台导出的键位 JSON。载入后显示文件中的角色和键位供逐项核对，点击确认后才替换该角色的本地配置。超过编辑器容量或无法原样表示的字段会提示导入失败。

录像复核时，在「源帧号」输入复核队列中的帧号并按回车，即可暂停播放并定位。连续覆盖视图支持跨窗口跳转，单窗口视图只接受该窗口覆盖的源帧号。

在录像画面下方点击「框选人物」，拖出要观察的人物范围；可使用 2× 放大、滚动与局部预览检查人物及其标记。原图整数坐标始终显示在选人面板，缩放留白不会计入人物框。填写复核人和目标说明、确认当前选区后，可下载 `actor-selection.json`，直接用于 `recording track --actor-selection`。换帧、换录像、播放或改动选区与说明会撤销旧确认，未加载完成的画面与迟到请求不能导出旧选择。

选人请求只携带已注册会话身份、绝对源帧号、当前帧 RGB 哈希、原图人物框和人工说明。服务端重新检查完整录像，并从对应窗口取得本地锚点时间；跨窗口覆盖视图的显示时间不会被误写成窗口时间。请求不包含本地文件路径，原录像目录保持只读。本页当前完成图形选人和导出，录像跟踪与独立身份／运动评测仍通过 CLI 运行。[目标选择](ACTOR_SELECTION.md) · [目标身份复核](ACTOR_EVALUATION.md)
