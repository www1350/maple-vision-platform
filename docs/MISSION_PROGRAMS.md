# 可配置地图任务

任务文件声明每张地图上按顺序完成的目标。支持与现有视觉任务共用的有限平台导航、战斗、拾取和对话，任务引擎继续处理生存抢占、动作调度、执行回执和效果验证。

下例先在 meadow 对话、战斗并拾取，再去 cavern 对话。

```json
{"schema_version":1,"stages":[
  {"map_id":"meadow","phases":["talk","fight","collect"]},
  {"map_id":"cavern","phases":["talk"]}
]}
```

```bash
python -m maple_vision lab run --mission-program examples/missions/talk-then-clear.json --output artifacts/custom-mission
python -m maple_vision lab replay artifacts/custom-mission
python -m maple_vision lab reinfer artifacts/custom-mission
```

`--mission-program` 自动选择 structured 引擎。run、suite、soak 都支持此参数，后两者在指定职业上重复同一任务。可以添加 `--class-pack` 和 `--map-pack`，例如使用 `examples/missions/three-map-visits.json` 与 `examples/maps/three-map-fixture.json`。

每个阶段对应一张地图，最多八张，地图顺序必须是所选地图包路线的前缀。schema 1 每张地图可选一至三个互不重复的目标。拾取要求本阶段已经声明战斗，以确保战利品目标有明确范围。任务间的地图移动由同一个运行时执行。

| 目标 | 视觉决策条件 | 独立验收依据 |
|---|---|---|
| navigate | 画面角色位置逐步符合已验证的平台路径 | 角色实际站上声明的平台，且路径包含要求的梯子边 |
| fight | 连续观测中看不到怪物 | 对应地图的实际怪物集合已清空 |
| collect | 连续观测中看不到掉落 | 实际怪物与掉落集合均为空 |
| talk | 对话界面可见 | 当前地图实际产生对话事件和对话内容 |

验收器按任务文件顺序记录地图、目标和帧序号。提前换图、缺少对话事件、角色死亡或目标未完成会阻止任务通过。动作重放从初始模拟状态重新收集见证记录，逐条比较报告。感知重推理再次执行视觉、时序状态和调度链。

使用自定义任务时，运行配置采用 schema2，将完整任务内容和摘要纳入运行证据，删除原任务文件后仍能重放。旧配置继续用 schema1。任务的可配置范围是上述有限目标及现有顺序地图出口。分支任务图、动态避障、任意正式地图平台导航和库存任务仍需后续扩展。

### 梯子换层阶段

任务程序 schema 2 可把 `navigate` 放在某张地图的第一个阶段，并声明目标平台与必须出现的梯子边：

```json
{"schema_version":2,"stages":[
  {"map_id":"drill_forest_1","phases":["navigate","fight"],
   "navigation":{"target_platform":"platform_1","required_edge_kind":"ladder"}}
]}
```

平台 ID 按地图包 `platforms` 数组编号。执行器用当前画面的角色框建立路径前置状态，不读取模拟器位置；平台图提供固定步动作及每步期望位置。每个动作仍经过 Task → Plan → Intent → Execution，下一画面与路径不符时任务明确失败。独立 oracle 才读取真值，确认角色确实站上目标平台。当前 `required_edge_kind` 只接受 `ladder`，避免把跳上平台误报成接绳验收。

## 工作台

在 `examples/missions/` 放置 JSON 后启动工作台。创建验证任务时可选择「任务流程」，选项按当前地图路线筛选。提交时保存任务内容，刷新页面或重新启动后仍可查看原记录。服务端会复核地图与任务是否匹配。

## 验证

```bash
python -m pytest tests/test_mission_program.py tests/test_workbench.py -q
python scripts/verify_workbench_browser.py
```

回归覆盖三种内置职业、更改目标顺序、单图提前结束、外部三图与弓手、输入文件删除、配置摘要篡改、重算文件哈希后的伪造见证、动作冻结失败和重启恢复。浏览器验证实际选择任务并检查完成报告。

## 双出口地图

在工作台选择「林地岔路」地图，选择「直达遗迹」可从 grove 直接前往 ruins。选择「三图访问」则经过 quarry。两个任务从同一个入口出发，模拟器加载同一份地图包。每个出口有固定位置和目的地，视觉任务结合当前画面中的传送门位置选择对应出口，到达后再读取地图标记。

```bash
python -m maple_vision maps route examples/maps/branch-grove.json grove ruins --map-pack
python -m maple_vision lab run --map-pack examples/maps/branch-grove.json --mission-program examples/missions/grove-shortcut.json --output artifacts/shortcut
python -m maple_vision lab reinfer artifacts/shortcut
```

schema2 地图包允许每张图声明最多四个出口，第一个出口定义默认巡回路线，其余出口可以通向包内其他地图。每张图的出口编号和目的地唯一，相邻出口的横坐标至少相距 56 像素。两图间的任务步骤必须有直接连接。寻路命令返回最短路线，任务文件决定实际执行路线。

## 国服怀旧服地图连接核查

[小册子的射手村东部小山地图详情](https://mxdc.dvg.cn/map_info.php?id=100010000)登记两个传送点。[地图工具](https://mxdc.dvg.cn/tools/maptools-v2/?dataset=cms-mxdc&map=100010000)标出西侧通向射手村、东侧通向射手村东部草丛。东部小山的地图 ID 为 100010000。示例中的节点标识和出口标识为项目自定义名称，范围标为 game_reference。图中每条有向边均有资料依据。平台、梯子和行走距离仍待录像校准。

```bash
python -m maple_vision maps route examples/navigation/cms-east-hill.json east_hill henesys
python -m maple_vision maps route examples/navigation/cms-east-hill.json east_hill east_grass
```

两条查询分别返回 west 和 east，均通过路线复核。每条边的代价 1 表示一次换图。真实画面中的门识别、平台跳跃和梯子攀爬仍需要单独实测。

### 射手村往返青蘑菇树林

本样本覆盖五张地图之间的八条有向连接。每一段代价为一次跨图。射手村前往东部小山依据小册子的推荐走法，其余连接依据出发地图的传送点列表。出口名称以目的地作为程序标识。

| 地图 | 核对依据 |
|---|---|
| 射手村 → 东部小山 | [东部小山详情的推荐走法](https://mxdc.dvg.cn/map_info.php?id=100010000) |
| 东部小山 → 射手村、东部草丛 | [东部小山地图分布](https://mxdc.dvg.cn/map_info.php?id=100010000) |
| 东部草丛 → 东部小山、东部树林 | [东部草丛地图分布](https://mxdc.dvg.cn/map_info.php?id=100020000) |
| 东部树林 → 东部草丛、青蘑菇树林 | [东部树林地图分布](https://mxdc.dvg.cn/map_info.php?id=100030000) |
| 青蘑菇树林 → 东部树林 | [青蘑菇树林地图分布](https://mxdc.dvg.cn/map_info.php?id=100030001) |

```bash
python -m maple_vision maps route examples/navigation/cms-east-hill.json henesys mushroom_forest
python -m maple_vision maps route examples/navigation/cms-east-hill.json mushroom_forest henesys
```

## 双图近战演练

在工作台选择龙骑士、「死亡之林双图演练（合成）」和「两图各击败一只怪物」。两张图各放一只通用模拟怪物，角色须在第一张图接敌并击杀，移动至传送门，进入第二张图再次接敌并击杀。工作台的「死亡之林」资料路线和此演练使用不同标识。演练平台、出生点、怪物属性和技能数值均来自独立模拟器。

以下命令用已登记的混合 ONNX 模型保存每一步的画面，然后重新执行感知与策略。默认种子为 0。去掉模型和清单参数可测像素基线；加上 `--fault freeze_input` 可核对无法走位时的失败结果。

```bash
python -m maple_vision lab run --job dragon_knight --map-pack examples/maps/dead-forest-drill.json --mission-program examples/missions/dead-forest-hunt.json --record-stride 1 --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json --output artifacts/forest-hunt
python -m maple_vision lab replay artifacts/forest-hunt
python -m maple_vision lab reinfer artifacts/forest-hunt --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
```
