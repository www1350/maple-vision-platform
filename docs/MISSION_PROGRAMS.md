# 可配置地图任务

任务文件声明每张地图上按顺序完成的目标。支持与现有视觉任务共用的战斗、拾取和对话，任务引擎继续处理生存抢占、动作调度、执行回执和效果验证。

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

每个阶段对应一张地图，最多八张，地图顺序必须是所选地图包路线的前缀。每张地图可选一至三个互不重复的目标。拾取要求本阶段已经声明战斗，以确保战利品目标有明确范围。任务间的地图移动由同一个运行时执行。

| 目标 | 视觉决策条件 | 独立验收依据 |
|---|---|---|
| fight | 连续观测中看不到怪物 | 对应地图的实际怪物集合已清空 |
| collect | 连续观测中看不到掉落 | 实际怪物与掉落集合均为空 |
| talk | 对话界面可见 | 当前地图实际产生对话事件和对话内容 |

验收器按任务文件顺序记录地图、目标和帧序号。提前换图、缺少对话事件、角色死亡或目标未完成会阻止任务通过。动作重放从初始模拟状态重新收集见证记录，逐条比较报告。感知重推理再次执行视觉、时序状态和调度链。

使用自定义任务时，运行配置采用 schema2，将完整任务内容和摘要纳入运行证据，删除原任务文件后仍能重放。旧配置继续用 schema1。任务的可配置范围是上述有限目标及现有顺序地图出口。分支任务图、任意平台闭环导航和库存任务仍需后续扩展。

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
