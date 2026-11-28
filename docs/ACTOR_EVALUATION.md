# 多人录像中的目标身份复核与评测

`recording benchmark-actor` 单独回答“这一帧选中的是否仍是要观察的人物”。它直接比较已复验的目标选择与独立人工人物参考，支持没有运动分析的回放。人物选对而运动证据不足时，身份仍可以正确计分；同一轨迹编号已经关联到其他玩家时，也能被人工参考揭示。

当前评分针对一个完整导入窗口。它测量程序与人工可见身份判断的对应关系，不认证角色所有权，也不证明遮挡期间的真实身份。未审帧、人工不确定和程序弃权分别保留。

## 从画面选人到独立复核

在[工作台录像页](WORKBENCH_NAVIGATION.md)选择一个已注册的完整录像窗口，定位清楚可辨的一帧，点击“框选人物”，在原画面上拖出人物框。可以放大查看身体、名字和标记；框选仍按原图坐标保存。填写复核人与目标说明，确认当前选区后下载 `actor-selection.json`。

该文件定义跟踪的起始对象。逐帧身份复核标签另行保存，用于独立评分；不要把程序预测框或选人锚点自动填成整段参考答案。

```sh
python -m maple_vision recording actor-labels artifacts/window-1200 --output artifacts/actor-labels.json
python -m maple_vision recording track artifacts/window-1200 --model artifacts/my-model/model.onnx --manifest artifacts/my-model/manifest.json --actor-selection actor-selection.json --output artifacts/actor-replay
python -m maple_vision recording benchmark-actor artifacts/window-1200 artifacts/actor-replay --labels artifacts/actor-labels.json --model artifacts/my-model/model.onnx --manifest artifacts/my-model/manifest.json --output artifacts/actor-benchmark.json
```

生成标签模板后，需要填写 `domain`、`reviewer`、`subject` 及实际审核帧的 `review`，再运行最后的评分命令。`domain` 为 `authored_fixture` 或 `game_recording`。推理与复验使用同一模型及 manifest；省略模型只使用项目颜色基线，适合工程用例，不代表已有国服检测模型。可在跟踪时另外加 `--movement`，身份评分的结果不依赖运动状态。

工作台目前提供图形选人和导出；录像跟踪及身份评测仍由上述 CLI 执行。页面没有启动真实录像推理的按钮，也没有把现有合成模型变成真实域模型。

## 逐帧标签

模板绑定窗口内所有帧，包括窗口序号、绝对源帧号、窗口时间、原片时间、原始 PTS 和 RGB 哈希。未审核的帧保留 `review: null`，不要删除、重排或补写缺失帧。

目标明确可见时，一个 `review` 如下。坐标仅用于说明格式，须根据实际原图独立填写：

```json
{
  "visibility": "visible",
  "actor_bbox": [100, 120, 124, 152],
  "other_actors": [
    {"actor_id": "player-B", "bbox": [170, 130, 194, 162]}
  ],
  "note": "目标下方标记可分辨，另一玩家位于右侧。"
}
```

| 复核状态 | 含义和字段要求 |
|---|---|
| `review: null` | 尚未审核。不代表目标缺席或程序正确。 |
| `visible` | 能明确指出目标当前的可见人物框；`actor_bbox` 必填。 |
| `not_visible` | 已复核，明确没有可辨认的目标人物；`actor_bbox` 为 `null`，必须说明离开画面、完全遮挡等原因。 |
| `uncertain` | 已复核，但无法确定目标是否可见、或哪一个人物是目标；`actor_bbox` 为 `null`，必须填写原因。 |

人物框使用原图整数 `x1, y1, x2, y2`，右、下边界不含在内，必须具有正面积并在画面范围内。每帧最多标 16 名其他人物。`actor_id` 是人工参考编号，必须使用简短 ASCII 字符且帧内唯一，不能使用保留名称 `target`。**同一个可辨认人物跨帧应使用同一个人工编号**；程序轨迹 ID 不能替代这项独立身份判断。

`other_actors` 可以只包含已经明确复核的其他人物。没有匹配任何参考框时，程序只能报告定位不匹配，不能据此确定它跟上了哪一个未标注的人物。身体重叠、名字归属不清或身份无法持续确认时，保留不确定；不要根据程序的选择反向补标签。

## 每帧会得到什么结果

先从源像素完整重跑检测、多人关联、目标选择及叠加回放，并确认全部输出一致，然后才读取人工标签评分。默认人物 IoU 门槛为 0.5，可用 `--actor-iou-threshold` 指定，实际门槛与配置哈希进入报告。

| `outcome` | 解释 |
|---|---|
| `correct_target` | 目标可见，选择框唯一匹配目标参考。 |
| `wrong_other_actor` | 目标可见，但选择框唯一匹配另一名已标注人物。 |
| `localization_mismatch` | 目标可见且程序选中了人，但未达到任一参考人物的匹配门槛。 |
| `ambiguous_reference_match` | 同一个选择框同时达到多个参考人物的匹配门槛，不授予正确命中。 |
| `missed_visible_target` | 目标明确可见，程序仍在等待或已经失效，没有选中目标。 |
| `correct_abstention_not_visible` | 目标明确不可见，程序也没有声称选中。 |
| `selected_when_not_visible` | 目标明确不可见，程序仍声称选中某个人；匹配到的其他参考身份另行保留。 |
| `unreviewed` | 该帧没有人工复核。 |
| `reference_uncertain` | 人工已经审核，但身份仍不确定。 |

`diagnostics` 保留每帧的源时间和像素身份、人工说明、目标及其他人物框、程序选择状态与原因、轨迹 ID、各参考框的 IoU 和最终结果。可以据此回到原片检查失效或串人位置。

## 分母与连续性

目标召回 `target_recall` 是正确目标帧数除以全部人工明确可见的目标帧数。等待、丢失后持续失效、错跟别人、定位不匹配和参考匹配歧义都留在这个分母。程序一直弃权不会获得完美成绩。

`visible_selection_coverage` 只表示可见目标帧中程序有多少帧作出了选择；其中可能选错人，必须与目标召回和各类错误计数同时查看。`not_visible_selection_rate` 单独统计目标明确不可见时仍选择人物的比例。未审和不确定不当负例；任何比例的分母为空时返回 `null`。

连续性报告保留：

- `eligible_adjacent_identity_pairs`：源帧相邻、双方都有明确复核且唯一匹配参考身份的帧对。
- `reference_identity_changes`：这些帧对中匹配的人工身份发生变化的次数。内部轨迹 ID 没变，但从目标变成另一人，也会计数。
- `correct_target_intervals`：每段连续正确选中目标的源帧和原片时间范围。
- `longest_correct_target_run_frames`：最长连续正确区间的帧数。

未审、不确定、匹配歧义、定位不匹配或没有选中都会切断连续性。不会跨这些缺口推算身份变化或连续成功；`reference_identity_changes` 也不是标准多目标跟踪评测中的 MOT IDSW。完整报告同时保留帧级分母，避免只看“串人次数为零”而忽略目标早已失效。

## 来源绑定与当前实录边界

报告绑定源 manifest、标签原始字节、跟踪报告、轨迹、GIF、模型元数据、状态和选人配置、评测算法版本与门槛。评分前后重新检查输入；篡改选中轨迹 ID 后仅刷新文件哈希，不能通过完整重推理。输出只创建新文件，不覆盖标签或旧报告，也不写入录像与回放目录。每次最多评测 5,000 帧。

死亡之林Ⅱ现存四张边界 PNG 可以帮助复核人物与原图坐标；它们仍是局部几何草稿，没有自动转成这份身份标签。按暂定人物对应，目标与另一名绳上玩家的上下顺序发生过交换，交叉附近是后续连续身份验收的重点。中间缺失的像素、独立身份标签、真实域模型和跨会话留出评测仍需完成；本增量不产生这段完整实录的身份准确率。
