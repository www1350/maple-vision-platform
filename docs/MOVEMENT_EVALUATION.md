# 录像运动候选与人工复核对照

`recording benchmark-movement` 对照独立人工复核与逐帧运动候选，输出误报、漏报、程序未知和人物不匹配清单。它先从源像素重新执行完整的跟踪、运动判断及回放校验，再读取已经复验的预测。人工标签只参与评分，不传给检测器或状态估计器。

本入口评估两个可见现象：`rope_climb_visible`（可见攀绳过程）与 `arrival_after_climb_visible`（攀爬后的可见平台到达）。结果描述候选与人工判断的对应关系，不验证抓绳物理接触、角色所有权或真实输入的动作成功率。首版只做逐帧对应，不提供事件起止误差、跨窗口身份或完整任务成功率。

身份选择可另用[目标身份评测](ACTOR_EVALUATION.md)核对。运动分析可能因置信度或地形证据不足而不接受一个已选中的人物，因此本报告的 `missing_prediction` 指运动没有采用人物，不能直接当作身份跟踪失败。独立身份评测直接读取 `actor_selection`，保留自己的分母和逐帧结果。

## 使用顺序

先导入连续录像窗口，并保留原始帧顺序和时钟。生成空白标签时需要完整、有效的导入会话；只有 manifest 或零散图片不能补出丢失帧。

```sh
python -m maple_vision recording movement-labels artifacts/window-1200 --output artifacts/movement-labels.json
```

在标签中填写 `domain`（`authored_fixture` 或 `game_recording`）、`reviewer` 和 `subject`，只编辑实际审阅过的帧的 `review`。`subject` 描述人工对照的人物，例如“中央带蓝色标记的角色”；这个描述不认证该角色属于谁。保留所有源身份字段及未审帧，不要删掉失败片段。

完成独立复核后，使用既有模型生成候选，再对照：

```sh
python -m maple_vision recording track artifacts/window-1200 --model artifacts/my-model/model.onnx --manifest artifacts/my-model/manifest.json --movement --output artifacts/movement-replay
python -m maple_vision recording benchmark-movement artifacts/window-1200 artifacts/movement-replay --labels artifacts/movement-labels.json --model artifacts/my-model/model.onnx --manifest artifacts/my-model/manifest.json --output artifacts/movement-benchmark.json
```

两次使用同一模型、manifest 和推理配置。评测从回放报告读取状态与运动配置；没有运动结果的旧回放不能直接评分。省略模型时仍使用项目合成颜色基线，仅适合工程样例，不构成国服模型验收。

## 部分复核的标签格式

模板列出窗口内的每个帧，包括本地及原片序号、两套毫秒时间、原始 PTS 和 RGB 哈希。`review: null` 表示该帧尚未人工审核。下面只是字段示例，框坐标须由实际画面独立确定，不能照抄预测框作为人工真值。

```json
{
  "actor_bbox": [100, 120, 124, 152],
  "rope_climb_visible": true,
  "arrival_after_climb_visible": false,
  "note": "连续画面中人物沿绳上移，尚未到达平台。"
}
```

两个现象各自允许 `true`、`false`、`null`。`false` 必须是人工明确的负例，不能由未列出事件推导；对象内的 `null` 表示已经审核、但该现象仍不确定。两项都是不确定时要填写原因，人物框可以为 `null`。任何一项为明确正例或负例时，都需要当前被审人物的原图整数 `xyxy` 框，右、下边界不含在内。两个现象不能同时标为 `true`；持续攀爬与攀爬后到达应分别审阅，过渡帧无法确定时保留不确定。

模板不自动把历史可见事件区间、局部几何草稿或用户背景确认转为正负标签。录像、人物与时间身份均须匹配，缺帧、重复、改序或身份不一致会拒绝评分。

## 怎样计算与查看结果

程序状态 `climbing_candidate` 对应攀爬候选为真，`platform_arrival_candidate` 对应到达候选为真，`near_rope` 对这两项均为假。`unknown` 始终属于程序弃权，不能算作正确负例。

预测所用人物轨迹须是当前帧的有效人物框，并与人工人物框达到指定 IoU。默认门槛为 0.5，可用 `--actor-iou-threshold` 明确设置；门槛会写入报告。框不匹配、没有可用预测人物或程序未知，均不授予命中；错人物产生的运动候选另列计数。几何重叠不能证明连续身份或所有权，重叠人物与遮挡仍需要人工保留不确定。

报告保留全部帧的 `diagnostics`，包含源帧号、原片时间及 PTS、人工标签、程序状态和原因、人物匹配结果以及每个现象的评分结果。可以据此回到原片逐项查看。两个现象各自汇总：

| 指标 | 含义 |
|---|---|
| `positives` / `negatives` | 人工明确的正例、负例帧数 |
| `unreviewed` / `uncertain` | 未审帧、已审但该现象不确定的帧数；不算正负例 |
| `true_positives` / `false_positives` | 人物匹配且候选为真时，人工确认的命中或误报 |
| `true_negatives` | 人物匹配、程序明确判断为否且人工也明确为否 |
| `false_negatives` | 全部人工正例中未命中的帧数，包含程序弃权 |
| `abstained_positive` / `abstained_negative` | 人工正例、负例中的程序未知或人物无法匹配数量 |
| `recall` | 命中数除以全部人工正例；未知正例保留在分母中 |
| `precision` | 命中数除以已对齐人物的正预测数；没有正预测时为 `null` |
| `decision_coverage` | 有明确程序判断且人物匹配的帧数除以人工明确正负例帧数 |

例如人工确认 10 帧在攀爬，程序命中 6 帧、明确漏判 2 帧、未知 2 帧，则召回率为 6/10，漏判总数为 4，其中 2 帧属于弃权。不能删去未知帧后写成 6/8。全部未审或不确定时，分母为空的指标为 `null`，不会给出“100% 通过”。报告不设总体准确率或发布通过阈值。

## 身份绑定与边界

报告记录标签原始字节哈希、导入 manifest、回放报告和轨迹哈希、模型元数据、状态配置、运动配置及完整重推理回执。评测前后核对输入身份；修改运动状态后只重算轨迹哈希仍会被完整复验拒绝。输出仅创建新文件，不能写入录像或回放目录，也不覆盖标签或已有报告。每次最多评测 5,000 帧，标签与输出均受编码大小上限约束。

多人录像可使用[明确目标选择](ACTOR_SELECTION.md)同时跟踪所有玩家，再对选定人物判断运动；评测将选择配置绑定到报告。未选人的旧兼容模式仍会在多人画面中返回未知，选择失效、人物或地形框尺寸变化也可能使目标运动未知。评测保留这些结果，不把它们从正例分母中移除。真实Ⅱ录像仍只有四帧局部几何草稿，不是本格式的已复核状态基准，也没有新增真实模型准确率。完整连续帧、独立标签、真实域模型与跨会话留出评测仍需各自完成。
