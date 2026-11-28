# 多人画面中的目标人物选择

多人同屏是正常输入。检测与跟踪负责保留所有可见玩家，目标选择另外回答“这次要持续观察谁”。`recording track --actor-selection` 将一个经过人工选定的源帧人物框绑定到几何轨迹，再对这个目标判断运动；其他玩家、地形和原始竞争框仍保留在同一观测中。

当前提供离线 API 和 CLI，尚未提供点击人物的图形界面。它不会将人工框当成模型预测，不认证角色所有权，也不启动游戏输入。

## 选定一个实际源帧

在已导入的连续录像窗口中，选一帧能清楚区分目标人物的画面，并填写人物框。示例的帧号和坐标只是调用格式，须按实际源画面填写。

```sh
python -m maple_vision recording select-actor artifacts/example-recording --source-frame 17 --bbox 100 120 124 152 --reviewer local-reviewer --subject "reviewed visible actor" --output artifacts/actor-selection.json
python -m maple_vision recording track artifacts/example-recording --model artifacts/my-model/model.onnx --manifest artifacts/my-model/manifest.json --actor-selection artifacts/actor-selection.json --movement --output artifacts/selected-replay
python -m maple_vision recording verify-tracking artifacts/example-recording artifacts/selected-replay --model artifacts/my-model/model.onnx --manifest artifacts/my-model/manifest.json
```

`--source-frame` 是原片的零基帧号，不是窗口内序号。例如窗口从 1200 开始，选择原片 2228 帧就填 2228。框采用原始画面的 `x1 y1 x2 y2`，右、下边界不含在内。工具从实际 manifest 读取这帧的窗口内时间和 RGB 哈希，同时绑定整段窗口身份；越界、来源不同或帧身份不符都会拒绝。

可用 `--selection-config path.json` 提供完整的 `ActorSelectionConfig.to_dict()` 配置，默认锚点 IoU 为 0.5、人物置信度门槛为 0.7、最大帧年龄和连续时间间隔均为 250 毫秒。这些是工程参数，未按国服画面标定。人物关联另使用回放中保存的 `StateConfig`，选择标识也绑定该配置。

选定文件保存源身份、人物锚点和选择配置。生成的回放内嵌同一份信息，复验不依赖原选定文件继续存在。不提供模型时，选定模式仍使用项目颜色检测基线，但通过 detector-only 适配器进入只读状态；不会解析或猜测真实地图、HP、MP。

## 三个选择状态

| 状态 | 行为 |
|---|---|
| `waiting` | 还没有到达指定锚点帧。保留所有人物轨迹，目标运动保持未知，不使用锚点之前的动作积累。 |
| `selected` | 锚点对应唯一可信的当前人物，此后同一轨迹在连续帧中无歧义匹配。其他玩家可以持续存在或在远处进出。 |
| `invalidated` | 锚点不符、目标丢失、关联歧义、重新关联、来源／epoch／时间／帧序失效等。此次选择持续失效，其他人物仍可跟踪，目标运动保持未知。 |

锚点处比较当前模型人物框与人工框；后续不再拿初始框的位置追人，而是沿同一轨迹的连续测量。原始低置信度玩家也参与局部竞争检查：与目标框或锚点重叠，或者在上一测量的预测位置上形成同等或更好的关联候选，会使选择失效。远处玩家不会仅因存在就使目标失效。

目标失效后，不会自动改选最近的人，也不会因原轨迹编号再次出现而自动恢复。需要在新的一次运行中明确选定新的可靠锚点。运动估计器一旦进入选人模式，遗漏选择结果也不会退回默认单人推断；改变选择需使用新的估计器，避免借用之前的人物运动证据。

## 跟踪、运动与回放

底层 `TemporalStateEstimator` 新增默认关闭的 `retain_multiple_player_hypotheses` 参数，需要与 `retain_detection_hypotheses=True` 一起启用。录像选人模式自动开启它。当前多个高置信度人物分别进入轨迹；单一 `player_bbox` 在存在任何第二个原始玩家框时清空，不任取一个框充当主角。地图、资源和可行动状态继续保持未知。

运动估计器接受当前帧的 `ActorSelection`，核对同一份 belief、源序号、时间和 epoch，然后使用所选轨迹。完整观测仍包括其他玩家和地形；镜头补偿继续只依赖地形。人物框尺寸变化、地形不足和绳索／平台关系不明仍可能使运动候选未知，即使人物选择本身还有效。

`tracks.jsonl` 记录每帧选择状态和原因，运动记录也保存同一选择结果。GIF 用蓝色 `TARGET` 标出当前选定人物，其他玩家显示独立轨迹；失效时显示原因，蓝色目标标记消失。报告保存选择算法版本、完整锚点与配置、状态及原因计数。`verify-tracking` 从像素重新运行检测、多人关联、目标选择、运动判断和绘图；仅修改所选编号并重算文件哈希不能通过。

不提供 `--actor-selection` 时，旧单人推断行为、身份计算和回放字节保持兼容。该兼容模式不会自动选择多人中的目标；在多人录像中需要明确选择要观察的人物。

## 与独立评测的关系和限制

`recording benchmark-movement` 可直接评测选人模式的回放，保存选择配置的绑定。选择锚点用于定义观察对象，人工评测标签用于比较每帧现象，两者用途不同。人工选中一个框不表示模型正确识别了它；选择失败、程序未知和错人物仍按评测规则计数。

当前机制是绑定人工锚点的几何跟踪，**不证明真实连续身份**。如果两个视觉上无法区分的人物在相邻采样帧间互换位置，几何关联仍可能认错；姓名、外观特征和跨遮挡重新识别尚未实现。数据哈希证明一致性，不认证来源、人工审核或所有权；进程内构造的选择对象也不是经过外部认证的身份凭证。

独立工程用例覆盖双人同屏静止／攀爬对照、镜头跟随、远处人物进出、局部交叉、低置信度竞争框、目标丢失、锚点前等待及完整复验。多人是用户原始录像已经具有的场景；本改动修正工程的单人假设，没有增加真实域训练、完整录像标注或实机操作验收结果。
