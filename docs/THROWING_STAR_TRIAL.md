# 标飞离线试用

从仓库根目录运行以下命令。在 Windows PowerShell 中也可直接复制整行。

```sh
python -m pip install -e ".[workbench]"
python -m maple_vision lab run --seed 17 --class-pack examples/classes/throwing_star.json --map-pack examples/maps/dead-forest-drill.json --mission-program examples/missions/dead-forest-layered-hunt.json --input-bindings examples/input-bindings/throwing-star.json --output artifacts/star-trial
python -m maple_vision lab reinfer artifacts/star-trial
```

打开 `artifacts/star-trial/preview.gif` 查看采样回放。再次运行时使用新的输出目录。

安装后也可运行 `maple-vision-practice`，核对并确认任意 A–Z 双飞斩键，再点击「生成双图离线路线」。这个入口把可见图块确认产生的 profile 和收据原样交给上述双图任务。输出的 `player-review.json` 给出两次换层和四次出手的源序号；`preview.gif` 展示完整合成路线。移动和战斗仍是自有模拟器动作，不是正式客户端输入。

复制示例键位文件即可修改 `lucky_seven` （双飞斩）的按键。S 是演示绑定。地面投掷和跳起后的空中投掷共用这个技能键。跳跃由模拟器的动作程序执行。

种子 17 的换层任务共 266 步。死亡之林Ⅰ先用 41 步到达 `platform_1`，死亡之林Ⅱ用 83 步到达 `platform_1`；两条路径都包含一条经独立物理复演的梯子边，实际执行合计 32 个 `climb` 步。每次到达后，视觉任务重新选择仍可见的地面目标，自行离开上层平台，再执行空中或地面双飞斩。全程跳跃两次、施放四次、每次两段命中、消耗八枚飞镖、击败两个目标。266 步动作回放与 267 帧完整感知/任务重推理一致。

将输入档案换为 `examples/input-bindings/spear-fighter.json` 后，第一张图仍会完成 41 步换层，但 12 次双飞斩请求全部因技能未绑定而被拒绝，零命中，任务明确失败；145 帧完整重推理一致。这证明“到达平台”不能绕过角色技能键位门禁。

## 证据边界

种子 17 的合成录像可按动作序号复核出手姿态。账本命令在 `synthetic_mechanics.cast_witnesses` 返回每次施放前的触地状态和结算后的命中、弹药。

| 练习图 | 动作序号 | 双飞斩出手姿态 | 命中段数 | 模拟飞镖 |
|---|---:|---|---:|---|
| 死亡之林Ⅰ | 71 | 空中 | 2 | 200 → 198 |
| 死亡之林Ⅰ | 103 | 地面 | 2 | 198 → 196 |
| 死亡之林Ⅱ | 250 | 空中 | 2 | 196 → 194 |
| 死亡之林Ⅱ | 262 | 空中 | 2 | 194 → 192 |

| 类别 | 本轮实际证据 | 不能推出的结论 |
|---|---|---|
| 用户实录 | 用户提供的死亡之林Ⅰ MP4 为 910×512、30 FPS、73.233 秒，SHA-256 `274efc91aa9bb0d63d0c3546bc2ef1810189ccabe9bb217a7f22ef6e9c901d8f`。两个受限窗口按原视频 PTS 连续覆盖全部 2197 帧（源帧 0–2196）；现有 6 个事件区间并集覆盖 178 帧（8.101957%），全片另切为 15 个待审段 | 15 段均未完成逐帧复核，区间外 2019 帧不是负样本。稀疏复核不识别技能身份、施放者、目标或键盘输入，不能生成双飞斩距离、命中段数或耗镖参数 |
| 合成机制 | 自写两图、平台图、梯子物理、目标视觉重识别、离开上层平台、地面/空中施放和两段结算均进入同一结构化任务链 | 不代表国服地图坐标、绳索容差、怪物位置或技能物理准确 |
| 角色键位 | 演示档案把 `S` 绑定到 `lucky_seven`；四次施放逐次保存键、技能、档案 ID 与 `bound_skill` 回执 | `S` 是自写演示键，不是从用户截图或录像自动读出的真实键位 |
| 耗镖 | 四次成功施放，每次 `ammo_cost=2`，模拟弹药从 200 降至 192；错误档案的 12 次请求均未产生攻击 | 每次两枚是当前合成职业包规则，仍需国服资料或实录独立校准 |

目标仍生成在自建练习图的地面。换层后任务会根据画面重新找到目标，再离开上层平台进行攻击；本轮没有伪造上层怪物。真实实录首段的四个区间分别为源帧 1540–1598（接绳可见）、1540–1600（层位变化可见）、1605–1610（攻击动作可见）和 1611–1669（伤害反馈可见）。尾段窗口为源帧 2145–2196，确认覆盖至解码结束；窗口内 0–51 帧有可见伤害反馈，35–45 帧有攻击动作。各报告均保存精确 PTS 时基、边界 RGB 哈希和完整 MP4 哈希。时间相邻不等于因果，也不能把攻击动作自动命名为双飞斩。真实地图绳索坐标、上层刷怪、双飞斩距离、时序、键位和耗镖仍待完整标注与独立校准。

可提交复核的摘要与原始离线证据分别保存在 `reports/release-0.10.0/throwing-star-layered-two-map-acceptance.json`、同名归档和 `throwing-star-layered-wrong-binding.tar.gz`。归档可直接交给 `lab replay` 与 `lab reinfer` 复验。真实画面边界报告为 `reports/release-0.10.0/throwing-star-real-visible-events.json` 和 `throwing-star-real-tail-visible-events.json`；两段连续性另由 `throwing-star-real-window-coverage.json` 绑定，覆盖哈希为 `0b2c6165e350838cce0c00263cdabb9f01efeff57aee5fc74ceea6e2c402c21f`。全片待审分母保存在 `throwing-star-real-review-queue.json`，15 段均为待审，队列哈希为 `84e4ed96bda7811492af9965cd8a023840b7c690269635210483be79b79cfc24`。原 MP4 和导入帧不进入仓库，持有原文件时可按[录像命令](RECORDINGS.md)重建验证。

四类证据的机器可复验索引为 `reports/release-0.10.0/throwing-star-trial-evidence-ledger.json`。运行 `python -m maple_vision lab verify-trial-evidence reports/release-0.10.0/throwing-star-trial-evidence-ledger.json --root .` 会重新核对全部引用文件哈希、实录覆盖与事件报告的来源关系、合成输入、正反键位归档，并实际执行两份归档的动作重放和完整重推理。返回值仍将真实录像、合成机制、键位与耗镖分栏；仓库没有原 MP4 和导入帧，因此真实栏只复验持久报告及其交叉关系，明确返回 `raw_pixels_reverified=false`。文本证据校验会先把 CRLF 规范化为 LF，以保持 Windows 与 Linux Git checkout 一致；二进制运行归档仍按原始字节严格校验。

可把同一账本只读挂到工作台：`python -m maple_vision workbench --workspace artifacts/workbench --examples examples --trial-evidence-root . --trial-evidence-ledger reports/release-0.10.0/throwing-star-trial-evidence-ledger.json`。服务启动时先完成全部复验，浏览器的「试用验收账本」页只展示四栏结果和限制，不能从页面改路径或提升真实域结论。

## 死亡之林Ⅰ开场的局部画面复核

对上表 SHA-256 对应的 910×512 原片，以原始桌面左上角为原点，逐帧查看源帧 120–149 左侧平台区域（左 187、上 205、宽 176、高 105）。共查看 30 个连续裁剪画面。核对完整源帧 149 时，右侧平台也有独立的紫色数字。视觉核验需要同时保留反馈的位置和目标归属；单凭整窗出现数字会混入另一处反馈。

| 源帧（从 0 起） | 原片时间 | 左侧区域的可见变化 |
|---|---|---|
| 128 → 129 | 4.267 → 4.300 秒 | 角色左侧出现明亮红白弧形特效。 |
| 136 → 137 | 4.533 → 4.567 秒 | 特效所在侧出现首个粉色数字组。 |
| 137–149 | 4.567–4.967 秒 | 后续帧数字纵向堆叠，末帧仍有数字；第 148 帧起右边再次出现弧形特效。 |

这一次两个可见起点相隔 8 帧，约 267 毫秒。按键时刻、技能身份、命中归属和库存均未知，因此只能用于核对视觉事件的先后顺序。该局部复核保留既有全片待审分母，未将任何一段标为完整审核。

持有原 MP4 时可用 FFmpeg 重建本次复核图。从左到右、从上到下每格一帧，首格 120，末格 149。

```sh
ffmpeg -v error -i "Desktop 2026.11.25 - 22.50.07.01.mp4" -map 0:v:0 -vf "select='between(n,120,149)',format=rgb24,crop=176:105:187:205,tile=6x5" -frames:v 1 opening-left-platform.png
```
