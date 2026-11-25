# 标飞离线试用

从仓库根目录运行以下命令。在 Windows PowerShell 中也可直接复制整行。

```sh
python -m pip install -e ".[workbench]"
python -m maple_vision lab run --seed 17 --class-pack examples/classes/throwing_star.json --map-pack examples/maps/dead-forest-drill.json --mission-program examples/missions/dead-forest-layered-hunt.json --input-bindings examples/input-bindings/throwing-star.json --output artifacts/star-trial
python -m maple_vision lab reinfer artifacts/star-trial
```

打开 `artifacts/star-trial/preview.gif` 查看采样回放。再次运行时使用新的输出目录。

复制示例键位文件即可修改 `lucky_seven` （双飞斩）的按键。S 是演示绑定。地面投掷和跳起后的空中投掷共用这个技能键。跳跃由模拟器的动作程序执行。

种子 17 的换层任务共 266 步。死亡之林Ⅰ先用 41 步到达 `platform_1`，死亡之林Ⅱ用 83 步到达 `platform_1`；两条路径都包含一条经独立物理复演的梯子边，实际执行合计 32 个 `climb` 步。每次到达后，视觉任务重新选择仍可见的地面目标，自行离开上层平台，再执行空中或地面双飞斩。全程跳跃两次、施放四次、每次两段命中、消耗八枚飞镖、击败两个目标。266 步动作回放与 267 帧完整感知/任务重推理一致。

将输入档案换为 `examples/input-bindings/spear-fighter.json` 后，第一张图仍会完成 41 步换层，但 12 次双飞斩请求全部因技能未绑定而被拒绝，零命中，任务明确失败；145 帧完整重推理一致。这证明“到达平台”不能绕过角色技能键位门禁。

## 证据边界

| 类别 | 本轮实际证据 | 不能推出的结论 |
|---|---|---|
| 用户实录 | 用户提供的死亡之林Ⅰ MP4 为 910×512、30 FPS、73.233 秒，SHA-256 `274efc91aa9bb0d63d0c3546bc2ef1810189ccabe9bb217a7f22ef6e9c901d8f`。两个受限窗口按原视频 PTS 连续覆盖全部 2197 帧（源帧 0–2196）；首段绑定接绳、层位变化、攻击动作和伤害反馈，尾段另绑定持续伤害反馈及一段攻击动作 | 稀疏复核不识别技能身份、施放者、目标或键盘输入；完整帧覆盖也不等于逐帧完整标注，不能生成双飞斩距离、命中段数或耗镖参数 |
| 合成机制 | 自写两图、平台图、梯子物理、目标视觉重识别、离开上层平台、地面/空中施放和两段结算均进入同一结构化任务链 | 不代表国服地图坐标、绳索容差、怪物位置或技能物理准确 |
| 角色键位 | 演示档案把 `S` 绑定到 `lucky_seven`；四次施放逐次保存键、技能、档案 ID 与 `bound_skill` 回执 | `S` 是自写演示键，不是从用户截图或录像自动读出的真实键位 |
| 耗镖 | 四次成功施放，每次 `ammo_cost=2`，模拟弹药从 200 降至 192；错误档案的 12 次请求均未产生攻击 | 每次两枚是当前合成职业包规则，仍需国服资料或实录独立校准 |

目标仍生成在自建练习图的地面。换层后任务会根据画面重新找到目标，再离开上层平台进行攻击；本轮没有伪造上层怪物。真实实录首段的四个区间分别为源帧 1540–1598（接绳可见）、1540–1600（层位变化可见）、1605–1610（攻击动作可见）和 1611–1669（伤害反馈可见）。尾段窗口为源帧 2145–2196，确认覆盖至解码结束；窗口内 0–51 帧有可见伤害反馈，35–45 帧有攻击动作。各报告均保存精确 PTS 时基、边界 RGB 哈希和完整 MP4 哈希。时间相邻不等于因果，也不能把攻击动作自动命名为双飞斩。真实地图绳索坐标、上层刷怪、双飞斩距离、时序、键位和耗镖仍待完整标注与独立校准。

可提交复核的摘要与原始离线证据分别保存在 `reports/release-0.10.0/throwing-star-layered-two-map-acceptance.json`、同名归档和 `throwing-star-layered-wrong-binding.tar.gz`。归档可直接交给 `lab replay` 与 `lab reinfer` 复验。真实画面边界报告为 `reports/release-0.10.0/throwing-star-real-visible-events.json` 和 `throwing-star-real-tail-visible-events.json`；原 MP4 和导入帧不进入仓库，持有原文件时可按 [录像复核命令](RECORDINGS.md#稀疏可见事件复核)重建验证。
