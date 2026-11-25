# 标飞离线试用

从仓库根目录运行以下命令。在 Windows PowerShell 中也可直接复制整行。

```sh
python -m pip install -e ".[workbench]"
python -m maple_vision lab run --seed 17 --class-pack examples/classes/throwing_star.json --map-pack examples/maps/dead-forest-drill.json --mission-program examples/missions/dead-forest-hunt.json --input-bindings examples/input-bindings/throwing-star.json --output artifacts/star-trial
python -m maple_vision lab reinfer artifacts/star-trial
```

打开 `artifacts/star-trial/preview.gif` 查看采样回放。再次运行时使用新的输出目录。

复制示例键位文件即可修改 `throw_star` 的按键。S 是演示绑定。地面投掷和跳起后的空中投掷共用这个技能键。跳跃由模拟器的动作程序执行。

种子 17 的合成双图实测共 130 步，跳跃两次、投掷四次、四次命中、击败两个目标，131 帧重推理一致。将输入档案换为 `examples/input-bindings/spear-fighter.json` 后，12 次投掷请求均因缺少技能绑定而被拒绝，零次投掷，任务以失败结束，89 帧重推理一致。

本样例覆盖自建死亡之林练习图上的两个地面目标。真实地图的绳索换层路线、国服技能距离和时序仍待验证。
