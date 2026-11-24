# v0.3 地形研究候选：未通过推广门禁

此目录保留一次真实训练的失败候选，供独立复现。**它不是已验收的地形模型，没有替换默认像素管线或 `examples/lifecycle-model/` 的已验收混合管线。**

| 项目 | 值 |
|---|---|
| ONNX SHA-256 | `be5cd99bce0f52b0cdfc7e033251f2c919e14d203c8b72d394a18d7a067751f0` |
| 文件大小 | 1782542 bytes |
| 输入 / 输出 | `[1,3,320,320]` / `[1,11,8400]` |
| 类别 | player、monster、loot、portal、npc、platform、ladder |
| 结构 | 自建串行 Conv/C2f/SPPF，stride4/8/16，reg_max16 |
| 训练 | 自建 lifecycle 数据，CPU 2线程，从零训练40epoch、batch8、seed42 |
| 结果 | 同一192帧val的14个固定质量门禁中13个通过，平台召回未达标 |

全宽地板 **192/192** 检出，顶边误差 p95 约 **0.0000315px**、左右端点 p95 约 **0.472 / 0.152px**。但所有平台只有 **472/576** 检出，召回率 **0.819444**，低于预先固定的 **0.98**。精确合成像素坐标下的极小误差不是对真实画面精度的承诺。

按照训练前冻结的单候选协议，**没有为该模型评分 test，没有追加第二次候选训练，也没有运行模型推广用的场景套件、任务重新推理和资源 soak**。工程层的模式/验证器测试另行进行，不能替代该模型的质量门禁。

## 复核已保存的报告

在仓库根目录运行以下命令，只需 Python 标准库：

```bash
python scripts/check_terrain_experiment.py --preregistration reports/release-0.3.0/preregistration.json --candidate reports/release-0.3.0/terrain-model-validation.json --baseline reports/release-0.3.0/lifecycle-baseline-validation.json --output artifacts/rechecked-terrain-gates.json
```

**预期退出码为 1**，输出中的 `platform_recall.passed=false`，且 `model_promotion_granted=false`。这检查报告数值与身份一致性，不重新运行预测，也不是报告真实性签名。

## 重新实际评分

安装项目开发环境后，可用冻结参数再生数据，并用该 ONNX 在 val 上实际检测。训练依赖不用于 ONNX 部署；数据生成和下面推理均不需要 LLM API。

```bash
python -m maple_vision dataset create artifacts/reproduced-terrain-data --profile lifecycle --sessions 18 --frames 48 --seed 2026112603
python -m maple_vision vision benchmark artifacts/reproduced-terrain-data --split val --model examples/terrain-candidate/model.onnx --manifest examples/terrain-candidate/manifest.json --terrain-metrics --output artifacts/reproduced-terrain-val.json
```

所有输出必须是新路径。数据 manifest 标识原 Linux 编码字节；不同 PNG 编码器或换行规则可能产生不同容器哈希，不能把不同哈希的重生数据报告直接冒充原冻结报告。模型由其 manifest 精确绑定，原始标签、匹配 IoU 和阈值均不因失败而修改。

完整预注册、训练、结构预检、验证失败与诊断见 [执行计划](../../docs/V0_3_PLAN.md)、[实验报告](../../docs/YOLO_TERRAIN_EXPERIMENT.md) 和 [发布证据](../../reports/release-0.3.0/)。模型训练使用 Ultralytics；依赖许可和固定版本见仓库的依赖与训练文档。本模型仅含本项目自建合成内容的训练结果。
