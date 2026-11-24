# v0.2 生命周期数据与模型实验

本次交付一个新的可直接运行的 ONNX 检查点，保持旧 v0.1 模型和历史报告不变。**新模型获得的结论是合成场景中的混合感知可用；纯 YOLO 地形、真实域及 24 小时墙钟验收仍未通过。** 全部实际 JSON、哈希与 JUnit 位于 [reports/release-0.2.0](../reports/release-0.2.0/)。

## 1. 实验问题与冻结输入

旧数据没有充分覆盖左侧出生、清场、拾取、对话和切图后的状态。v0.2 先修复数据覆盖，用独立动作轨迹核对事实，再检查新数据是否改善同一模型结构的表现。

| 冻结项 | 实际值 |
|---|---|
| 生成器 / schema | `lifecycle-v1` / 2 |
| 参数 | sessions=18，frames_per_session=48，seed=20261125 |
| 总量 | 864 帧，7098 个可见标注 |
| train / val / test | 480 / 192 / 192 帧，10 / 4 / 4 个独立会话 |
| 内容 | 三职业、两地图、六阶段、五种横向位置分组 |
| 数据 manifest SHA-256 | `670f6dae181f0f4b3107ba6a47cb733f19cfbc216c4e8e8c23f96b756601b3ff` |

训练生成器可以读取模拟器真值来安排动作和自动标注；运行模型与策略只能收到画面及观测。生成的轨迹能够完成目标，不等于视觉策略已经成功。校验器重放 11164 个动作步骤，核对像素、可见标签和阶段事实；只重写 coverage 并重新计算哈希无法伪造覆盖。[数据与覆盖报告](../reports/release-0.2.0/dataset-coverage.json)

会话隔离并不等于地图隔离：两张地图仍出现在所有分区。192 张 test 图片只来自四个会话，相关帧不能当作 192 个独立真实场景。这里报告点估计和样本量，不据此推断真实域成功率。

## 2. 训练、选择与时间顺序

实际训练配置为 `configs/tiny-yolo.yaml`，320px、40 epoch、batch 8、seed 42；CPU 两线程、无预训练权重，实际用时 **1067.895699 秒**。只训练了这一份新候选；旧模型作为同一新 val 上的比较基线。[训练记录](../reports/release-0.2.0/training.json)

训练时使用私有输入副本；数据 manifest 与复制后的模型 YAML 均与冻结输入一致。模型配置 SHA-256 为 `122a5e4040cdf15f5acdd692974c55b1b3260794601b1f8d5756595866b81bf6`，`best.pt` SHA-256 为 `2f5fec25a7c5bf94e80d6911d57b59bdf2842c1cbe1f61eb86845e76374cee02`。该长训练进程启动后才加入训练前后指纹不变检查，因此本次输入一致性由独立的训练后检查确认；不倒推声称进程执行了后来加入的代码。当前实现的防护另有原生与篡改回归。[副本来源检查](../reports/release-0.2.0/training-input-provenance.json)

候选实际导出 ONNX 后，先在 val 上比较七类检测，再执行三职业×三种子和四个故障场景。工程准入要求是：macro AP50、平台 recall 和五个动态类的 recall 不低于旧模型，13 个场景均符合预期。**这些门禁在查看首候选 val 后记录，属于工程选择，不是预注册实验。**

2026-11-24 20:06:01（Asia/Shanghai）固定首候选，随后才对 test 评估一次；没有根据 test 改参数或继续搜索。三职业逐帧重新推理和短时资源 soak 完成后，接受其用于明确的混合管线。[选择记录](../reports/release-0.2.0/model-selection.json) · [推广范围](../reports/release-0.2.0/model-promotion.json)

最终 ONNX SHA-256 为 `7e9f61576f63be400909db6f9eb6cb89b5b935385d01a24155c7f765d2beb6e3`，文件 841472 字节；输入 `[1,3,320,320]`，输出 `[1,11,8000]`，opset 17。用真实 ONNX Runtime 检查了输出形状和有限数值；部署包不包含训练所需 Torch。[导出记录](../reports/release-0.2.0/export.json)

## 3. 同一验证集的比较

以下均来自同一份 192 帧 val、1579 个目标；confidence≥0.25、IoU=0.5，类别匹配和 NMS 设置相同。不能拿旧版本不同 test 的数字替代这一基线。

| 指标 | 旧 v0.1 模型 | 新生命周期模型 |
|---|---:|---:|
| TP / FP / FN | 1248 / 41 / 331 | 1380 / 2 / 199 |
| micro precision | 0.968192 | 0.998553 |
| micro recall | 0.790374 | 0.873971 |
| 截断 macro AP50 | 0.900697 | 0.946252 |
| 平台 recall | 0.467014 | 0.661458 |
| 玩家 / 怪物 recall | 1.000000 / 1.000000 | 1.000000 / 1.000000 |
| 掉落 recall | 0.971154 | 0.990385 |
| 门 / NPC recall | 0.994792 / 0.965714 | 1.000000 / 0.994286 |

新数据分布、数据数量、数据生成 seed 和训练轮数与历史实验不同；训练 seed 两者均为 42。结果支持这次工程候选选择，但不能把全部提升归因于某一个变量。[旧模型新 val](../reports/release-0.2.0/prior-model-validation.json) · [新模型 val](../reports/release-0.2.0/lifecycle-model-validation.json)

## 4. 选择后的保留测试集

192 帧、1572 个标注：**TP 1372、FP 3、FN 200**，micro precision **0.997818**、recall **0.872774**，macro AP50 **0.944822**。AP 是置信度截断后的 101 点插值单 IoU 指标，报告中的 `map50_95` 明确为 `null`；它不是完整 COCO mAP50–95。

| 类别 | GT | TP | FP | FN | recall |
|---|---:|---:|---:|---:|---:|
| player | 192 | 192 | 0 | 0 | 1.000000 |
| monster | 142 | 142 | 0 | 0 | 1.000000 |
| loot | 104 | 101 | 0 | 3 | 0.971154 |
| portal | 192 | 192 | 0 | 0 | 1.000000 |
| npc | 174 | 174 | 0 | 0 | 1.000000 |
| platform | 576 | 381 | 0 | 195 | 0.661458 |
| ladder | 192 | 190 | 3 | 2 | 0.989583 |

分组结果使平均值之外的边界可见：

| 阶段 | 图片数 | macro AP50 |
|---|---:|---:|
| spawn | 12 | 0.943894 |
| combat | 44 | 0.933522 |
| loot | 44 | 0.943894 |
| cleared | 68 | 0.926616 |
| dialogue | 16 | 0.915842 |
| transition | 8 | 0.943894 |

左右边缘组分别只有 24、10 帧；left 组为 49 帧，AP50 0.893741。组内有效真值类别可能不同，各组 AP 独立重排计算，不能相加或用简单平均恢复总体分数；覆盖标签还互相重叠。完整按地图、地图×阶段、位置和标签分组的 TP/FP/FN/AP 已保留。[完整 test 报告](../reports/release-0.2.0/lifecycle-model-test.json)

本机 `detect(frame)` 计时为 p50 **15.44ms**、p95 **24.47ms**，192 次调用、排除三次预热，包含预处理与解码但不包含 PNG I/O。它不是帧到动作的总延迟或专用硬件基准，也不是对其他机器的速度承诺。

## 5. 为什么整条地板仍然漏检

独立审查直接读取导出的 ONNX 和本机安装的上游实现，并执行实际正样本分配方法。模型 DFL 权重确为 0..7，stride 常量确为 4/8；每侧距离是 0..7 的 softmax 加权值，因此最大框宽为：

$$
w_{\max}=2(8-1)\times8=112\text{ px}
$$

画面横向缩放为 1。对宽 320、高 4 的整条地板，即使其他位置完全准确，单框 IoU 仍不超过 `112/320=0.35`。val 中有 192 个这样的地板框，必然无法达到 IoU50，解释了 195 个平台漏检中的至少 192 个；剩余三个不能用这条结构证明归因。

没有盲目训练第二个候选。追加 stride32 的头虽然能扩大理论框宽，但实际 `TaskAlignedAssigner` 对本场景扩展后的地板区域为 y=230..238，最近的 stride32 中心在 y=240，**eligible positive anchors 为 0**。提高 `reg_max` 又会扩大高分辨率回归卷积，需要单独衡量 CPU 代价与薄框定位，不能只凭数学范围宣布成功。[结构与锚点实测](../reports/release-0.2.0/architecture-diagnosis.json)

因此保留现有透明合同：YOLO 检测玩家、怪物、掉落、NPC 和门；像素方法读取自建地形、梯子、HP/MP、地图和对话标记。动态对象漏检不使用颜色回退；纯 YOLO 地形仍为未验收。后续地形模型须先验证表达范围、正样本分配、几何误差和成本。

## 6. 闭环、真正复验与短时资源

新模型通过三职业×三种子九个正常任务；短暂黑屏和重复帧恢复，持续黑屏停止，输入冻结有界失败，**13 个场景全部符合预期**。通过故障检查不等于每个故障任务都完成。[场景套件](../reports/release-0.2.0/model-scenario-suite.json)

| 职业，seed 0 | 执行步数 | 实际重新感知帧数 | 独立目标与逐帧复验 |
|---|---:|---:|---|
| assassin | 327 | 328 | 通过 |
| dragon_knight | 325 | 326 | 通过 |
| cleric | 317 | 318 | 通过 |

每次重新执行检测器、监督和策略，包括最后的终止观测；模型已加载字节哈希、完整 manifest、阈值匹配，运行身份一致，原证据文件未改变。它不同于仅重放记录动作的 `lab replay`。[任务与重推理](../reports/release-0.2.0/model-missions-and-reinference.json)

soak 请求 60 秒，实际 **61.510525 秒**，完成 10 轮、累计模拟 160 秒；63 个资源样本全部保留，无采样错误。进程 CPU 累计约 61.389 秒，最高采样当前 RSS 为 **94470144 bytes，即 90.09375 MiB**。CPU 百分比按单逻辑核口径，略高于 100% 不表示整机占用率；当前 RSS 与进程生命周期峰值分开记录。采样包括证据捕获 I/O，短峰值可能漏采，一分钟结果不能证明不存在长期泄漏。[原始资源与样本](../reports/release-0.2.0/model-resource-soak.json)

原始失败轮次只执行一次，失败保留同一份轨迹和终止上下文，成功清理本次 staging；测试不再通过重跑来制造“失败证据”。24 小时门禁继续为 `not_completed`，真实域和未见地图也没有因此通过。

## 7. 从零复现

先按 [TRAINING.md](TRAINING.md) 安装固定训练依赖。以下使用新输出目录；自建图像可确定性再生，不需要下载官方素材。

```bash
python -m maple_vision dataset create artifacts/reproduce-v02-data --profile lifecycle --sessions 18 --frames 48 --seed 20261125
python -m maple_vision dataset check artifacts/reproduce-v02-data
python -m maple_vision model train artifacts/reproduce-v02-data --output artifacts/reproduce-v02-train --config configs/tiny-yolo.yaml --epochs 40 --image-size 320 --batch 8 --seed 42
python -m maple_vision model export artifacts/reproduce-v02-train/run/weights/best.pt --output artifacts/reproduce-v02-export --image-size 320
python -m maple_vision vision benchmark artifacts/reproduce-v02-data --output artifacts/reproduce-v02-val.json --split val --model artifacts/reproduce-v02-export/model.onnx --manifest artifacts/reproduce-v02-export/manifest.json
```

若只是复核交付模型，可直接使用 `examples/lifecycle-model/`，无需重新训练。先记录选择，之后再运行 `vision benchmark --split test`；不以 test 反馈反复选模型。通过 `lab run`、`lab reinfer`、`lab suite`、`lab soak` 可分别复现任务、逐帧一致性、故障和资源记录。[完整执行方案](FINAL_PLAN.md)

固定种子、依赖和数据能约束实验条件，但不同 CPU、运行库和调度环境的训练浮点结果或时序不保证逐字节相同。本文数据 manifest 标识本次 Linux 冻结文件；其他平台的 PNG 编码和文本换行可能导致容器字节哈希不同，数据工具仍检查本地每个文件的完整性，历史兼容另有严格内容指纹验证。交付 ONNX 则由 manifest 哈希精确标识。运行报告与本机训练日志保留完整数值，不能只看四舍五入的文档表格。
