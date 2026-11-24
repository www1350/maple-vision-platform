# v0.3 有界地形研究与模型对象管线

本轮检验一个明确问题：**在保留现有离线模拟器和训练数据合同的前提下，修复模型对宽而薄的平台的表达限制，并让七类对象全部来自指定 ONNX 模型。** 血量、魔量、地图标记和对话状态仍由 fixture 像素解析器读取。因此本轮称“模型对象管线”，不称所有视觉信息均由 YOLO 取得。

当前状态：唯一候选完成训练和val，**模型准入失败**。平台召回472/576=0.819444低于0.98，14项通过13项也不推广，不评分test、不执行候选推广任务、不训练第二候选。工具首轮远端native暴露日志流问题；事后窄修复后完整本地834项、同序原生155项、日志13项及重建wheel通过，修复版远端CI仍待结果。实际结果见 [RELEASE_0_3_ACCEPTANCE.md](RELEASE_0_3_ACCEPTANCE.md)，合同见 [v03-contract.yaml](../engineering/v03-contract.yaml)。v0.2与原实验证据保持历史内容。

## 1. 预注册与有限预算

[preregistration.json](../reports/release-0.3.0/preregistration.json) 在 **2026-11-24 13:51:15.597437 UTC（上海 21:51:15.597437）**冻结，早于候选训练及候选验证。文件 SHA-256 为 `751baa9fae9832f4d76ef19ceef11f5c9d1371c368d8887d1f8be4b6976a6a47`。冻结前已查看旧模型在本轮新 val 上的成绩，也用旧 v0.2 val 的几何进行了结构审查；没有查看新候选结果。哈希用于核对内容完整性，不是签名或审批凭据。

本轮只训练 **一个配置、一次 40 epoch**，从零开始，不读取预训练权重。`best.pt` 由既有训练器的验证集 fitness 选取，不增加第二候选、不追加 epoch，也不在看到结果后改变数据、阈值、NMS 或评分定义。通过门禁才推广；失败同样保留配置、训练过程和原始评估。结构预检中的假设 `reg_max=12` 仅用于计算可回归锚点数，没有训练或评价其模型质量。

| 固定训练输入 | 值 |
|---|---|
| 配置 | `configs/terrain-yolo.yaml`，包内资源副本相同 |
| 配置 SHA-256 | `b4a42896b69b9a55d1fc2324e2c4b55cc12d8fdd4ac69ab26583957b00839111` |
| 实际检测头 | stride 4/8/16，`reg_max=16`，DFL bins 0…15 |
| 参数量 | 398357；旧配置为 165798 |
| 分辨率、batch、epoch、seed | 320、8、40、42 |
| 训练运行资源 | CPU，PyTorch 2 线程，DataLoader workers=0 |
| 图像增强与优化 | 使用冻结的 `training.py` 选项；不另行搜索增强或学习率 |

训练、数据、渲染、模拟器和类型定义的源文件哈希也在预注册文件内。训练使用经过校验的私有输入副本，结束后重新校验实际副本和配置。该副本包含所有 split：**完整性检查和复制可以读取 test 字节，但 test 不得进入训练、候选评分或选模。** 不能将这一流程表述成“test 文件从未被读取”。详细本地训练约束见 [TRAINING.md](TRAINING.md)。

正式40epoch训练与val评分期间，五项冻结源码保持预注册字节，1730个输入副本文件一致，证据为 [training-input-provenance.json](../reports/release-0.3.0/training-input-provenance.json)。历史 `training.py` 可从 `ff357c8951da8e84265d31c5a8b86a48934a4f01`读取，其SHA-256仍为 `f15fd0a5051780225b9952505c98e45cbb61c34ef4ba15efe97957fd9b09e8d6`。实验结束后，远端原生CI发现logger引用已关闭流；窄范围日志生命周期修复不追溯进入实验，不修改候选、数据或搜索预算。结构审查新增分配预算同样与训练入口分开，不能追溯计入早期审查报告。

## 2. 结构预检为什么必要

v0.2 实际模型的 `reg_max=8`、stride 4/8 使单框宽度上限只有 `2 × 7 × 8 = 112` 像素，对 320 像素全宽地板的单框 IoU 上限是 0.35。这个限制无法通过增加训练次数消除。仅追加 stride32 也不足：已安装分配器对 4 像素薄地板给该层的合格正样本中心数为零。[v0.2 实际结构诊断](YOLO_LIFECYCLE_EXPERIMENT.md)

本轮分别核对三个条件：框的理论支持范围、分配器允许的锚点，以及每一边训练目标是否会被 DFL 截断。当前安装版本把目标四边距离截到 `(reg_max−1−0.01) × stride`；总宽度足够不代表靠近边缘的锚点能表达完整框。

唯一候选在 stride16 上具有 480 像素的理论总宽度支持；全宽地板有 **20 个合格锚点，其中 10 个的完整四边训练目标不被截断**。预检使用旧 v0.2 val 的 768 个可见地形框，576 个平台和 192 个梯子全部至少有一个完整可回归锚点。没有使用新 test 的模型结果来选择结构。证据归档为 [preliminary-terrain-audit.json](../reports/release-0.3.0/preliminary-terrain-audit.json)。

随机权重的本地 Torch 前向预检，候选 p50 约 22.06ms、旧配置约 17.47ms，均为 2 线程。它只用于检查结构和计算成本，**不代表训练后 ONNX 速度或检测质量**。部署性能必须使用真正导出的模型和实际适配器重新测量。

## 3. 新会话数据与留出集

新数据使用 `lifecycle-v1`、schema v2，seed **2026112603**。数据 manifest SHA-256 为 `f73d7390980fa6aedea3b3a12ada2f0a8f3658519e0ca435ca93414bbd02aac4`：18 个会话、每会话 48 帧，共 **864 帧、7091 个可见对象**；实际动作、画面和标签验证共重放 11080 步。[生命周期数据合同](DATASET_LIFECYCLE.md)

| split | 会话 | 帧 | 用途 |
|---|---:|---:|---|
| train | 10 | 480 | 梯度训练 |
| val | 4 | 192 | 训练器选择 best；旧模型与候选的同集比较；推广前门禁 |
| test | 4 | 192 | 候选固定后的一次独立评分；不用于调参 |

三职业、六阶段、五个玩家位置区间和两张内置地图都保持现有定义。与 v0.1 原始 288 张图片和 v0.2 原始 864 张图片逐一比较，解码后 RGB 完全重复数均为 0，会话种子交集为空；新数据内部跨 split RGB 重复数也为 0。隔离证据归档为 [dataset-isolation.json](../reports/release-0.3.0/dataset-isolation.json)，其目的只是输入身份与重复检查，没有运行 test 模型预测或评分。

两张地图仍是 `meadow` 与 `cavern`，同时出现在所有 split；布局和机制相同。新种子与无重复图片提供新的会话留出，**不能证明未见地图或真实域泛化**。192 帧来自 4 个相关会话，不是 192 个独立世界；分组指标是描述性工程证据，不据此给出任务可靠率的统计保证。

## 4. 同一 val 的固定基线

基线为已交付的 `examples/lifecycle-model/`，ONNX SHA-256 `7e9f61576f63be400909db6f9eb6cb89b5b935385d01a24155c7f765d2beb6e3`。只比较本轮同一份新 val，不将旧数据的成绩直接充当新实验基线。

| 新 val 的实测指标 | v0.2 模型 |
|---|---:|
| 截断 macro AP50 | 0.9448299740688355 |
| micro precision | 0.999271137026239 |
| micro recall | 0.8738049713193117 |
| platform recall | 0.6649305555555556 |
| ladder recall | 0.9947916666666666 |
| player / monster / portal / npc recall | 各 1.0 |
| loot recall | 0.96 |
| TP / FP / FN | 1371 / 1 / 198 |

基线报告归档为 [lifecycle-baseline-validation.json](../reports/release-0.3.0/lifecycle-baseline-validation.json)。AP 是置信度≥0.25 后的 101 点插值、IoU=0.5 指标，只有单 IoU，不是 COCO mAP50–95。

## 5. 冻结的模型推广门禁

所有评估固定 detector confidence=0.25、NMS IoU=0.45、真值匹配 IoU=0.5。数值门禁来源是预注册文件，不因候选成绩改变。

| 指标 | val 门禁 | 一次 test 门禁 |
|---|---:|---:|
| platform recall | ≥0.98 | ≥0.98 |
| ladder recall | ≥0.98 | ≥0.98 |
| 全宽地板 recall | ≥0.98 | ≥0.98 |
| 已匹配全宽地板顶边绝对误差 p95 | ≤1px | ≤1px |
| 已匹配全宽地板左、右端点各自绝对误差 p95 | 各≤4px | 各≤4px |
| 每个动态类别 recall | 同新 val 基线减 0.01 | 各≥0.95 |
| macro AP50 | ≥0.9448299740688355 | ≥0.97 |
| micro precision | ≥0.98 | ≥0.98 |
| 实际 ONNX detect p95 | ≤75ms | ≤75ms |

动态类别为 player、monster、loot、portal、npc。val 规则按已冻结基线展开为 player/monster/portal/npc 各≥0.99、loot≥0.95；没有使用基线 test 来确定这些阈值。

地板门禁必须单独计算：val 共 576 个平台，其中 192 个是全宽地板。总平台 recall 达到 0.98 仍可能把 11 次遗漏全部集中到地板，地板 recall 仅约 0.943。IoU=0.5 同样不能保证可用边界：一段略宽于半幅的框，或包含真值的 8 像素厚框，都可能达到阈值而偏离策略所需支撑面。因此另行报告顶边和两个水平端点误差。

几何评分先在同图、同类的**全部平台**间按置信度做一对一 IoU 匹配，再按全宽真值分层。全宽真值定义为原图坐标左边界=0、右边界=image_width，绝对容差 1e-6；不猜测遮挡后的隐藏几何。误差只在已匹配地板上计算，各端点分开报告，p95 使用线性插值；遗漏仍进入 recall。地板真值必须非空，无法定义的几何指标不能当作通过。`vision benchmark --terrain-metrics` 将同一批模型预测交给此评分器，在 schema3 报告中追加 `terrain_geometry`，不会为几何指标再次推理；省略选项时保留旧报告合同。

性能测量固定使用 `OnnxDetector` 的 **ORT intra-op 1 线程**；训练用 Torch 2 线程是另一项配置。每个 split 排除 3 次预热，记录 192 次 `detect(frame)`，包含预处理、推理、解码，排除 PNG I/O。计时阶段不并行训练，并记录实际主机与依赖。75ms 是当前主机的工程预算，不是整个控制循环的实时承诺。

### 本次实际结果：13/14阈值通过，模型不推广

唯一训练实际完成40epoch，用时 **1467.923792秒**；best.pt SHA-256为 `85a4b789dae6b33940f1f5cf22b7c1f6a080f595f23925cb7e589ea784fd4748`。导出ONNX SHA-256为 `be5cd99bce0f52b0cdfc7e033251f2c919e14d203c8b72d394a18d7a067751f0`，输入 `[1,3,320,320]`、输出 `[1,11,8400]`。失败候选保留位置是 `examples/terrain-candidate`，已验收的 `examples/lifecycle-model` 混合管线保持不变。

| 同一新val实测 | 候选结果 | 冻结结论 |
|---|---:|---|
| platform recall | 472/576 = 0.8194444444 | **未达到0.98** |
| ladder recall | 190/192 = 0.9895833333 | 通过 |
| 全宽地板recall | 192/192 = 1.0 | 通过 |
| 地板顶边p95绝对误差 | 0.0000314776px | 通过 |
| 地板左 / 右端点p95绝对误差 | 0.4719100952 / 0.1517791748px | 分别通过 |
| player/monster/portal/npc recall | 各1.0 | 分别通过 |
| loot recall | 0.96 | 通过 |
| macro AP50 | 0.9646245103 | 通过同val基线门禁 |
| micro precision / recall | 0.9993150685 / 0.9298916507 | precision通过；recall如实报告 |
| TP / FP / FN | 1459 / 1 / 110 | 原始计数 |
| detect p50 / p95 | 22.6070945 / 24.98615355ms | p95通过 |

平台FN共104，而全宽地板FN为0，说明“整幅地板已经识别”没有解决所有平台遗漏。原因分析只使用val，具体研究记录由 [YOLO_TERRAIN_EXPERIMENT.md](YOLO_TERRAIN_EXPERIMENT.md)维护。原始评分与门禁归档为 [terrain-model-validation.json](../reports/release-0.3.0/terrain-model-validation.json) 和 [validation-gates.json](../reports/release-0.3.0/validation-gates.json)；14项是冻结阈值检查，不是14个独立实验或总体可靠率样本。

这已触发预注册失败出口。[model-decision.json](../reports/release-0.3.0/model-decision.json)明确记录新test未评分、候选推广场景不执行、仅一次训练。工具实现的工程验收继续；未获模型推广不妨碍交付可复核的结构/几何工具和失败证据。

## 6. 七类对象来源与独立复验

显式传入 `--terrain-source model` 才启用 `onnx-all-objects+fixture-hud-v1`：player、monster、loot、portal、npc、platform、ladder 都只来自模型。缺失的平台或梯子不能从像素解析器补回。HP/MP、地图标记和对话状态仍来自 fixture HUD。默认 `fixture` 模式和 v0.2 的历史混合证据保持兼容。[模型运行与复验合同](MODEL_VALIDATION.md)

预注册为通过val的候选要求：该模式下三职业×三个种子，加四个故障场景，共 **13 个预期结果全部符合约定**；分别保存三职业 seed 0 的实际任务，并重新运行每一帧感知、监督和策略，包括终止观测。**本次候选因val失败未进入这一分支。** 工具本身仍须保证suite、soak、运行包和reinfer绑定同一模型字节、manifest、阈值与terrain source，错误来源在首次推理前拒绝；重新播放记录动作不能代替实际重新感知。

策略目前会利用平台顶边判断跳标及离开高平台，但没有根据 ladder 检测结果执行攀爬的决策分支。本轮梯子准入是离线检测质量；13 场景通过也不据此宣称梯子导航已交付。

预注册的短时资源运行请求为60秒，需记录实际墙钟、任务结果、CPU/RSS来源、单位和采样情况，并保留原始失败轮次；本次失败候选未执行此项。它即便通过也不能代替24小时墙钟验收。模型场景与test数据评分是不同证据，当前两者均没有候选通过成绩。

## 7. 执行命令与失败出口

以下命令保留冻结配置的独立复现方法；**本轮已完成唯一训练，继续当前交付时不得另起候选或对未评分test运行模型**。已有实验目录不得覆盖。研究复现也必须明确区别于这次冻结单候选的执行次数，不能把重跑结果替换进原报告。

```bash
maple-vision dataset create artifacts/v03-repeat/dataset --profile lifecycle --sessions 18 --frames 48 --seed 2026112603
maple-vision model audit-terrain artifacts/v03-repeat/dataset --config configs/terrain-yolo.yaml --output artifacts/v03-repeat/structure.json --split train --image-size 320
maple-vision model train artifacts/v03-repeat/dataset --output artifacts/v03-repeat/training --config configs/terrain-yolo.yaml --epochs 40 --image-size 320 --batch 8 --seed 42
maple-vision model export artifacts/v03-repeat/training/run/weights/best.pt --output artifacts/v03-repeat/export --image-size 320
maple-vision vision benchmark artifacts/v03-repeat/dataset --split val --model artifacts/v03-repeat/export/model.onnx --manifest artifacts/v03-repeat/export/manifest.json --terrain-metrics --output artifacts/v03-repeat/validation.json
```

以下是通过模型门禁后才允许执行的推广接口示例；本次候选失败，因此这些命令不属于本轮剩余执行任务：

```bash
maple-vision lab suite --output artifacts/v03-repeat/suite --seeds 3 --model artifacts/v03-repeat/export/model.onnx --manifest artifacts/v03-repeat/export/manifest.json --terrain-source model
maple-vision lab run --output artifacts/v03-repeat/assassin --job assassin --seed 0 --model artifacts/v03-repeat/export/model.onnx --manifest artifacts/v03-repeat/export/manifest.json --terrain-source model
maple-vision lab reinfer artifacts/v03-repeat/assassin --model artifacts/v03-repeat/export/model.onnx --manifest artifacts/v03-repeat/export/manifest.json --terrain-source model
maple-vision lab soak --output artifacts/v03-repeat/soak --seconds 60 --model artifacts/v03-repeat/export/model.onnx --manifest artifacts/v03-repeat/export/manifest.json --terrain-source model
```

另两职业同样运行及实际复验。`model audit-terrain` 只接受 train/val，并在随机权重下检查实际检测头及锚点，不能当作候选质量评分。上例审查新复现数据的 train；正式冻结前的历史审查使用的是旧 v0.2 val，两个来源在报告中分别保留。`vision benchmark` 直接评估检测器输出，不接收 terrain source 参数，也不会使用混合感知补全地形；它的默认 split 是 test，候选比较必须明确写 `--split val`。

本轮已执行val失败出口；原条件式test门禁保留而没有执行。日志工程修复后完整本地为 **834 passed、0 failed/error/skipped，110.65秒**，JUnit110.497秒，6条上游warning；重建wheel20.961秒通过。旧823项/旧wheel另存修复前历史，不能覆盖新源码。当前剩余是修复后准确提交的Ubuntu/Windows/native CI与归档，模型推广仍未通过。[本地证据](../reports/release-0.3.0/local-verification.json)

## 8. 本轮以外

真实录像与人工标注、当前游戏规则、网络小册子适配器、外部地图/职业包、跟踪与 belief state、任务 DSL、真实 HUD/OCR 校准、梯子导航和 24 小时运行仍不在本轮已验收结论内。v8 全计划继续保持未完成。所有动作只发生在独立模拟器，既没有正式客户端控制，也不产生反作弊或账号安全结论。
