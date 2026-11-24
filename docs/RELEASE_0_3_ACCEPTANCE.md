# v0.3 地形研究与模型对象管线验收

**v0.3工程实现已通过本地834项回归、原生与Ubuntu/Windows CI，重建安装包通过；研究候选仍拒绝推广。** 候选14项阈值通过13项，但平台召回472/576=0.819444低于0.98，依协议不评分test、不执行推广suite/reinfer/soak、不训练第二候选。首轮远端原生作业的日志缺陷和事后工程修复另行记录，不改变实验。执行步骤见 [V0_3_PLAN.md](V0_3_PLAN.md)，合同见 [v03-contract.yaml](../engineering/v03-contract.yaml)。[v0.2验收](RELEASE_0_2_ACCEPTANCE.md)保持历史内容。

## 1. 目标与当前状态

本轮为离线合成平台增加显式 `--terrain-source model` 路径：七类对象全部来自指定模型，不以fixture像素地形补回漏检。HP/MP、地图标记、对话状态仍来自fixture像素解析器。冻结门禁已经拒绝当前候选；接口本身已取得独立工程验收，不能把工具可运行解释为模型通过。

| 项目 | 当前有证据的状态 |
|---|---|
| 新生命周期数据 | 完整性、轨迹、像素、标签和覆盖验证通过；864 帧、7091 个对象 |
| 新旧数据隔离 | 解码 RGB 完全重复为 0，会话种子无交集；新数据跨 split RGB 重复为 0 |
| 唯一候选结构 | 实际 stride 4/8/16、DFL bins 0…15；地板具有 10 个完整可回归 stride16 锚点 |
| 同一新 val 的旧模型基线 | 已实测并在冻结前查看；不是新候选成绩 |
| 预注册 | 已在候选训练与候选验证前冻结 |
| 几何评分独立审查 | 早期独立21项与100组对照通过；当前26项已包含修复后834项全量 |
| 候选训练和导出 | 40epoch完成，实际1467.923792秒；ONNX导出完成 |
| 候选 val、地板几何、部署性能 | 13/14阈值通过；唯一失败为平台总体召回0.819444＜0.98 |
| 候选模型模式 13 场景、三职业实际 reinfer、短时资源运行 | 按val失败出口不执行，不能补做以替代失败 |
| 结构审查构建前分配预算 | 实际原生审查、恢复与训练/导出回归通过；复用既有离线运行域 |
| 新 test | 因val失败保持未评分，不是待补测任务 |
| 修复后完整本地与安装 | 834通过、0失败/error/跳过；同序原生155项、日志13项与重建wheel通过，重叠不相加 |
| 准确提交的远端CI | 首轮失败保留；修复实现提交的三组作业与两个隔离wheel全部通过 |
| 模型推广 | **未通过；保留失败候选，原已验收混合模型不变** |

## 2. 已冻结的身份

基线提交为 `d11e285f6ba648c34f3bcabb1c3d193f63350da1`。工程实现与测试提交 `3340d8f2453082aff002a62ae56f43123fa83c63`（tree `88b077072ba4835ebe269bbf935ae67082f8f2a7`）已通过[CI 45421893081](https://github.com/www1350/maple-vision-platform/actions/runs/45421893081)。文档和证据归档提交必须另行通过同一组检查才可合并；其准确head、CI和合并结果由 [PR #4](https://github.com/www1350/maple-vision-platform/pull/4)记录，避免在提交中自引用自身SHA。

| 对象 | SHA-256 / 标识 |
|---|---|
| [预注册文件](../reports/release-0.3.0/preregistration.json) | `751baa9fae9832f4d76ef19ceef11f5c9d1371c368d8887d1f8be4b6976a6a47` |
| 新数据 manifest | `f73d7390980fa6aedea3b3a12ada2f0a8f3658519e0ca435ca93414bbd02aac4` |
| 候选 YAML | `b4a42896b69b9a55d1fc2324e2c4b55cc12d8fdd4ac69ab26583957b00839111` |
| v0.2 已交付基线 ONNX | `7e9f61576f63be400909db6f9eb6cb89b5b935385d01a24155c7f765d2beb6e3` |
| 新候选 best.pt | `85a4b789dae6b33940f1f5cf22b7c1f6a080f595f23925cb7e589ea784fd4748` |
| 新候选 ONNX | `be5cd99bce0f52b0cdfc7e033251f2c919e14d203c8b72d394a18d7a067751f0` |

预注册时间为 **2026-11-24 21:51:15.597437（北京时间；UTC 13:51:15.597437）**。冻结前已看到旧模型新 val 基线，未看到候选结果。与 v0.2 在首候选 val 后制定工程要求的记录不同，本轮准入条件确实先于候选训练冻结；不回写旧实验的时间顺序。

## 3. 已取得的数据、结构与基线证据

新 seed=2026112603，18 会话×48 帧；train/val/test 分别为 10/4/4 会话、480/192/192 帧。验证器重放了 11080 步。旧 v0.1 的 288 张图片和 v0.2 的 864 张图片都参与输入重复检查，未见与新数据完全相同的解码 RGB，也没有会话种子交集。两张地图和机制仍相同，不能称未见地图测试。归档记录为 [dataset-coverage.json](../reports/release-0.3.0/dataset-coverage.json) 和 [dataset-isolation.json](../reports/release-0.3.0/dataset-isolation.json)。

结构预检读取旧 v0.2 val 的 768 个地形真值框；唯一候选的所有这些框都具有至少一个完整、未截断的四边训练目标锚点。全宽地板 stride16 的 20 个合格中心中有 10 个完整可回归，区别于只检查总宽度是否足够。候选 398357 参数、实际 `reg_max=16`；2 线程随机权重 Torch 前向 p50 约 22.06ms，不能替代训练后 ONNX 性能。归档记录为 [preliminary-terrain-audit.json](../reports/release-0.3.0/preliminary-terrain-audit.json)。

| 旧模型在本轮新 val：192 帧、1569 个真值 | 实测 |
|---|---:|
| macro AP50 | 0.9448299740688355 |
| micro precision / recall | 0.999271137026239 / 0.8738049713193117 |
| platform / ladder recall | 0.6649305555555556 / 0.9947916666666666 |
| player、monster、portal、npc recall | 各 1.0 |
| loot recall | 0.96 |
| TP / FP / FN | 1371 / 1 / 198 |

基线报告归档为 [lifecycle-baseline-validation.json](../reports/release-0.3.0/lifecycle-baseline-validation.json)。所有AP数值均为confidence≥0.25、IoU=0.5后的截断单IoU指标，不等同COCO mAP50–95。候选使用下节同一份val实测成绩，不能与旧版其他数据集成绩直接比较。

## 4. 唯一候选实测与拒绝推广

冻结训练已完成：一个配置、40epoch、320px、batch8、seed42、CPU2线程，从零训练实际 **1467.923792秒**。checkpoint仅由既有训练器val fitness选择。导出的ONNX输入为 `[1,3,320,320]`，输出为 `[1,11,8400]`。训练和导出记录归档为 [training.json](../reports/release-0.3.0/training.json)、[export.json](../reports/release-0.3.0/export.json)；失败候选保留在 `examples/terrain-candidate`，不覆盖 `examples/lifecycle-model`。

| 同一新val，192帧、1569个真值 | 实测 | 门禁 |
|---|---:|---|
| platform recall | **472/576 = 0.8194444444** | **失败：需≥0.98** |
| ladder recall | 190/192 = 0.9895833333 | 通过 |
| 全宽地板 recall | 192/192 = 1.0 | 通过 |
| 地板顶边绝对误差 p95 | 0.0000314776px | 通过，≤1px |
| 地板左端点绝对误差 p95 | 0.4719100952px | 通过，≤4px |
| 地板右端点绝对误差 p95 | 0.1517791748px | 通过，≤4px |
| player/monster/portal/npc recall | 各1.0 | 四项分别通过 |
| loot recall | 0.96 | 通过，≥0.95 |
| macro AP50 | 0.9646245103 | 通过，≥同val基线0.9448299741 |
| micro precision / recall | 0.9993150685 / 0.9298916507 | precision通过；recall如实报告 |
| TP / FP / FN | 1459 / 1 / 110 | 原始计数 |
| detect p50 / p95 | 22.6070945 / 24.98615355ms | p95通过，≤75ms |

因此14项冻结阈值检查中13项通过，**整体模型质量门禁仍失败**。平台FN为104，全宽地板FN为0；地板修复有价值，但不能替代所有平台的识别标准。评分归档为 [terrain-model-validation.json](../reports/release-0.3.0/terrain-model-validation.json)，条件判断为 [validation-gates.json](../reports/release-0.3.0/validation-gates.json)。门禁工具核对输入报告及身份声明，没有重新评分预测；它不能单靠内容哈希证明预注册时间、测试隔离或机器负载。具体val错误分析由 [YOLO_TERRAIN_EXPERIMENT.md](YOLO_TERRAIN_EXPERIMENT.md)维护。

| 验收层 | 固定要求 | 当前结论 |
|---|---|---|
| val 地形 | platform、ladder、全宽地板 recall 各≥0.98；地板顶边 p95≤1px、左右端点各 p95≤4px | 仅总体platform recall失败，拒绝准入 |
| val 动态对象 | 每类 recall≥同新 val 基线−0.01；macro AP50≥0.9448299740688355；micro precision≥0.98 | 上述阈值全部通过，不能覆盖地形失败 |
| 部署成本 | ORT intra-op1，3次预热后192帧detect，含预处理/解码、不含PNG I/O | p95=24.986154ms，门禁通过 |
| 候选模型对象任务 | 若val通过，需13场景及三职业seed0实际N+1帧reinfer | 因val失败不执行 |
| 候选资源与失败保留 | 若获准继续，请求60秒并记录任务与资源 | 因val失败不执行 |
| 一次 test | 原预注册条件保留，不能在失败后修改 | 因val失败保持未评分 |
| 工程发布 | 完整检查、原生训练/导出、干净wheel、准确提交的Ubuntu/Windows/native CI | 本地与准确实现提交的三组远端CI均通过 |

本次已经执行val失败出口：不评分test、不追加候选、不使用后续任务通过替代质量失败。[model-decision.json](../reports/release-0.3.0/model-decision.json)保存决定和所依据报告的原字节哈希。上述未执行项是按协议关闭的推广分支，不是当前交付的待补任务。完整性读取和私有副本复制可以接触test文件字节；新test没有模型预测或评分结果。

正式40epoch训练与val评分期间，五项冻结源码保持预注册字节；历史 `training.py` 可从提交 `ff357c8951da8e84265d31c5a8b86a48934a4f01`读取，SHA-256为 `f15fd0a5051780225b9952505c98e45cbb61c34ef4ba15efe97957fd9b09e8d6`。1730个源数据/训练副本文件一致，前后检查实际执行，见 [training-input-provenance.json](../reports/release-0.3.0/training-input-provenance.json)。实验结束后的日志修复不追溯进入该实验；首批audit之后新增的构建前预算也不能追溯归给早期报告。原预注册、模型及评分保持不变。

地板错误采用所有平台一对一匹配后分层，p95只统计匹配地板，同时必须报告遗漏与非空真值数，不能以缺失值当通过。详细定义已冻结在预注册文件和执行计划中。

## 5. 日志修复后的本地工程与安装

修复后完整JUnit包含 **834个testcase，0 failures、0 errors、0 skipped**；pytest为 **834 passed、6 warnings、110.65秒**，JUnit suite时间110.497秒。6条上游warning来自真实ONNX导出中的TorchScript弃用和静态trace提示，未当作失败或跳过。Ruff格式检查113文件、Ruff检查及mypy30源文件通过。[当前本地总报告](../reports/release-0.3.0/local-verification.json) · [修复后JUnit](../reports/release-0.3.0/local-pytest.xml) · [修复后日志](../reports/release-0.3.0/local-pytest.log)

原生结构审查、一epoch小样本训练/导出/ORT、Pillow与离线设置恢复实际执行。修复后按CI原生命令顺序 **155 passed、0失败/跳过，pytest10.83秒、JUnit10.681秒，5条warning**；日志生命周期专项 **13 passed，pytest0.45秒、JUnit0.305秒**，新增11个反例。这些组与834项全量覆盖重叠，不能相加。工程smoke不参加40epoch研究选模，不是第二研究候选。[同序原生](../reports/release-0.3.0/ci-native-order-after-fix.xml) · [日志专项](../reports/release-0.3.0/runtime-stream-fix-tests.xml)

修复前的823项全量、106.26秒结果另存 [before-runtime-fix总报告](../reports/release-0.3.0/local-verification-before-runtime-fix.json) 和 [原JUnit](../reports/release-0.3.0/local-pytest-before-runtime-fix.xml)，旧48项原生审查组合也保留为历史；它们不能覆盖后来修改的`training.py`。

初次全量为 **821 passed、2 failed**：两个真实审查用例错误地将两张val图像的地板总数预期为1。修正后先独立核对两帧分别来自两图、每份标签恰有一块全宽地板，再严格断言总数为2；没有改生产模型、降低质量阈值或跳过原生检查。初次结果保留在 [initial JUnit](../reports/release-0.3.0/local-pytest-initial.xml) 与 [initial log](../reports/release-0.3.0/local-pytest-initial.log)。

修复后重建wheel验证 **通过，20.961秒**，SHA-256为 `f0b7603f2cd2589127d27bd3c05ae310973c6b45e7b4c07a8feb08d3235c719f`；30个包内Python文件和两份YAML与当前源码逐字节一致。干净环境未装Torch/ORT，确认配置打包、自定义detector模型地形路由不调用fixture对象检测，以及P0、知识、数据、素材、pixel任务与326帧实际reinfer。它不使用失败候选做推广任务。[当前安装报告](../reports/release-0.3.0/wheel-smoke.json)；旧21.709秒、SHA以`fe38f69f`开头的安装报告另存 [修复前wheel](../reports/release-0.3.0/wheel-smoke-before-runtime-fix.json)。

以上本地结果已由修复后准确实现提交的三组远端CI补充验证。远端结果详见下一节；模型拒绝推广结论保持独立。

## 6. 首次远端CI失败、修复与实际通过

首轮实现提交为 `ff357c8951da8e84265d31c5a8b86a48934a4f01`，tree `10e5339a03a2d9fa0dde0bb416821899038587d5`。[CI 45417925554](https://github.com/www1350/maple-vision-platform/actions/runs/45417925554)整体失败，事实如下。

| 首轮实际作业 | 测试结果 | 隔离安装 |
|---|---|---|
| Ubuntu Python3.12 | 816通过、0失败、7跳过，85.93秒 | 通过，11.654秒 |
| Windows Python3.12 | 812通过、0失败、11跳过，107.03秒 | 通过，19.941秒 |
| Ubuntu原生训练/导出 | 152通过、3失败、0跳过，6.25秒 | 由前两项覆盖 |

错误来自 `training._offline_runtime` 重绑logger流时，`StreamHandler.setStream`先flush前次测试已经关闭的capture stream。按CI同序本地复现为152通过/3失败、10.28秒。这是运行时生命周期代码缺陷，与此前本地两项“地板总数预期错误”不同。[首轮远端记录](../reports/release-0.3.0/remote-ci.json) · [修复前同序复现](../reports/release-0.3.0/local-native-order-before-fix.xml)

实验结束后，仅`training.py`与`test_offline_runtime.py`进行了日志生命周期工程修复和反例回归，并完成独立审查。修复后`training.py` SHA-256为 `c1f6be9c6d5b9a028e089fa59a892adaf8c2e60c2e793d52be0823767fb2c739`，与正式实验的历史f15…源码分开记录；没有修改实验产物、质量门禁、模型或追加40epoch候选。[runtime-stream-fix.json](../reports/release-0.3.0/runtime-stream-fix.json)

修复后实现提交 **`3340d8f2453082aff002a62ae56f43123fa83c63`**、tree **`88b077072ba4835ebe269bbf935ae67082f8f2a7`** 的 [CI 45421893081](https://github.com/www1350/maple-vision-platform/actions/runs/45421893081) 三组全部成功。

| 修复版实际作业 | Python | 测试结果 | 测试用时 | 隔离安装 |
|---|---|---|---:|---:|
| Ubuntu | 3.12.16 | 827通过、0失败、7跳过 | 110.69秒 | 通过，13.129秒 |
| Windows | 3.12.10 | 823通过、0失败、11跳过 | 136.15秒 | 通过，23.301秒 |
| Ubuntu原生训练/导出 | 3.12.16 | 155通过、0失败、0跳过，5条依赖warning | 5.68秒 | 由前两项覆盖 |

基础作业的跳过来自未安装或未启用的可选原生能力；实际训练、导出、视频、OCR和结构审查由独立原生作业执行。三组覆盖重叠，不应相加。两边Ruff、mypy、依赖检查、构建与隔离安装步骤也均成功。

[完整远端记录](../reports/release-0.3.0/remote-ci.json)保留首轮失败和修复后成功；[Ubuntu wheel](../reports/release-0.3.0/remote-wheel-ubuntu.json) 与 [Windows wheel](../reports/release-0.3.0/remote-wheel-windows.json)保存实际环境、SHA和运行报告。本文的实现成绩绑定上述准确提交。后续文档/证据归档head仍必须运行自身CI再合并，最终准确head与成功run记录在 [PR #4](https://github.com/www1350/maple-vision-platform/pull/4)；这些结果不授予失败模型推广资格。

## 7. 几何评分的独立审查

`terrain_metrics.py` 已完成当前工作树的只读审查。它先匹配全部平台，再按全宽地板真值分层；漏检继续计入 FN，不伪造成零误差。原图顶边和左右端点独立计量，p95采用线性插值。半宽框与过厚框虽然可达到 IoU0.5，但会暴露端点或顶边误差；无地板或全漏检时几何指标保持 `null`，须由推广判定明确拒绝。

早期独立审查时专项为 **21 passed，0.64秒**，另100组小例与一般评估器的平台/梯子TP/FP/FN/recall一致。后来增加身份及manifest替换回归，当前26项已包含在修复后834项全量通过中。早期审查没有运行模型或接触test预测，远端准确提交结果仍另记。

`vision benchmark --terrain-metrics` 产生带 `terrain_geometry` 的 schema3 报告，使用原检测批次的预测，不重复推理。省略参数时保留旧报告格式和指标。结构入口 `model audit-terrain` 只接受 train/val，其随机权重结构与成本报告不充当训练后识别质量。

## 8. 结论边界

模型对象模式没有地形像素回退，HUD仍是专门的合成像素解析。当前策略会使用平台框，但没有梯子检测驱动的攀爬分支；梯子检测通过也不能升级为梯子导航通过。

新 test 的 192 帧来自 4 个会话，两内置地图在所有 split 出现。无完全重复 RGB 不是独立真实场景的证明；对象 recall、固定场景通过率、反事实动作验证和真实任务可靠率不能互相替代。

正式客户端控制、当前真实游戏数据、网络小册子适配器、真实域识别、未见地图泛化、24小时墙钟和完整v8均未取得本轮验收。工具工程实现已有准确提交与实际CI成功证据，最终归档检查和合并由PR #4记录；模型、研究发现和完整v8未完成事项继续分别报告。
