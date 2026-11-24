# v0.3 地形研究与模型对象管线验收

**工具本地完整回归823项全部通过，隔离安装通过；唯一研究候选仍因平台召回率未达标而拒绝推广。** 14项预注册val阈值中13项通过，但platform recall=472/576=0.819444低于0.98。按协议不评分test、不执行候选推广suite/reinfer/soak、不训练第二候选。准确提交的远端CI仍待完成。执行步骤见 [V0_3_PLAN.md](V0_3_PLAN.md)，机器可读合同见 [v03-contract.yaml](../engineering/v03-contract.yaml)。[v0.2 发布验收](RELEASE_0_2_ACCEPTANCE.md)保持为历史记录。

## 1. 目标与当前状态

本轮为离线合成平台增加显式 `--terrain-source model` 路径：七类对象全部来自指定模型，不以fixture像素地形补回漏检。HP/MP、地图标记、对话状态仍来自fixture像素解析器。冻结门禁已经拒绝当前候选；接口本身仍可取得独立工程验收，不能把工具可运行解释为模型通过。

| 项目 | 当前有证据的状态 |
|---|---|
| 新生命周期数据 | 完整性、轨迹、像素、标签和覆盖验证通过；864 帧、7091 个对象 |
| 新旧数据隔离 | 解码 RGB 完全重复为 0，会话种子无交集；新数据跨 split RGB 重复为 0 |
| 唯一候选结构 | 实际 stride 4/8/16、DFL bins 0…15；地板具有 10 个完整可回归 stride16 锚点 |
| 同一新 val 的旧模型基线 | 已实测并在冻结前查看；不是新候选成绩 |
| 预注册 | 已在候选训练与候选验证前冻结 |
| 几何评分独立审查 | 21项针对测试通过；另100组小例与一般评估器的TP/FP/FN/recall一致 |
| 候选训练和导出 | 40epoch完成，实际1467.923792秒；ONNX导出完成 |
| 候选 val、地板几何、部署性能 | 13/14阈值通过；唯一失败为平台总体召回0.819444＜0.98 |
| 候选模型模式 13 场景、三职业实际 reinfer、短时资源运行 | 按val失败出口不执行，不能补做以替代失败 |
| 结构审查构建前分配预算 | 实际原生审查、恢复与训练/导出回归通过；复用既有离线运行域 |
| 新 test | 因val失败保持未评分，不是待补测任务 |
| 本地完整回归与隔离安装 | 823通过、0失败/跳过；48项含原生审查的组合通过；wheel通过，覆盖重叠不相加 |
| 准确提交的远端CI | 待完成，不能由本地结果代填 |
| 模型推广 | **未通过；保留失败候选，原已验收混合模型不变** |

## 2. 已冻结的身份

基线提交为 `d11e285f6ba648c34f3bcabb1c3d193f63350da1`。本轮源代码、测试和证据仍在形成，尚无可声称全部通过的最终提交。

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
| 工程发布 | 完整检查、原生训练/导出、干净wheel、准确提交的Ubuntu/Windows/native CI | 本地通过；远端CI待完成 |

本次已经执行val失败出口：不评分test、不追加候选、不使用后续任务通过替代质量失败。[model-decision.json](../reports/release-0.3.0/model-decision.json)保存决定和所依据报告的原字节哈希。上述未执行项是按协议关闭的推广分支，不是当前交付的待补任务。完整性读取和私有副本复制可以接触test文件字节；新test没有模型预测或评分结果。

冻结训练源码未修改，源数据与实际训练副本的1730个文件逐字节一致，训练入口的前后副本检查实际执行，记录在 [training-input-provenance.json](../reports/release-0.3.0/training-input-provenance.json)。审查构建前分配预算在首批audit报告后加入，且与训练入口分开；不能把新增保护追溯算作原报告已经执行。新入口复用既有离线运行域，当前本地原生与完整回归已通过，新提交远端CI仍待完成。

地板错误采用所有平台一对一匹配后分层，p95只统计匹配地板，同时必须报告遗漏与非空真值数，不能以缺失值当通过。详细定义已冻结在预注册文件和执行计划中。

## 5. 本地工程、原生与隔离安装

最终本地JUnit包含 **823个testcase，0 failures、0 errors、0 skipped**；pytest日志为 **823 passed、6 warnings、106.26秒**，JUnit suite时间为106.129秒。6条上游warning来自两个真实ONNX导出用例中的TorchScript导出弃用和静态trace提示，未当作失败或跳过。Ruff格式检查113文件、Ruff检查通过，mypy30个源文件通过。[本地总报告](../reports/release-0.3.0/local-verification.json) · [最终JUnit](../reports/release-0.3.0/local-pytest.xml) · [最终日志](../reports/release-0.3.0/local-pytest.log)

原生结构审查、候选配置的一epoch小样本训练/导出/ORT、Pillow与离线设置恢复均实际执行。独立含原生审查的组合 **48 passed，0失败/跳过，JUnit 8.634秒**；它与823项全量回归覆盖重叠，不能相加。一epoch工程smoke只验证执行链路，没有参与本轮40epoch研究的选模，也不是追加的研究候选。[原生审查组合](../reports/release-0.3.0/native-terrain-audit.xml)

初次全量为 **821 passed、2 failed**：两个真实审查用例错误地将两张val图像的地板总数预期为1。修正后先独立核对两帧分别来自两图、每份标签恰有一块全宽地板，再严格断言总数为2；没有改生产模型、降低质量阈值或跳过原生检查。初次结果保留在 [initial JUnit](../reports/release-0.3.0/local-pytest-initial.xml) 与 [initial log](../reports/release-0.3.0/local-pytest-initial.log)。

隔离wheel验证 **通过，21.709秒**。安装包 `maple_vision_platform-0.3.0-py3-none-any.whl` SHA-256为 `fe38f69f994d987168e9c74e3a3e94b8d697767842d721b0f798f4df7a9245d6`。干净环境未安装Torch/ORT，确认两份模型YAML均被打包，自定义detector的模型地形路由不调用fixture对象检测；同时验证P0、目录快照与增量、数据集、素材以及pixel任务和326帧实际reinfer。它不使用失败候选做推广任务，也不授予候选识别质量通过。[安装报告](../reports/release-0.3.0/wheel-smoke.json)

以上是当前本地证据，**准确远端提交的Ubuntu、Windows、native CI仍待完成**。未来最终提交编号和实际Actions结果记录后，才能更新远端验收状态；模型拒绝推广结论保持独立。

## 6. 几何评分的独立审查

`terrain_metrics.py` 已完成当前工作树的只读审查。它先匹配全部平台，再按全宽地板真值分层；漏检继续计入 FN，不伪造成零误差。原图顶边和左右端点独立计量，p95采用线性插值。半宽框与过厚框虽然可达到 IoU0.5，但会暴露端点或顶边误差；无地板或全漏检时几何指标保持 `null`，须由推广判定明确拒绝。

早期独立审查时，`tests/test_terrain_metrics.py` 实际 **21 passed，0.64秒**；随后增加身份绑定和manifest变化回归，当前26项均已包含在最终823项通过结果中。独立的100组小型随机目标/预测组合，与已有一般评估器的平台、梯子 TP/FP/FN/recall全部一致。该审查未运行模型或接触 test 预测，不改变冻结协议；最终本地完整回归已通过，远端仍须对准确提交执行CI。

`vision benchmark --terrain-metrics` 产生带 `terrain_geometry` 的 schema3 报告，使用原检测批次的预测，不重复推理。省略参数时保留旧报告格式和指标。结构入口 `model audit-terrain` 只接受 train/val，其随机权重结构与成本报告不充当训练后识别质量。

## 7. 结论边界

模型对象模式没有地形像素回退，HUD仍是专门的合成像素解析。当前策略会使用平台框，但没有梯子检测驱动的攀爬分支；梯子检测通过也不能升级为梯子导航通过。

新 test 的 192 帧来自 4 个会话，两内置地图在所有 split 出现。无完全重复 RGB 不是独立真实场景的证明；对象 recall、固定场景通过率、反事实动作验证和真实任务可靠率不能互相替代。

正式客户端控制、当前真实游戏数据、网络小册子适配器、真实域识别、未见地图泛化、24小时墙钟和完整v8均未取得本轮验收。最终发布只有在准确提交与实际 CI 结果存在时记录成功；源代码、模型、研究发现和未完成事项分别报告。
