# v0.5 时序状态与结构化任务验收

**最终源码的本地完整回归、实际任务/故障验证及隔离安装通过。** 准确提交的 Ubuntu、Windows、原生 CI 和合并回执由本次 GitHub PR 记录；只有该 head 全部通过后才能合并。原始 v8 仍有开放工作，见 [逐章对照](PLAN_ALIGNMENT.md)。本轮执行合同见 [V0_5_PLAN](V0_5_PLAN.md) 与 [v05-contract.yaml](../engineering/v05-contract.yaml)。

## 实现范围

在显式 `--runtime structured` 路径中，真实执行以下链路：

`VisualObservation → TemporalStateEstimator → BeliefState → TaskCandidate/Arbiter → TaskInstance → ActionPlanner → ActionScheduler → ActionIntent → Simulator → ExecutionReceipt → EffectVerification`。

- 字段和几何轨迹具有来源、观测时刻、置信度、有效期与本帧测量标记。未知/重复/未来帧不污染已接受水位，地图或感知来源变化重置关联上下文；保留框不会冒充当前检测。
- 生存、战斗、Buff、拾取、交互、切图、恢复和等待候选经明确优先级仲裁；任务生命周期、抢占/暂停、取消、超时与有界失败均在实际运行中执行。
- 多步骤计划独立于动作；标飞转向后跳标使用新的观测。调度器一次只允许一个注册意图，拒绝过期、取消、重放、错序和上下文不符的动作。
- 控制器只执行获准动作，真实回执与效果分离。当前攻击/拾取/Buff 缺少足够可见状态时保持 UNKNOWN；命令接受或目标框消失都不是成功证明。
- schema2 运行包嵌入全部状态/任务配置，并分别保存原始观测、belief、任务/计划/意图、dispatch、实际 receipt 及最终状态。实际 reinfer 重建全部 N+1 帧，不以旧轨迹代替新推理。

默认 legacy 行为、旧 RunConfig 与 schema1 继续兼容。三职业固定任务子集、有限地图与合成测量边界没有因新架构而自动扩大成任意职业/任务或真实国服准入。

## 最终源码身份与本地工程结果

| 对象 | 证据 |
|---|---|
| 基线 | `87de967ad036c60ef4eacd86a4d44ef17ec9a2c8`，完整原文对齐后的 main |
| 包版本 | `0.5.0` |
| 35 个生产 Python 模块聚合 SHA-256 | `a11ff92ee7cf1ff560367d8a18179cf4bf3102c94d4ba32721dcc46206285068` |
| 最终本地 wheel SHA-256 | `1423093687a93ffac310c16a681f5a16a40114c32c40ec22b63c4b3db5b45932` |
| 完整 pytest | **1187 passed，0 failures/errors/skipped，6 warnings，122.78 秒** |
| Ruff / mypy / 依赖 | 格式、lint、35 模块严格类型、pip check 通过 |
| 隔离安装 | **通过，23.182 秒**；仓库外仅安装运行依赖，从 wheel 导入 |

完整测试使用 `MAPLE_TRAINING_SMOKE=1`，包含真实的一 epoch 工具测试、ORT、录像与 OCR。这些工程测试不构成新的研究候选，也没有重新打开已拒绝的 v0.3 模型实验。6 条 warning 来自原有上游导出/trace提示，原始日志保留。

新增测试分为时序状态60项、任务72项、独立审查34项、配置/CLI10项。它们已包含于1187项中，不能再相加。安装测试同时实际运行 legacy、外部地图和 structured 任务；structured安装任务320步，321帧重推理，确认导入路径与所有证据未被改写。

[完整测试日志](../reports/release-0.5.0/local-pytest.log) · [JUnit](../reports/release-0.5.0/local-pytest.xml) · [安装结果](../reports/release-0.5.0/wheel-smoke.json) · [文件身份与检查汇总](../reports/release-0.5.0/local-verification.json)

## 实际任务与逐帧复验

各行均为实际独立执行，完成后由真值 oracle 核验全部目标，再执行动作 replay 与新的感知/状态/任务 reinfer。正式验收过程开始与结束的生产源码身份一致。

| 感知/地图 | 职业 | 执行步数 | 重新推理帧数 | 结果 |
|---|---|---:|---:|---|
| 像素/内置两图 | 标飞 | 320 | 321 | 完成、独立目标及两种复验通过 |
| 像素/内置两图 | 龙骑 | 342 | 343 | 同上 |
| 像素/内置两图 | 牧师 | 310 | 311 | 同上 |
| 像素/外部三图 | 标飞 | 564 | 565 | 同上 |
| 像素/外部三图 | 龙骑 | 511 | 512 | 同上 |
| 像素/外部三图 | 牧师 | 499 | 500 | 同上 |
| 既有 YOLO 混合/内置两图 | 标飞 | 322 | 323 | 同上，且核验实际加载模型哈希 |
| 既有 YOLO 混合/内置两图 | 龙骑 | 320 | 321 | 同上 |
| 既有 YOLO 混合/内置两图 | 牧师 | 312 | 313 | 同上 |

这里使用的是已准入的 `examples/lifecycle-model/model.onnx`，SHA-256 为 `7e9f61576f63be400909db6f9eb6cb89b5b935385d01a24155c7f765d2beb6e3`。模型提供动态对象，地形/HUD仍为合成像素；没有新增训练、检测质量评分、模型推广或新地图泛化结论。[逐任务结果](../reports/release-0.5.0/structured-jobs.json)

内置/外部地图分别运行三职业×seed0/1/2的9个正常任务，加4类故障，各 **13/13 符合预期**：短时丢帧和重复帧恢复完成，持续未知停止，输入冻结有界失败。两个13项中的停止/失败是预期故障处理结果，不算业务任务完成。全部8个故障包还逐帧重新推理验证。[矩阵结果](../reports/release-0.5.0/structured-suites.json)

8个实际故障轮次的完整原始证据以独立 tar.gz 保存在报告目录，索引含哈希和大小；可解包后直接执行 `lab replay` / `lab reinfer`。完整正式报告另以 gzip 保留，没有省略历史状态来缩小原始证据。[故障包索引](../reports/release-0.5.0/retained-fault-bundles.json) · [完整正式报告](../reports/release-0.5.0/structured-acceptance.json.gz)

## 稳定性与测量边界

短时运行请求60秒，实际 **62.287847秒、28轮**，全部任务完成。采样64次，无采样错误；最大采样当前进程 RSS 为197099520字节。进程此前已加载 ORT 并运行模型任务，此数字包含整个进程，不是纯像素策略或单模型内存；同时执行的其他进程也可能影响主机调度。

这仍是短时工程验证。模拟时间与墙钟分别保存，**没有完成86400秒，也没有把短时结果换算为24小时**。[运行与资源完整记录](../reports/release-0.5.0/structured-soak.json)

## 独立审查与真实修复

| 发现 | 已验证修复 |
|---|---|
| 复合计划可能使用重复/倒序观测 | 后续步骤要求新的单调来源身份 |
| `1 == True` 掩盖执行回执类型矛盾 | structured执行、dispatch和receipt都要求严格布尔 |
| Lab忽略任务引擎的 stopped | 引擎终止立即退出，不再空转到预算耗尽 |
| 感知来源改变仍确认移动或继续旧计划 | 上下文切换取消旧计划，旧效果不能成功 |
| 重放边界允许在来源有效期恰好结束时授权 | 与调度器一致严格拒绝到期动作 |
| 导出的生命周期历史字典引用内部状态 | 每条事件深拷贝，外部修改快照不污染运行时 |

最后一项在第一轮正式验收已通过后才发现。根审查也实际复现了断言失败；修复前源码摘要为 `f98fd0f3dffb9690c8965b2c86292b9b5215b051af808310d8b22788419e13db`。第一次探针误用字段名得到KeyError，随后使用真实event字段复现污染，两份日志均保留。旧完整测试、wheel及正式结果以 `before-snapshot-fix` 归档；修复后重新执行完整1187项、全部正式任务/故障/短时运行、重新构建并隔离安装。本文上面的数字全部来自最终源码。

[真实缺陷反例](../reports/release-0.5.0/snapshot-isolation-before-fix.log) · [修复前完整验收](../reports/release-0.5.0/structured-acceptance-before-snapshot-fix.json.gz) · [独立审查回归](../tests/test_v05_review.py)

## 复现与继续实施

```bash
python scripts/verify_structured_runtime.py --output artifacts/v05-acceptance
MAPLE_TRAINING_SMOKE=1 python -m pytest -q
python -m build --wheel --no-isolation
python scripts/wheel_smoke.py --wheel dist/maple_vision_platform-0.5.0-py3-none-any.whl
```

验收入口需要已有inference依赖和仓库内模型/地图文件，不下载模型。每次输出使用新目录；完整配置、真实失败与源码身份保留。

本轮闭合状态/任务基础，不修改原始总纲，也不宣称完整v8完成。后续继续领域数据、动画、能力编译、导航、技能效果、验证产品及管理界面；真实源许可/数据、独立真实录像、完整模型对照与实际24小时仍有各自门禁。准确最终head检查和合并由GitHub PR记录，避免报告自身提交哈希产生重复归档循环。
