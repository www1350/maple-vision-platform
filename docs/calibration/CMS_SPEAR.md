# 国服怀旧版龙骑士技能校准记录

2026-11-25 从国服怀旧服小册子分别核对两个技能。以下为资料字段，供未来逐帧比对。

| 技能及等级 | 伤害与目标 | 消耗 | 表列范围宽 × 高 |
| --- | --- | --- | --- |
| 枪连击 1311001，Lv30 | 每段 170%，3 段，最多 3 个目标 | 24 MP | 134 × 70 |
| 龙咆哮 1311006，Lv30 | 240%，最多 15 个目标 | 30 MP，表列耗血 30% | 800 × 600 |

枪连击的资料动作名为 burster2。Lv25 最多两个目标，Lv26 增至三个。龙咆哮 Lv30 表列技能间隔为 2 秒。出手至首次命中、末次命中至可移动的实测时间仍为空。资料宽高缺少人物原点、摄像机缩放与目标受击框，不能直接转换为离线场景的前后射程。

本轮没有取得同技能、同等级与武器、可核对命中和落空两侧边界的国服连续实录。工作台现有“模拟范围攻击”继续使用合成技能标识，两个真实技能名保持未支持状态。后续采样需保留原始帧和时间戳、施放方向、人物与目标位置以及可见伤害反馈。

离线能力包现已能表示“一次施放、一次资源消耗和一次冷却，对最多 N 个目标各结算 M 段”的结构，并用 `spear_fighter` 自写样例验证 3 段 × 3 目标、提前击杀和单次掉落。该能力只补齐执行结构：样例的伤害、射程、冷却和蓝耗均为合成值，三段在一个离散步中结算。它没有把下表的 170%、134 × 70、24 MP 或 burster2 动画写入国服运行配置，也不构成枪连击已校准。真实参数仍必须由合法连续实录逐帧核对后另立版本。

来源：[枪连击技能表](https://mxdc.dvg.cn/skill_info.php?id=1311001) · [龙咆哮技能表](https://mxdc.dvg.cn/skill_info.php?id=1311006)

## 测量一段有标注的录像

先导入原录像并逐帧填写地图和对象标注。在施放动作首次可见的帧中选定角色和目标，索引为该帧 objects 数组的零基下标。再指定第一条可归因的反馈所在帧；如观察期内无反馈，填写观察截止帧和 no_feedback。以下序号仅演示参数格式。

```bash
python -m maple_vision recording measure-strike SESSION LABELS REPORT --start 15 --end 22 --player-index 0 --monster-index 1 --skill-id 1311001 --level 30 --weapon spear --facing right --feedback damage_visible --feedback-frame 18 --feedback-frame 20 --feedback-frame 22 --reviewer reviewer-id
```

报告将录像会话、标注摘要、首尾帧哈希、时间基准和观察者绑定在一起。距离按起始帧原图像素计算，前向中心差为负表示目标在身后；纵向中心差向下为正。`--feedback-frame` 可重复 1–16 次，必须按时间严格递增且落在观察窗口内；报告保留每次可见反馈的原帧哈希、距动作起始的毫秒数和相邻反馈间隔。`no_feedback` 禁止同时填写反馈帧。观察区间从可见动作起算，不含物理按键至画面响应的延迟。伤害数字的归属需要人工核对；同一伤害数字持续多帧不能重复标成多段。出现 MISS 可能是命中率判定，无反馈也可能源于遮挡，单次样本无法给出射程边界。

多条报告可在同一领域、技能等级、武器和原图尺寸下汇总：

```bash
python -m maple_vision recording verify-strike RECORDING LABELS.json HIT-1.json
python -m maple_vision recording summarize-strikes --report HIT-1.json --report HIT-2.json --report MISS.json --output SUMMARY.json
```

汇总分别报告可见伤害、MISS 和无反馈样本的原图距离与观察时间分布，并按左右朝向拆分。含反馈序列时另列可见事件数分布、首次反馈偏移和相邻反馈间隔，但固定保持 `calibrated_timing=false`：可见反馈不保证与真实伤害段一一对应。最远可见伤害只记为“已观察到的命中距离下界”。MISS 可能来自命中率，无反馈可能来自遮挡或归因错误，因此报告不会生成射程上界，也不会宣布已完成射程校准。重复录像区间、不同技能等级、武器或画面尺寸会被拒绝。

需要保留逐份复核结果时，直接用原录像、冻结标注和报告三元组生成汇总：

```bash
python -m maple_vision recording summarize-strikes --verified-case SESSION_A LABELS_A.json HIT-1.json --verified-case SESSION_B LABELS_B.json MISS.json --output VERIFIED.json
```

每个窗口的来源字段都通过重算后，汇总才会发布，`source_fields_verified` 记录通过复核的窗口数。报告摘要取自参与统计的同一份字节。技能、朝向和伤害归属仍由复核人判断。

有连续收招画面时，把 `--end` 延长到观察结束，并用 `--movement-frame` 标记最后一次可见反馈后首次可归因于角色主动移动的帧。无法区分主动移动、击退或镜头平移时留空。报告保存原帧身份与两帧之间的毫秒数；汇总只统计实际标记的窗口。玩家可能延迟移动，因此这项观察仍无法确定技能最早可取消或可移动的时刻。
