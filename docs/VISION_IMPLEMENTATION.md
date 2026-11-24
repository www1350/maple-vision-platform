# 离线视觉与数据集实现

## 当前能力及证据边界

`scene.py` 默认使用项目内自行编写的几何图形生成 320×192 RGB 场景。没有捆绑或导出游戏资源。通过可选 [SpritePack](SPRITE_PACK.md) 可以加载已经注册并校验的本地静态 RGBA 图片；默认数据和渲染保持兼容。
远景、视差装饰、地形、梯子、传送门、NPC、掉落、怪物、玩家和 HUD 分层绘制；可设置较窄视口验证镜头变换。
所有玩法数值来自 `synthetic-fixture-v1` 模拟器，不能解释为国服规则。

`vision.py` 的 `PixelPerception` 是固定调色板基线，利用图像像素、四邻域连通分量及可见的 HP/MP 条、地图标记和对话框识别状态。
它没有读取模拟器、对象列表或标签的接口。其在合成场景上的准确率不能代表 YOLO，更不能代表真实录像上的准确率。
实际图像域需要独立标注、独立真实会话测试和另行验证的检测模型。

`synthetic-palette-v2` 增加可见遮挡依据的平台片段融合：仅考虑上、下边缘相同的水平片段，且片段间矩形区域的每个像素都必须属于已识别的前景调色板（玩家、怪物、掉落、传送门、NPC 或梯子）。
出现任意背景或未知颜色像素就保留间隙；不同高度也不合并。不使用地图结构或 Ground Truth 补齐平台。
这是对可见遮挡的保守推断，不能证明遮挡物后方实际存在连续地形；真实背景间隙和未知遮挡均有独立测试。

## Python 接口

```python
ImageFrame(sequence, timestamp_ms, width, height, pixels)
ImageFrame.from_file(path, sequence=0, timestamp_ms=0)
frame.to_image()
frame.png_bytes()

renderer = SceneRenderer(width=320, height=192, style_seed=0, sprite_pack=None)
frame = renderer.render(snapshot)  # 给视觉 Agent 的唯一输入
sample = renderer.render_sample(snapshot)  # 仅数据生成和独立验证
sample.frame
sample.annotations  # tuple[Detection, ...]

observation = PixelPerception().observe(frame)
detections = PixelPerception().detect(frame)

generate_dataset(output, sessions=9, frames_per_session=8, seed=0, sprite_pack=None)
validate_dataset(output)
```

`ImageFrame` 只包含帧序号、时间、宽、高和 RGB24 bytes。buffer 长度必须严格匹配尺寸。
`VisualObservation` 包含身份、状态、置信度、检测框、HP/MP 比例、地图 ID、对话框可见性和来源 ID。
资源条被遮挡、图像为空、地图标记损坏或玩家不唯一时状态为 UNKNOWN；缺失值不填成 0 或成功。
HUD 数值由 100 像素条读取，比例量化误差不超过 0.005。置信度是基线的确定性匹配标记，不是校准概率。

类别顺序为 `player, monster, loot, portal, npc, platform, ladder`，YOLO 标签类别编号依此固定。
框使用像素 `xyxy`，右、下边界不包含在内。绘制器使用实例可见性 mask 去除完全遮挡、可见面积低于 25% 或少于 3 像素的对象。
部分遮挡框覆盖剩余可见像素；所有框裁剪到视口。该标注约定须在真实训练集导入时保持一致。

## 数据集布局与独立检查

```text
output/
  data.yaml
  manifest.json
  images/train/*.png
  images/val/*.png
  images/test/*.png
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
```

至少 3 个会话，按约 60%/20%/20% 分组划分；同会话所有帧处于同一分区。
当每张地图有至少 3 个会话时，为各分区预留每张地图的会话；小样本时比例会调整以保持地图覆盖。
每个会话记录种子、职业、地图和独立背景样式种子。相同配置产生相同 manifest 与图像/标签内容；不同输出目录不改变内容。
背景样式变化不改变对象类别或标签。此划分防止同会话相邻帧泄漏，但不能单独证明真实场景泛化，也不是跨地图留一评估。
每个会话先执行确定性移动、攻击、离开的准备动作，产生掉落并保留活怪；不修改模拟器内部状态。
验证摘要包含每个分区的类别正例数量、缺失类别、会话数和地图分布；小数据集不能凭总帧数声称覆盖全部真实场景。

`data.yaml` 使用 YAML 兼容的 JSON 格式，省略固定绝对路径，使数据集可整体移动。
生成时先写临时目录、运行完整验证，再重命名为指定输出；现有目录不会覆盖。
验证包括 PNG 解码、RGB 尺寸、图像、标签与训练配置 SHA-256、标签类别及归一化边界、帧序号和时间单调性、会话一致性、跨分区重复图像、未登记文件、目录越界和训练配置。
修改标签后重算哈希不能绕过几何校验。SHA-256 用于内容完整性，不是来源真实性或授权证明。

## 已覆盖的故障

测试使用直接绘制的独立像素图验证检测，避免全部测试都依赖同一个渲染实现。
覆盖分离目标、连通分量合并、对角不连通、行间断层、单像素噪声、0 HP 与未知的区分、坏 HUD、坏地图标记、视口裁剪、完全遮挡、数据污染及路径越界。
实际执行结果以项目验收文档和 CI 日志为准。

## 平台片段修复的独立对照

2026-11-24，对同一份 18 会话 × 16 帧数据的 test 分区 64 帧重新评估，IoU=0.5、置信度阈值=0.25。
数据种子为 20261124，manifest SHA-256 为 `ecc0ff6c4e9bbebc9915f0313bb3b227a99ecc321051ee2ae2e8a0eff096ad67`。
评估使用排除训练生成缓存的临时副本，图像、标签和 manifest 未修改；两次输入哈希一致。

| 指标 | 原连通分量基线 | 增加可见遮挡融合 |
|---|---:|---:|
| TP / FP / FN | 568 / 134 / 65 | 633 / 3 / 0 |
| micro precision | 0.809117 | 0.995283 |
| micro recall | 0.897314 | 1.000000 |
| macro AP50 | 0.905217 | 0.998362 |
| 平台 AP50 | 0.336521 | 0.988531 |

仍有 3 个平台 FP，未把它们隐藏或改成成功。这只证明该合成样本上的修复效果，不代表未知地图或真实录像精度。
独立像素测试覆盖真实背景间隙不合并、各种已知前景遮挡时合并、一个未知像素阻止合并、多次遮挡合并和不同高度不合并。

## 当前训练集位置覆盖缺口

上述数据中玩家框左边缘 x 的分布为：train 160 张，范围 52–264，中位数 130；val 64 张，范围 44–240，中位数 140；test 64 张，范围 72–244，中位数 168。
所有分区均没有 x<40 的玩家样本。模拟器起点 x=24 因此位于当前训练数据位置覆盖之外；较高 test AP 不能直接证明起点识别可靠。
此处仅记录测量，没有在正在训练时修改数据工厂或已有数据。补充起点、边缘、空场景及独立新会话后，应生成新版本并重新训练、评估和跑闭环。
