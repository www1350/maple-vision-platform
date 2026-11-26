# 本地录像接入与 OCR

`recordings.py` 把已经保存在本地的 PNG 帧目录或 MP4 文件转换为可校验的 RGB 帧会话，并支持逐帧感知回放。`ocr.py` 通过已经安装的本地 Tesseract 识别文本。这些入口均不需要运行时 LLM，也不会产生 API token 调用。

## 数据流与验证能力

| 输入与模式 | 实际能验证的内容 |
|---|---|
| 本地 PNG / MP4 导入 | 文件和像素哈希、帧顺序、声明的时间基准、尺寸、来源与导入完整性 |
| 固定录像的感知回放 | 检测器对这些既有画面的输出、未知状态数量、检测框数量 |
| 有独立真值的离线评估 | 另行通过评估入口比较检测结果与标注，计算准确性指标 |
| 模拟器闭环 | 在可交互的独立环境中实际执行动作，再观察后续变化验证结果 |

**固定录像不能验证反事实动作。** 给同一段录像加入一个“吃药”意图，不会改变录像中的血条；因此录像回放报告没有动作成功率，也不会证明正式游戏规则或账号安全。没有标注的录像也不能单凭检测框数量计算准确率。

## 导入 API

```python
from pathlib import Path
from maple_vision.recordings import (
    import_recording,
    validate_recording,
    iter_recording,
    observe_recording,
)

source = Path("my-recording-frames")
result = import_recording(
    source,
    Path("artifacts/recording-session"),
    source_uri=source.resolve().as_uri(),
    permission="explicit_permission",
    license="Owner-provided recording; local analysis permitted",
    fps=20,
    max_frames=300,
)
validate_recording(Path(result["output"]))

for frame in iter_recording(Path(result["output"])):
    print(frame.sequence, frame.timestamp_ms, frame.width, frame.height)
```

输入仅为存在的本地路径，输出目录必须不存在，且不能位于输入会话之内。来源元数据复用知识目录的规则：明确 `source_uri`、`permission` 与 `license`。这些字段保存导入者的来源和授权说明，不将其当作自动验证的所有权证明。

### PNG 帧目录

只处理目录直接子级的 `.png` 文件，按文件名字典序排列。建议使用 `000001.png`、`000002.png` 等补零名称，避免 `10.png` 排在 `2.png` 前面。PNG 目录必须显式提供 fps；没有足够信息时不会猜测原始录制速度。

帧被解码并规范化为 RGB PNG，要求尺寸一致且图片不透明。混合尺寸或有透明度的输入必须先拆分会话或明确合成。导入保留原 PNG 文件哈希，同时记录规范化 PNG 与 RGB 像素哈希。

### 本地 MP4

MP4 解码依赖可选的 OpenCV/FFmpeg 后端。入口检查普通本地文件、`.mp4` 扩展名和 MP4 `ftyp` 文件头，显式选择 FFmpeg 后端，并把其允许的传输协议限制为本地 `file` / `pipe`。不会将输入名称解释为网址、RTSP 流、播放列表或桌面采集请求。

`fps=None` 时读取视频元数据 FPS；明确传入 fps 时采用调用方声明的时间基准。fps 不是抽帧频率：本入口保留连续的前 `max_frames` 个可解码帧。输出每帧仍记录其来源帧序号，避免将遗漏帧虚构为完整录像。

较长录像可以用 `--start-frame` 从零基源帧号开始分段导入，而不提高单会话的资源上限：

```bash
python -m maple_vision recording import sample.mp4 artifacts/part-2 \
  --source-uri urn:local:sample --permission explicit_permission \
  --license "owner-approved local analysis" --presentation-time \
  --start-frame 300 --max-frames 300
```

窗口会话使用 schema3，并保存 `source_start_frame`；MP4 帧的 `source_name` 保留原视频帧号，精确时序模式还保留原始 PTS。会话内的 `sequence` 和 `timestamp_ms` 仍从 0 开始，便于独立回放；跨窗口拼接必须使用源帧号，精确模式可进一步使用 PTS，不应把两个窗口各自的零点当成同一时刻。`start_frame=0` 与旧行为和会话身份完全兼容。起点已越过可解码帧时导入失败且不发布输出。

多个精确 PTS 的 MP4 窗口可生成并复验一份覆盖报告：

```bash
python -m maple_vision recording summarize-windows \
  --session artifacts/part-1 --session artifacts/part-2 \
  --output artifacts/window-coverage.json
python -m maple_vision recording verify-windows artifacts/window-coverage.json \
  --session artifacts/part-1 --session artifacts/part-2
```

汇总会重新完整校验每个会话并按源帧起点排序。所有窗口必须绑定同一 MP4 哈希、来源说明、尺寸、PTS 时基和流编号；首段必须从 0 开始，中间不能缺帧或重叠，非末段必须因帧上限截断，末段必须未截断且到达解码器停止。报告保存每段会话 ID、源帧范围、首尾 PTS 与 RGB 哈希。解码器停止不能独立证明 MP4 容器无损，也不表示所有游戏现象已逐帧标注。

默认导入按恒定帧率估算毫秒时间，报告标记 `timing_accuracy=nominal_fps_only`。对可变帧率 MP4，安装精确时序扩展并显式选用原视频时钟。

```bash
python -m pip install -e ".[video-pts]"
python -m maple_vision recording import sample.mp4 artifacts/pts-session --source-uri urn:local:sample --permission explicit_permission --license "owner-approved local analysis" --presentation-time
```

该模式读取第一路视频流。每一帧的 RGB 和显示时间戳取自同一解码帧。非窗口 schema2 和窗口 schema3 的 `video_clock` 保存流编号及有理数时基，每帧保存原始整数 `pts`。时间间隔等于两帧 PTS 差乘以时基，单位为秒。界面与旧接口使用从本窗口第一帧开始、舍入至毫秒的时间。缺失或倒序的时间戳会中止导入，完整校验后才发布目录。变帧率回归的四帧相对时间为 0、40、160、380 毫秒，测试也核对了对应画面和原始起点。报告标记 `timing_accuracy=source_pts`。旧 schema1 录像仍按其保存的恒定帧率读取。

视频解码器返回停止时，记录 `decoder_stopped`；这一信号本身不区分文件结束和解码中断，元数据中的帧数只作为提示，不冒充实际完整性证明。`truncated=true` 特指已确认还有后续帧、但达到本次帧数上限的情况。

### 大小限制

默认最多导入 300 帧，可显式设置 1–10000；`start_frame` 范围为 0–2^53。FPS 范围为 0.1–240；单帧文件最多 64 MiB、尺寸各边最多 8192、最多 16 百万像素；一个会话额外限制累计十亿像素和 1 GiB 编码图像。MP4 输入上限为 2 GiB。达到资源限制时明确失败，不发布一个被误认为已验证的半成品会话。

## 会话合同

输出包含：

```text
recording-session/
  manifest.json
  frames/
    000000.png
    000001.png
```

manifest 保存 schema、session_id、来源类型、来源说明、源内容指纹、FPS、时间基准、尺寸、帧数、截断状态和各帧条目。session_id 是完整 manifest 主体的规范 JSON SHA-256；每个帧条目记录连续序号、毫秒时间戳、文件路径、规范化 PNG 哈希、RGB 哈希和源帧名称。

验证会拒绝重复/不连续序号、窗口起点与 MP4 源帧名不一致、时间基准不一致、超界路径、缺失或额外帧文件、文件或像素哈希不符、尺寸变化和不兼容 schema。迭代器在消费每帧时再次校验内容，避免只在会话开始检查一次。返回的 `ImageFrame` 只包含像素和图像身份，不携带模拟器状态或训练标注。

文件哈希用于完整性校验，不是对来源真实性的密码学签名。PNG 目录的源指纹覆盖本次选中的原始 PNG 输入；MP4 的源指纹覆盖整个输入文件。源文件未打包进会话的情况会通过 `source_hash_basis` 区分。

## 感知回放

```python
from pathlib import Path
from maple_vision.recordings import observe_recording
from maple_vision.onnx_detector import OnnxDetector

# 初始调试：仅适用于本项目几何图形场景的颜色基线。
observe_recording(Path("artifacts/recording-session"), Path("artifacts/palette-replay"))

# 指定已有本地模型：模型哈希仍由 OnnxDetector 校验。
detector = OnnxDetector(Path("my-model/model.onnx"), Path("my-model/manifest.json"))
observe_recording(Path("artifacts/recording-session"), Path("artifacts/model-replay"), detector)
```

输出 `observations.jsonl` 和 `report.json`。报告给出处理帧数、状态计数、分类检测数量、观测轨迹哈希；使用已知模型 manifest 的检测器时同时记录模型指纹和推理阈值。

默认 `PixelPerception` 的 HUD 和地图颜色规则属于本项目的合成测试场景。传入仅有 `detect()` 的自定义检测器时，检测框不能自动变成完整世界状态，因此 HP、MP、地图仍为未知，状态保持 UNKNOWN。即使检测器识别出了玩家，也不会推断“正式游戏已知且正确”。

报告固定记录 `action_execution=none`、`effect_success_rate=null`、`accuracy_metrics_available=false`。录像观察输出没有任务意图、执行回执或正式客户端接口。

项目在导入可选 ONNX Runtime 前设置禁用遥测环境变量，推理适配器另行调用对应禁用接口。嵌入其他程序时，仍须在该程序首次导入原生 ORT 前设置环境；录制文件的本地路径不能代替这一初始化顺序。

## 本地 OCR

```python
from pathlib import Path
from maple_vision.ocr import TesseractOCR
from maple_vision.scene import ImageFrame

ocr = TesseractOCR(
    binary=Path("/usr/bin/tesseract"),
    language="eng",
    timeout_seconds=5,
    psm=6,
)
frame = ImageFrame.from_file("artifacts/recording-session/frames/000000.png")
result = ocr.recognize(frame, crop=(0, 0, frame.width, frame.height))
print(result.text)
print(result.words)
```

不传 binary 时用 `shutil.which("tesseract")` 明确解析本地可执行文件，随后校验版本和已安装语言。显式 binary 必须是绝对路径；Windows 必须为 `.exe`，不接受命令脚本。可通过 `tessdata_dir` 指定已经存在的本地语言数据目录，缺少语言时返回错误，不自动下载。

`recognize(frame, crop=None)` 接收原图坐标的整数 xyxy 裁剪框，要求正面积、位于画面内，最多四百万像素。单次外部命令超时范围 0.1–30 秒，默认 5 秒；命令使用参数数组和 `shell=False`。PNG 通过标准输入传入，TSV 通过标准输出返回。结果包含每个词的文本、原图坐标框和分数，空结果明确为 `words=()`、`text=""`。

语言可使用 `eng+chi_sim` 形式，但对应语言包必须已安装。Tesseract 分数除以 100 后作为 `[0,1]` 范围的 confidence 输出；它不是经过校准的正确概率，也不能将识别文本自动升级为已核实游戏事实。这里没有自动回复或发送聊天消息的接口。

## 本次验证证据

专属测试 **71 项通过**，包括真实本地 MP4 编码/导入、像素顺序验证、manifest 和帧篡改检查、固定录像不会生成动作成功结论、OCR 参数和裁剪边界、超时和错误处理。

实际环境检测到 `/usr/bin/tesseract`，版本 **5.5.0**，已安装语言 **eng、osd**。用本地 Pillow 生成的清晰文字图片实际识别为 `MAPLE 123`，两个词分数约 0.96619、0.96343。该 smoke 只验证本地 CLI 调用、输出解析和坐标合同，不代表游戏小字、复杂背景或中文 OCR 准确率。

可重查本次本地产物 `artifacts/recording-ocr-smoke/evidence.json`。其中同一文字帧经过默认感知回放得到 UNKNOWN、零检测框和空动作结果，符合其不具备游戏场景依据的实际输入。

```bash
python -m pytest -q tests/test_recordings.py tests/test_ocr.py
```

未安装可选 MP4 解码器或 Tesseract 的环境会明确跳过相应真实引擎 smoke；纯接口和边界测试继续运行。跳过不是引擎准确率通过。

## 逐帧标注与评测

先用上面的导入命令建立录像会话，再生成与原始画面绑定的空白标注表。保留输入录像，只在单独的 JSON 文件中编辑标注。

```bash
python -m maple_vision recording label-template artifacts/recording-session artifacts/session-labels.json
```

填写顶层的 `reviewer` 和 `domain`。自写样例填 `authored_fixture`，有使用许可的游戏实录填 `game_recording`。每帧保留 `sequence` 和 `rgb_sha256`，填写 `map_id` 与 `objects`。类别清单为 player、monster、loot、portal、npc、platform、ladder。以下是一个物体的标注，下标为原始画面的像素坐标，右、下边界不含在框内。确认没有目标的帧填空数组，尚未审过的帧保留 null 会阻止出分。

```json
{"class_id":"monster","bbox":[60,50,70,60],"occluded":true}
```

```bash
python -m maple_vision recording benchmark --case artifacts/recording-session artifacts/session-labels.json --output artifacts/recording-score.json --model examples/lifecycle-model/model.onnx --manifest examples/lifecycle-model/manifest.json
```

可以重复传入 `--case` 汇总同一来源类型的不同会话。可重复指定 `--development-dataset DATASET`，逐张检查该数据集 train/val 与录像的解码 RGB 是否完全相同，重编码为 PNG 仍能查出。近似画面、其它训练集以及同地图的泄漏需要另外核对。报告记录指定数据集的摘要和实际检查帧数。输出文件采用独占创建，一个路径只能承载一次结果。

每个会话和地图分别报告 IoU 0.50 下的 TP、FP、FN 及逐类指标。汇总 AP 使用固定分数阈值后的 101 点精度包络，按有真值的类别求均值，IoU 从 0.50 至 0.95，间隔 0.05。默认分数阈值是 0.25。遮挡对象由标注人逐个声明，小目标为原画面积小于 32×32 像素的目标。分母为零时召回值为 null。耗时覆盖本次检测器逐帧调用，包括首次调用。报告仅评估固定画面上的目标检测。真实国服识别率仍需要独立标注的实录。

## 稀疏可见事件复核

只需要标记少量可见时间窗时，不必先给每一帧画全量物体框。下面的命令把人工选定的闭区间绑定到会话 ID、原视频 SHA-256、起止帧序号、原始显示时间和 RGB 像素哈希。输出仍须放在录像会话之外，且只允许新建，不能覆盖。

```bash
python -m maple_vision recording review-events artifacts/pts-session artifacts/visible-events.json --reviewer reviewer-id --event rope_contact_visible 120 180 --event layer_change_visible 120 190 --event damage_feedback_visible 210 225
python -m maple_vision recording verify-events artifacts/pts-session artifacts/visible-events.json
```

允许的事件名为 `attack_motion_visible`、`projectile_visible`、`damage_feedback_visible`、`airborne_visible`、`rope_contact_visible`、`layer_change_visible` 和 `camera_motion_visible`。一次报告接受 1–256 个不重复区间，起止序号均计入；输出按起点排序。复验会重新完整校验录像会话、从源帧重建每个区间，并逐字段比较报告，帧内容或报告被修改都会失败。

这些名字只陈述复核者在像素中看到的现象。它们不证明按了哪个键，不把相邻的攻击画面和伤害数字自动连成因果，也不能仅凭攻击或投射物画面识别技能。截断导入报告会明确保留 `truncated=true`，未导入的尾帧不在证据范围内。要测距离、目标框或技能反馈时序，仍需使用完整逐帧物体标注及施放测量入口。

### 导出可观看的复核事件回放

持有原录像导入窗口时，可以把已经复验的可见事件导出为 GIF。每个窗口必须配一份由该窗口生成的事件报告，覆盖报告必须证明这些窗口从源帧 0 连续到解码停止。输出采用新目录，不覆盖原录像、导入帧或人工报告。

```bash
python -m maple_vision recording render-reviewed-events \
  --window artifacts/recording-head artifacts/head-events.json \
  --window artifacts/recording-tail artifacts/tail-events.json \
  --coverage artifacts/window-coverage.json \
  --output artifacts/reviewed-event-replay
```

输出目录包含 `reviewed-events.gif` 和 `evidence.json`。画面顶部显示源帧序号、原视频时间和当前人工复核事件；默认每三帧采一帧，并在每个区间前后保留六帧上下文。`--stride`、`--context-frames` 和 `--max-width` 可在有界范围内调整。不同事件区间之间的空白时间会压缩，源帧号保留真实跳跃，证据文件列出每一张回放帧的原始 RGB 哈希。

导出前会重新验证所有窗口、完整覆盖报告和每份事件报告。回放只是原像素和人工可见事件的展示层，不新增目标框、技能名称、施放者、按键或因果结论，也不能用来证明如果平台采取另一动作会发生什么。

## 核对实录标注中的换图

`python -m maple_vision recording audit-route SESSION LABELS examples/navigation/cms-east-hill.json REPORT`

使用上面的逐帧标注表。报告列出相邻采样帧中地图名改变的位置、两帧时间以及资料图是否登记该方向的直达出口。连续相同的地图只计作停留。报告保留资料图以外的地图名和无匹配边的变化。两帧之间可能经过未采样的地图，时间区间也只是观测边界。此检查使用人工标注，不能据此计算自动识图准确率或证明角色使用了某个出口。
