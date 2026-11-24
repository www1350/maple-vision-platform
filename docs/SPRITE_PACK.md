# 注册素材接入渲染与数据集

`SpritePack` 把本地 Asset Registry 中通过验证的静态图片接入 `SceneRenderer` 和 Dataset Factory。
默认不启用 pack，原自绘场景和数据 manifest 内容保持一致；没有附送或下载真实游戏图片。

## 使用方式

先通过现有 `assets import` 或 `asset_registry.import_asset()` 注册自己获准使用的 PNG/JPEG/WebP，再把返回的 asset_id 写入映射文件：

```json
{
  "schema_version": 1,
  "scope": "synthetic_fixture",
  "classes": {
    "player": "替换为注册返回的64位asset_id",
    "monster": "替换为注册返回的64位asset_id"
  }
}
```

这个示例里的占位 ID 必须替换成真实注册 ID。scope 可选 synthetic_fixture / authorized_assets；后者仍是导入者对来源的声明，不是权利认证。
classes 是 player/monster/loot/portal/npc/platform/ladder 的非空子集；未映射类别沿用原合成图形。
registry 路径单独提供，不从映射 JSON 加载任意网络或系统路径。

```python
from pathlib import Path
from maple_vision.sprite_pack import load_sprite_pack
from maple_vision.scene import SceneRenderer
from maple_vision.simulation import Simulator
from maple_vision.dataset import generate_dataset

pack = load_sprite_pack(Path("my-registry"), Path("sprite-map.json"))
renderer = SceneRenderer(sprite_pack=pack)
sample = renderer.render_sample(Simulator().snapshot())
Path("sample.png").write_bytes(sample.frame.png_bytes())

generate_dataset(
    Path("sprite-dataset"), sessions=9, frames_per_session=8, seed=42, sprite_pack=pack
)
```

CLI 对应：

```bash
maple-vision dataset create sprite-dataset --sessions 9 --frames 8 --seed 42 --asset-registry my-registry --sprite-map sprite-map.json
maple-vision dataset check sprite-dataset
```

两个素材参数必须成对出现。`lab run` 也支持同样参数；可与 `--model/--manifest` 一起使用，在实际自定义画面上验证检测器闭环。自定义图片不能直接依赖调色板 PixelPerception，未识别到玩家时会暂停并在达到未知预算后停止，不会改读真值或偷偷换回默认角色图形。

```bash
maple-vision lab run --output sprite-run --asset-registry my-registry --sprite-map sprite-map.json --model model.onnx --manifest model-manifest.json
maple-vision lab replay sprite-run
```

Python 对应 `run_episode(output, config, detector, sprite_pack=pack)`。运行包嵌入 `sprite-pack.json` 和原始 `sprite_objects/` 图片，记录包内所有文件哈希，并要求 summary 的 pack 元数据一致。回放在渲染前再次校验原始图片、来源元数据、RGBA 和 pack 哈希；删除原 registry 和源图片后仍可验证。默认运行不新增这些文件，也不改变原帧像素。

## 像素、alpha 和标签

源图片先验证 registry 元数据及原始 bytes 的 SHA-256，再解码成不可变 RGBA。全透明图片和多帧图片被拒绝。
当前按实体矩形使用 NEAREST 拉伸，源图宽高比可以变化；碰撞矩形和物理仍使用独立 fixture，尚无动画、锚点或真实角色碰撞校准。

alpha=0 的像素既不绘入画面，也不扩大该实例的标签；alpha=255 覆盖背景并遮挡之前的实例。
0<alpha<255 使用 alpha 混合，同时保留仍可透见的下层实例。标签以非零 alpha 可见像素生成；仅全不透明像素把下层实例视作完全遮挡。
最终框裁剪到视口，少于 3 像素或被遮挡后不足原可见面积 25% 的实例不输出。玩家自绘的眼睛不会叠加到注册角色图片上。

这是可见图像标签，不是碰撞盒标签。极低 alpha、与背景同色或细小纹理是否可辨认，仍需独立训练集质量评估。

## 可复现与溯源

启用 pack 的数据集保留 origin=synthetic-fixture-v1，因为场景、动作和规则仍是合成的；另加 visual_origin=synthetic_geometry_with_authorized_assets 和 sprite_pack。
pack 记录每类 AssetEntry、原始文件哈希、解码 RGBA 哈希以及整体 pack 哈希；完整原始图片复制到 `sprite_objects/`。
验证器重新读取这些图片，核对来源元数据、内容和解码结果；仅修改 JSON 哈希不能隐藏错误的 RGBA 哈希或源文件损坏。
不依赖原 registry 目录即可验证生成数据；重新生成时仍需同一个 pack 源文件和规则版本。

默认兼容性已有固定像素和标签哈希回归。接入后重新生成原 18×16、seed=20261124 数据，manifest SHA-256 仍为：

`ecc0ff6c4e9bbebc9915f0313bb3b227a99ecc321051ee2ae2e8a0eff096ad67`

这保证本次可选接入未改变正在使用的默认训练数据；自定义素材数据自然具有不同内容哈希。
