# MapleNecrocer 离线地图工具核对

核对对象为 `www1350/MapleNecrocer` 的公开 C# 源码，固定提交 `2aeb7db32a238522fec13cea8e66f495b0a707c4`。下表记录源码中可以定位的行为。本次未取得群聊截图对应的二进制、获准国服 WZ 输入或配对录像。

| 项目 | 核对结果 |
|---|---|
| 软件来源 | 根目录包含 MIT LICENSE，列明 Elem8100 与 2023 年版权信息。仓库 README 指定 Visual Studio 2022 或更新版本。[LICENSE](https://github.com/www1350/MapleNecrocer/blob/2aeb7db32a238522fec13cea8e66f495b0a707c4/LICENSE) · [README](https://github.com/www1350/MapleNecrocer/blob/2aeb7db32a238522fec13cea8e66f495b0a707c4/README.md) |
| 导出画面 | SaveMapForm 读取地图宽度和纵向边界，改变相机与视口，将 SpriteEngine 绘制到目标纹理，再保存 PNG。渲染选项及相机位置是复现导出所需的输入。[SaveMapForm.cs](https://github.com/www1350/MapleNecrocer/blob/2aeb7db32a238522fec13cea8e66f495b0a707c4/MapleNecrocer/SaveMapForm.cs) |
| 背景与视差 | MapBack 读取背景的 rx、ry，并在绘制/移动逻辑中读取相机坐标或速度。单张导出 PNG 仅对应一个渲染条件。[MapBack.cs](https://github.com/www1350/MapleNecrocer/blob/2aeb7db32a238522fec13cea8e66f495b0a707c4/MapleNecrocer/Client/MapBack.cs) |
| 平台结构 | Foothold 包含端点、编号、前后关联等字段，几何结构可独立核对。平台图目前只接受项目支持的静态几何，图像不能替代该结构。[Footholds.cs](https://github.com/www1350/MapleNecrocer/blob/2aeb7db32a238522fec13cea8e66f495b0a707c4/MapleNecrocer/Client/Footholds.cs) |

后续样本核对需要分别记录使用工具的构建版本、输入文件版本与许可、相机/可见图层、原始坐标系和导出摘要。当前没有用该工具导入国服素材，也没有验证指定国服版本的读写兼容性。
