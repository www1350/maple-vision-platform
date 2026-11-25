"""Explicit animation-to-render-class bindings with portable captured sources.

Scene clips use the simulator timestamp as an absolute animation clock. Anchors
attach to an entity's bottom center at native pixel size. These are renderer
inputs, not observation metadata or inferred gameplay animation states.
"""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from .animation_pack import AnimationFrame, AnimationPack, verify_animation_pack
from .catalog import canonical_json, strict_json
from .sprite_pack import SPRITE_CLASSES

MAX_SCENE_ANIMATION_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SceneAnimation:
    pack: AnimationPack
    bindings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.pack) is not AnimationPack:
            raise ValueError("scene animation requires a verified animation pack")
        if not isinstance(self.bindings, tuple) or not 1 <= len(self.bindings) <= 7:
            raise ValueError("scene animation requires 1..7 immutable class bindings")
        names: set[str] = set()
        for binding in self.bindings:
            if not isinstance(binding, tuple) or len(binding) != 2:
                raise ValueError("invalid scene animation binding")
            class_id, clip_id = binding
            if not isinstance(class_id, str) or class_id not in SPRITE_CLASSES or class_id in names:
                raise ValueError("duplicate or unsupported animation render class")
            clip = self.pack.get(clip_id)
            if clip is None or clip.class_id != class_id:
                raise ValueError("scene binding does not match a declared clip class")
            names.add(class_id)
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings)))

    def sample(self, class_id: str, timestamp_ms: int) -> AnimationFrame | None:
        clip_id = next((clip for name, clip in self.bindings if name == class_id), None)
        if clip_id is None:
            return None
        clip = self.pack.get(clip_id)
        assert clip is not None
        return clip.sample(timestamp_ms).frame

    def manifest(self) -> dict[str, Any]:
        content = {"pack": self.pack.manifest(), "bindings": dict(self.bindings)}
        return {
            "schema_version": 1,
            "content": content,
            "content_sha256": hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest(),
        }

    def write_sources(self, output: Path) -> None:
        """Write into a caller-owned fresh staging directory; never overwrite."""
        raw = (canonical_json(self.manifest()) + "\n").encode("utf-8")
        if len(raw) > MAX_SCENE_ANIMATION_BYTES:
            raise ValueError("scene animation exceeds manifest byte budget")
        path = output / "scene-animation.json"
        if path.exists() or path.is_symlink():
            raise FileExistsError("scene animation evidence already exists")
        self.pack.write_sources(output)
        with path.open("xb") as stream:
            stream.write(raw)


def bind_scene_animation(pack: AnimationPack, bindings: object) -> SceneAnimation:
    if not isinstance(bindings, dict) or not 1 <= len(bindings) <= 7:
        raise ValueError("animation bindings must be an object with 1..7 classes")
    return SceneAnimation(pack, tuple(bindings.items()))


def export_scene_animation(scene: SceneAnimation, output: Path) -> dict[str, Any]:
    if type(scene) is not SceneAnimation:
        raise ValueError("scene export requires verified animation bindings")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("animation scene output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        scene.write_sources(staging)
        if load_scene_animation(staging) != scene:
            raise ValueError("exported scene animation did not reproduce captured content")
        if output.exists() or output.is_symlink():
            raise FileExistsError("animation scene output already exists")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return scene.manifest()


def load_scene_animation(bundle: Path, manifest: object | None = None) -> SceneAnimation:
    bundle = Path(bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("invalid animation scene directory")
    if manifest is None:
        path = bundle / "scene-animation.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing animation scene manifest")
        with path.open("rb") as stream:
            raw = stream.read(MAX_SCENE_ANIMATION_BYTES + 1)
        if len(raw) > MAX_SCENE_ANIMATION_BYTES:
            raise ValueError("scene animation exceeds manifest byte budget")
        try:
            manifest = strict_json(raw.decode("utf-8"))
        except UnicodeError as exc:
            raise ValueError("animation scene manifest must be UTF-8") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "content", "content_sha256"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["content"], dict)
        or set(manifest["content"]) != {"pack", "bindings"}
    ):
        raise ValueError("invalid animation scene manifest")
    content = manifest["content"]
    if len(canonical_json(manifest).encode("utf-8")) > MAX_SCENE_ANIMATION_BYTES:
        raise ValueError("scene animation exceeds manifest byte budget")
    pack = verify_animation_pack(bundle, content["pack"])
    scene = bind_scene_animation(pack, content["bindings"])
    if canonical_json(scene.manifest()) != canonical_json(manifest):
        raise ValueError("animation scene identity mismatch")
    return scene
