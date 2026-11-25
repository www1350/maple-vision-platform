"""Build a tiny self-drawn animation pack without network or game files."""

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from maple_vision.animation_pack import (
    AnimationPlacement,
    compose_animation,
    export_animation_pack,
    load_animation_pack,
    verify_animation_pack,
)
from maple_vision.asset_index import index_assets, verify_asset_index
from maple_vision.asset_registry import import_asset


def draw_frame(path: Path, pose: str) -> None:
    """Original geometric character; no downloaded, extracted or traced artwork."""
    image = Image.new("RGBA", (24, 24))
    draw = ImageDraw.Draw(image)
    rise = -1 if pose == "step" else 0
    draw.rectangle((9, 3 + rise, 14, 8 + rise), fill=(250, 202, 140, 255))
    draw.rectangle((8, 1 + rise, 15, 3 + rise), fill=(46, 77, 122, 255))
    draw.point((13, 5 + rise), fill=(22, 28, 36, 255))
    draw.rectangle((8, 9 + rise, 15, 17), fill=(49, 155, 189, 255))
    draw.rectangle((9, 18, 11, 22), fill=(44, 62, 92, 255))
    draw.rectangle((14, 18, 16, 22 if pose != "step" else 20), fill=(44, 62, 92, 255))
    if pose in {"windup", "strike"}:
        draw.rectangle((15, 10, 19, 12), fill=(250, 202, 140, 255))
        if pose == "strike":
            draw.rectangle((20, 8, 22, 14), fill=(252, 220, 93, 220))
    else:
        draw.rectangle((5 if pose == "step" else 6, 10, 7, 15), fill=(250, 202, 140, 255))
    image.save(path, format="PNG", optimize=False, compress_level=9)


def build(output: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="animation-fixture-") as temporary:
        root = Path(temporary)
        registry = root / "registry"
        assets = {}
        for pose in ("idle", "step", "windup", "strike"):
            source = root / f"{pose}.png"
            draw_frame(source, pose)
            assets[pose] = import_asset(
                registry,
                source,
                entity_id="fixture.player",
                game_version="synthetic-animation-v1",
                source_uri=f"urn:self:maple-vision-animation-fixture-v1:{pose}",
                license="CC0-1.0",
                permission="self_authored",
            )
        index = root / "assets.sqlite"
        index_assets(index, registry, tuple(entry.asset_id for entry in assets.values()))
        indexed = len(verify_asset_index(index, registry))
        clips = []
        for action, playback, poses, durations in (
            ("idle", "loop", ("idle", "step"), (240, 240)),
            ("move", "loop", ("idle", "step"), (100, 100)),
            ("attack", "once", ("windup", "strike"), (80, 120)),
        ):
            clips.append(
                {
                    "clip_id": f"fixture.player.{action}",
                    "entity_id": "fixture.player",
                    "game_version": "synthetic-animation-v1",
                    "class_id": "player",
                    "action_state": action,
                    "playback": playback,
                    "frames": [
                        {
                            "asset_id": assets[pose].asset_id,
                            "duration_ms": duration,
                            "anchor": [12, 23],
                        }
                        for pose, duration in zip(poses, durations, strict=True)
                    ],
                }
            )
        mapping = root / "mapping.json"
        mapping.write_text(
            json.dumps({"schema_version": 1, "scope": "synthetic_fixture", "clips": clips}),
            encoding="utf-8",
        )
        pack = load_animation_pack(registry, mapping)
        export_animation_pack(pack, output)
    # Temporary source PNGs, mapping, registry and SQLite index are now gone.
    restored = verify_animation_pack(output)
    placements = tuple(
        AnimationPlacement(clip, timestamp, (20 + index * 32, 30))
        for index, (clip, timestamp) in enumerate(
            [
                ("fixture.player.idle", 0),
                ("fixture.player.idle", 240),
                ("fixture.player.move", 100),
                ("fixture.player.attack", 0),
                ("fixture.player.attack", 80),
                ("fixture.player.attack", 200),
            ]
        )
    )
    compose_animation(restored, placements, size=(200, 40)).save_png(output / "preview.png")
    return {
        "scope": restored.scope,
        "clips": len(restored.clips),
        "source_images": len(restored.sources),
        "verified_index_entries": indexed,
        "pack_sha256": restored.manifest()["pack_sha256"],
        "original_paths_removed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="a new directory for the exported fixture")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))
