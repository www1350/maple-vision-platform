"""Extension contracts only. P0 does not crawl, import assets or run models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .contracts import Detection, Frame


@dataclass(frozen=True)
class KnowledgeRecord:
    entity_id: str
    game_version: str
    source_uri: str
    payload_sha256: str
    payload_json: str


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    game_version: str
    source_uri: str
    sha256: str
    local_path: Path


class KnowledgeProvider(Protocol):
    def get(self, entity_id: str, game_version: str) -> KnowledgeRecord | None: ...


class AssetImporter(Protocol):
    def import_asset(self, path: Path, game_version: str) -> AssetRecord: ...


class ObjectDetector(Protocol):
    def detect(self, frame: Frame) -> tuple[Detection, ...]: ...
