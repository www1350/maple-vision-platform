"""Load bounded synthetic class rules from the local, provenance-checked catalog.

The loaded rules are immutable static knowledge, not simulator observations. A
simulation embeds their canonical payload and hash, so replay has no database or
website dependency. Provenance remains explicit in the loader's result.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .catalog import CatalogRecord, _decode_record, canonical_json, query_catalog
from .sim_types import SUPPORTED_JOBS, ClassSpec


@dataclass(frozen=True)
class LoadedClassProfile:
    class_spec: ClassSpec
    profile_sha256: str
    provenance_sha256: str
    records: tuple[CatalogRecord, ...]

    def __post_init__(self) -> None:
        if self.profile_sha256 != class_spec_sha256(self.class_spec):
            raise ValueError("loaded class profile SHA256 mismatch")
        if (
            not isinstance(self.records, tuple)
            or not self.records
            or any(not isinstance(record, CatalogRecord) for record in self.records)
        ):
            raise ValueError("class profile provenance requires immutable catalog records")
        if len({record.record_id for record in self.records}) != len(self.records):
            raise ValueError("class profile provenance contains duplicate source records")
        if len({record.game_version for record in self.records}) != 1:
            raise ValueError("class profile provenance mixes game versions")
        for record in self.records:
            # Reuse the catalog's row validation so serialized evidence has the
            # same hash/status/provenance guarantees without reopening a database.
            _decode_record(tuple(asdict(record).values()))
            if record.kind != "classes" or record.entity_id != self.class_spec.id:
                raise ValueError("class profile provenance identifies a different entity")
            if record.scope != "synthetic_fixture" or record.verification_status != "VERIFIED":
                raise ValueError("class profile provenance must be VERIFIED synthetic_fixture data")
            if record.payload_sha256 != self.profile_sha256:
                raise ValueError("class profile provenance does not match the embedded rules")
        calculated = sha256(
            canonical_json([asdict(record) for record in self.records]).encode("utf-8")
        ).hexdigest()
        if self.provenance_sha256 != calculated:
            raise ValueError("class profile provenance SHA256 mismatch")


def parse_class_spec(payload: object) -> ClassSpec:
    """Require the complete ClassSpec schema, including normally defaulted fields."""
    if not isinstance(payload, dict) or set(payload) != set(ClassSpec.__dataclass_fields__):
        raise ValueError("class profile payload must contain exactly all ClassSpec fields")
    try:
        return ClassSpec(**payload)
    except TypeError as exc:
        raise ValueError("invalid class profile field types") from exc


def class_spec_sha256(spec: ClassSpec) -> str:
    if not isinstance(spec, ClassSpec):
        raise ValueError("profile must contain a ClassSpec")
    return sha256(canonical_json(asdict(spec)).encode("utf-8")).hexdigest()


def parse_loaded_class_profile(payload: object) -> LoadedClassProfile:
    """Validate a serialized loader result for offline evidence replay."""
    if not isinstance(payload, dict) or set(payload) != set(
        LoadedClassProfile.__dataclass_fields__
    ):
        raise ValueError("invalid serialized class profile fields")
    rows = payload["records"]
    if not isinstance(rows, (list, tuple)) or not rows or len(rows) > 50_000:
        raise ValueError("class profile records must be a bounded nonempty sequence")
    records: list[CatalogRecord] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(CatalogRecord.__dataclass_fields__):
            raise ValueError("invalid serialized profile provenance record")
        records.append(CatalogRecord(**row))
    return LoadedClassProfile(
        parse_class_spec(payload["class_spec"]),
        payload["profile_sha256"],
        payload["provenance_sha256"],
        tuple(records),
    )


def load_class_profile(database: str | Path, job: str, game_version: str) -> LoadedClassProfile:
    """Load one explicit fixture version; do not choose or promote disputed data.

    All candidates at this key must be VERIFIED, unconflicted synthetic fixtures.
    Identical values from multiple verified sources retain all their provenance.
    """
    if not isinstance(job, str) or job not in SUPPORTED_JOBS:
        raise ValueError("unknown fixture job for class profile")
    records = query_catalog(database, "classes", job, game_version, allow_unverified=True)
    if not records:
        raise ValueError("class profile is missing for this job and version")
    if any(record.verification_status != "VERIFIED" for record in records):
        raise ValueError("class profile requires VERIFIED, unconflicted catalog records")
    if any(record.scope != "synthetic_fixture" for record in records):
        raise ValueError("simulator class profile requires synthetic_fixture scope")
    if len({record.payload_sha256 for record in records}) != 1:
        raise ValueError("class profile has conflicting payloads")
    spec = parse_class_spec(records[0].payload)
    if spec.id != job:
        raise ValueError("class profile id does not match its catalog entity/job")
    payload_hash = class_spec_sha256(spec)
    if any(record.payload_sha256 != payload_hash for record in records):
        raise ValueError("class profile payload hash does not match validated rules")
    provenance = [asdict(record) for record in records]
    provenance_hash = sha256(canonical_json(provenance).encode("utf-8")).hexdigest()
    return LoadedClassProfile(spec, payload_hash, provenance_hash, records)
