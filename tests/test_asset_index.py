"""SQLite discovery never substitutes for verified registry content."""

import json
import os
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from maple_vision.asset_index import (
    MAX_INDEX_BYTES,
    index_assets,
    query_assets,
    verify_asset_index,
)
from maple_vision.asset_registry import MAX_ASSET_BYTES, import_asset


def add(registry, source, *, version="fixture-v1", entity="fixture.player", color="red"):
    Image.new("RGBA", (8, 8), color).save(source)
    return import_asset(
        registry,
        source,
        entity_id=entity,
        game_version=version,
        source_uri="urn:self:asset-index-fixture",
        license="CC0-1.0",
        permission="self_authored",
    )


@pytest.fixture
def indexed(tmp_path):
    registry, database = tmp_path / "registry", tmp_path / "assets.sqlite"
    first = add(registry, tmp_path / "first.png")
    second = add(registry, tmp_path / "second.png", color="blue")
    other = add(registry, tmp_path / "other.png", version="fixture-v2")
    result = index_assets(database, registry, (first.asset_id, second.asset_id, other.asset_id))
    assert (result.imported, result.unchanged, result.total) == (3, 0, 3)
    return database, registry, first, second, other


def test_exact_version_pagination_and_verified_reimport(indexed):
    database, registry, first, second, other = indexed
    ordered = tuple(sorted((first, second), key=lambda entry: entry.asset_id))
    query = dict(entity_id="fixture.player", game_version="fixture-v1")
    assert query_assets(database, registry, **query) == ordered
    assert query_assets(database, registry, **query, limit=1) == ordered[:1]
    assert (
        query_assets(database, registry, **query, after_asset_id=ordered[0].asset_id) == ordered[1:]
    )
    assert query_assets(
        database, registry, entity_id="fixture.player", game_version="fixture-v2"
    ) == (other,)
    assert (
        query_assets(database, registry, entity_id="fixture.player", game_version="absent-version")
        == ()
    )
    assert set(verify_asset_index(database, registry)) == {first, second, other}
    repeated = index_assets(database, registry, (first.asset_id, second.asset_id))
    assert (repeated.imported, repeated.unchanged, repeated.total) == (0, 2, 3)


def test_query_reverifies_objects_even_when_metadata_and_index_are_unchanged(indexed):
    database, registry, first, _, _ = indexed
    (registry / first.relative_path).write_bytes(b"modified image")
    with pytest.raises(ValueError, match="SHA256"):
        query_assets(database, registry, entity_id=first.entity_id, game_version=first.game_version)
    with pytest.raises(ValueError, match="SHA256"):
        verify_asset_index(database, registry)


def test_failed_source_batch_does_not_create_a_database(tmp_path):
    registry = tmp_path / "registry"
    valid = add(registry, tmp_path / "first.png")
    database = tmp_path / "assets.sqlite"
    with pytest.raises(ValueError):
        index_assets(database, registry, (valid.asset_id, "0" * 64))
    assert not database.exists()


def test_new_index_publish_failure_leaves_no_partial_database(tmp_path, monkeypatch):
    registry, database = tmp_path / "registry", tmp_path / "assets.sqlite"
    entry = add(registry, tmp_path / "source.png")
    before = set(tmp_path.iterdir())

    def failed_publish(*args, **kwargs):
        raise OSError("review injected publication failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", failed_publish)
        with pytest.raises(OSError, match="publication failure"):
            index_assets(database, registry, (entry.asset_id,))
    assert set(tmp_path.iterdir()) == before and not database.exists()
    assert index_assets(database, registry, (entry.asset_id,)).imported == 1


def test_failed_source_batch_leaves_existing_index_unchanged(indexed, tmp_path):
    database, registry, first, _, _ = indexed
    before = database.read_bytes()
    new = add(registry, tmp_path / "new.png", color="green")
    with pytest.raises(ValueError):
        index_assets(database, registry, (new.asset_id, "0" * 64))
    assert database.read_bytes() == before
    assert len(verify_asset_index(database, registry)) == 3
    assert first in verify_asset_index(database, registry)


@pytest.mark.parametrize(
    "change", ["column", "metadata", "trigger", "hidden-trigger", "index", "version"]
)
def test_altered_sqlite_content_or_schema_is_not_trusted(indexed, change):
    database, registry, first, _, _ = indexed
    with sqlite3.connect(database) as connection:
        if change == "column":
            connection.execute("UPDATE asset_entries SET game_version = 'forged'")
        elif change == "metadata":
            row = connection.execute(
                "SELECT metadata_json FROM asset_entries WHERE asset_id = ?", (first.asset_id,)
            ).fetchone()
            metadata = json.loads(row[0])
            metadata["source_uri"] = "urn:forged:source"
            connection.execute(
                "UPDATE asset_entries SET metadata_json = ? WHERE asset_id = ?",
                (json.dumps(metadata), first.asset_id),
            )
        elif change in {"trigger", "hidden-trigger"}:
            name = "forged" if change == "trigger" else "sqliteXconcealed"
            connection.execute(
                f"CREATE TRIGGER {name} AFTER INSERT ON asset_entries "
                "BEGIN DELETE FROM asset_entries; END"
            )
        elif change == "index":
            connection.execute("DROP INDEX asset_entity_version")
            connection.execute("CREATE INDEX asset_entity_version ON asset_entries(sha256)")
        else:
            connection.execute("PRAGMA user_version = 2")
    with pytest.raises(ValueError):
        verify_asset_index(database, registry)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sqlite_sidecars_cannot_bypass_the_main_file_budget(indexed, suffix):
    database, registry, *_ = indexed
    Path(str(database) + suffix).write_bytes(b"unverified SQLite state")
    with pytest.raises(ValueError, match="sidecar"):
        verify_asset_index(database, registry)


def test_checkpointed_wal_mode_is_rejected_without_recreating_sidecars(indexed):
    database, registry, *_ = indexed
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    before = set(database.parent.iterdir())
    assert not Path(str(database) + "-wal").exists()
    with pytest.raises(ValueError, match="non-WAL"):
        verify_asset_index(database, registry)
    assert set(database.parent.iterdir()) == before


def test_oversized_sqlite_json_rejected_before_row_decode(indexed, monkeypatch):
    import maple_vision.asset_index as module

    database, registry, *_ = indexed
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE asset_entries SET metadata_json = ?", (json.dumps("x" * 65536),))

    def forbidden(*args, **kwargs):
        raise AssertionError("SQL field size guard must precede loading metadata rows")

    monkeypatch.setattr(module, "_decode", forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        verify_asset_index(database, registry)


def test_foreign_database_is_rejected_without_overwriting_it(tmp_path):
    database, registry = tmp_path / "foreign.sqlite", tmp_path / "registry"
    entry = add(registry, tmp_path / "image.png")
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE important(value TEXT)")
        connection.execute("INSERT INTO important VALUES ('keep')")
    before = database.read_bytes()
    with pytest.raises(ValueError, match="schema version"):
        index_assets(database, registry, (entry.asset_id,))
    assert database.read_bytes() == before


def test_sql_values_are_bound_and_original_import_path_is_not_required(tmp_path):
    registry, database = tmp_path / "registry", tmp_path / "index.sqlite"
    source = tmp_path / "source.png"
    entity = "fixture'; DROP TABLE asset_entries; --"
    entry = add(registry, source, entity=entity)
    source.unlink()
    index_assets(database, registry, (entry.asset_id,))
    assert query_assets(database, registry, entity_id=entity, game_version="fixture-v1") == (entry,)
    assert verify_asset_index(database, registry) == (entry,)


@pytest.mark.parametrize("part", ["record", "object"])
def test_source_reads_are_bounded_before_hashing_or_unbounded_read_bytes(
    tmp_path, monkeypatch, part
):
    registry, database = tmp_path / "registry", tmp_path / "index.sqlite"
    entry = add(registry, tmp_path / "source.png")
    target = (
        registry / "records" / f"{entry.asset_id}.json"
        if part == "record"
        else registry / entry.relative_path
    )
    budget = 64 * 1024 if part == "record" else MAX_ASSET_BYTES
    with target.open("wb") as stream:
        stream.truncate(budget + 1)

    def forbidden(self):
        raise AssertionError("index source verification must use bounded reads")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        index_assets(database, registry, (entry.asset_id,))
    assert not database.exists()


def test_database_byte_budget_precedes_connect(tmp_path, monkeypatch):
    database = tmp_path / "large.sqlite"
    with database.open("wb") as stream:
        stream.truncate(MAX_INDEX_BYTES + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("database budget must precede SQLite open")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        verify_asset_index(database, tmp_path / "unused-registry")


@pytest.mark.parametrize(
    "batch", [(), (True,), ("../private",), ("0" * 64,) * 2, ("0" * 64,) * 257]
)
def test_invalid_batch_rejected_without_filesystem_access(tmp_path, batch):
    database = tmp_path / "index.sqlite"
    with pytest.raises(ValueError):
        index_assets(database, tmp_path / "missing", batch)
    assert not database.exists()


@pytest.mark.parametrize("limit", [True, 0, 257, 1.0])
def test_query_limit_has_strict_type_and_bounds(indexed, limit):
    database, registry, first, _, _ = indexed
    with pytest.raises(ValueError):
        query_assets(
            database,
            registry,
            entity_id=first.entity_id,
            game_version=first.game_version,
            limit=limit,
        )


def test_index_and_source_symlinks_are_rejected(indexed, tmp_path):
    if os.name == "nt":
        pytest.skip("symlink creation is not available to unprivileged Windows CI users")
    database, registry, first, _, _ = indexed
    linked = tmp_path / "linked.sqlite"
    linked.symlink_to(database)
    with pytest.raises(ValueError, match="regular file"):
        verify_asset_index(linked, registry)
    source = registry / first.relative_path
    external = tmp_path / "external.png"
    source.rename(external)
    source.symlink_to(external)
    with pytest.raises(ValueError, match="nonsymlink"):
        verify_asset_index(database, registry)
