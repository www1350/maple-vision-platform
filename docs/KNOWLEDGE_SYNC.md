# Authorized local knowledge synchronization

The v0.2 synchronization layer turns a local catalog JSON export into an auditable
plan, then applies that exact plan atomically to an existing v2 catalog. It keeps
the original evidence, game versions, verification state, and conflicting
records. It does not fetch a website or infer that an export contains current
official game rules.

`LocalExportSource` is the implemented source adapter. A live booklet connector,
network crawler, access-permission discovery, and automatic game-version mapping
remain future adapters. The included `fixtures/knowledge-demo.json` contains
synthetic fixtures; it is not an export of real game data.

## Commands and Python API

Initialize a v2 catalog once, then plan and apply an authorized local export:

```bash
maple-vision knowledge init artifacts/knowledge.sqlite
maple-vision knowledge plan-sync artifacts/knowledge.sqlite fixtures/knowledge-demo.json artifacts/knowledge-plan.json
maple-vision knowledge apply-sync artifacts/knowledge.sqlite artifacts/knowledge-plan.json
```

The initialization command is the catalog command introduced with v0.1. The
legacy `db init`/`db check` commands still target the original P0 schema; they do
not implement synchronization plans. See [CATALOG.md](CATALOG.md)
for migration and the catalog export format.

```python
from pathlib import Path
from maple_vision.knowledge_sync import apply_sync, plan_sync

summary = plan_sync(
    database=Path("artifacts/knowledge.sqlite"),
    source_json=Path("authorized-export.json"),
    output_plan=Path("artifacts/knowledge-plan.json"),
)
result = apply_sync(
    database=Path("artifacts/knowledge.sqlite"),
    plan_path=Path("artifacts/knowledge-plan.json"),
)
```

Public interfaces:

```python
plan_sync(
    database: str | Path,
    source_json: str | Path,
    output_plan: str | Path,
) -> dict[str, object]

apply_sync(
    database: str | Path,
    plan_path: str | Path,
) -> dict[str, object]

LocalExportSource(path: Path).read() -> SourceSnapshot
```

`SourceSnapshot` is a frozen dataclass containing `name: str`, `content: bytes`,
and `records: tuple[CatalogRecord, ...]`. `SourceAdapter` defines a read-only
`adapter_id` property and `read()` method. The current adapter ID is
`local_catalog_json_v1`; the public planning API deliberately accepts a local
path rather than dynamically loading an adapter from untrusted plan content.

## Input contract

The input uses the existing strict catalog export envelope:

```json
{
  "schema_version": 1,
  "records": [
    {
      "kind": "skills",
      "entity_id": "fixture-skill",
      "game_version": "synthetic-fixture-v1",
      "source_uri": "urn:self:fixture",
      "collected_at": "2026-11-24T00:00:00Z",
      "permission": "self_authored",
      "license": "CC0-1.0",
      "scope": "synthetic_fixture",
      "verification_status": "UNKNOWN",
      "payload": {"name": "Synthetic example"}
    }
  ]
}
```

Use the permission and license that actually apply to the source. These fields
record a supplied provenance assertion; parsing them does not independently
establish redistribution rights or factual accuracy. All catalog validation
still applies, including supported entity kinds, UTC collection timestamps,
strict field sets, canonical payload hashing, duplicate JSON-key rejection,
and rejection of non-finite JSON numbers.

Each evidence record has a stable ID derived from the entire validated record.
Changing its payload, collection time, provenance, declared verification state,
or game version creates a different evidence record. The same entity in two
game versions remains two separate entity-version groups.

## What a plan captures

Planning reads the source once and captures its exact UTF-8 bytes inside the
plan. It reads all catalog records in a consistent read transaction, including
unknown, provisional, and conflicted rows that normal verified-only queries may
exclude. It also captures the preserved P0 `source_records` rows.

The single JSON plan has schema version `1` and these fields:

| Field | Meaning |
| --- | --- |
| `kind`, `adapter` | Fixed `knowledge_sync_plan` and `local_catalog_json_v1` identifiers. |
| `source` | Original file name, exact `export_utf8`, byte count, SHA-256, record counts, and derived provenance/version/scope summaries. |
| `baseline` | Complete catalog evidence and legacy source rows, catalog schema version, and schema fingerprint. |
| `baseline_fingerprint` | SHA-256 of the canonical logical baseline document. |
| `expected_fingerprint` | SHA-256 of the independently calculated logical state after import. |
| `changes` | Sorted evidence IDs for `added`, `unchanged`, `changed`, and `missing`, plus effective `status_changes`. |
| `expected_import` | Expected imported/unchanged counts, conflicted entity count, and resulting catalog record count. |
| `plan_id` | SHA-256 of the canonical plan body, excluding `plan_id` itself. |

The plan contains no executable SQL, database destination, source-fetch URL, or
filesystem paths to follow. The source name is a plain file name used as audit
metadata. A source URI remains catalog provenance; applying a plan never opens
it.

Fingerprints include each record's effective verification status and the
catalog schema identity. They describe logical catalog contents rather than the
SQLite file's page layout. Repacking an otherwise identical database therefore
does not by itself invalidate a plan. Auxiliary application-table contents are
outside this catalog fingerprint; adding or changing a table, view, index, or
trigger changes the schema fingerprint.

`plan_id` and the other hashes provide integrity checks and stable identifiers.
They are not digital signatures, approval records, or authentication. Someone
who can replace every captured source and baseline field and recompute a fully
consistent new plan can create a different valid plan. Approval systems should
bind their own authenticated record to a particular `plan_id`.

The plan is self-contained after creation. Editing, removing, or replacing the
original source file does not change what an existing plan applies. To import
the revised file, create a new plan. This makes it possible to inspect and apply
the same captured evidence even when the original export is no longer present.

## Delta and conflict semantics

Deltas count evidence records, not just entities. Every unique incoming record
ID appears in exactly one of `added`, `unchanged`, or `changed`:

| Classification | Condition | Apply behavior |
| --- | --- | --- |
| `added` | New record ID and its `(kind, entity_id, game_version)` group is absent in the baseline. | Insert the evidence record. |
| `unchanged` | The same record ID already exists. | Keep the stored record, including any existing conflict state. |
| `changed` | New record ID under an existing same-version entity group. | Insert additional evidence and keep earlier evidence. |
| `missing` | A baseline record ID is absent from this particular source export. | Keep it unchanged. |

`missing` is computed against the complete catalog and is informational. For a
partial export it can include unrelated sources or versions. When an incoming
record changes an existing entity, the old evidence record can appear in
`missing` while the new evidence record appears in `changed`. Neither case
implies deletion or that an entity no longer exists in the game.

Different payload hashes within one `(kind, entity_id, game_version)` group
mark all of that group's evidence as `CONFLICTED`, using the existing catalog
import behavior. A provenance-only change with an identical payload adds
evidence without introducing a payload conflict. Existing conflicts are never
resolved by sync. Unknown and provisional evidence is never upgraded to
verified by the synchronizer.

`status_changes` records the affected record ID, previous and resulting
effective state, and whether the record already existed. New records that
arrive with a declared state and immediately enter a conflict are included.
For an identical already-conflicted record, its stored conflict state remains
unchanged even if its original declaration was verified.

Duplicate identical entries in a source export are accepted consistently with
catalog import. Delta ID lists remain unique. `expected_import.unchanged`
includes repeated entries that do not cause another insertion.

## Apply, concurrency, and idempotency

Applying first verifies the plan's strict schema and integrity hash, parses the
captured source bytes again, validates every captured baseline row, and
regenerates all deltas, provenance summaries, counts, and fingerprints. Editing
a delta or expected fingerprint and merely recomputing the outer checksum does
not pass this consistency check.

The database operation then runs in one `BEGIN IMMEDIATE` transaction:

1. Read the full current logical catalog under the writer reservation.
2. If it equals the expected after-state, return an explicit idempotent result.
3. Otherwise require an exact baseline match before importing any records.
4. Apply the existing catalog importer inside a nested savepoint.
5. Read the resulting catalog and verify both the expected fingerprint and
   import counters before committing.

A stale database, failing insert, unexpected trigger effect on catalog/legacy
rows, or failed postcondition rolls back the entire transaction. The baseline
comparison and writes share the same transaction, so another writer cannot
change the catalog between these steps. SQLite's connection timeout is five
seconds; persistent writer contention fails instead of applying a stale plan.

Returned statuses are explicit:

| Status | Meaning |
| --- | --- |
| `planned` | A new plan file was published; the database was not modified. |
| `applied` | The baseline matched and the complete import committed. |
| `already_applied` | The current catalog exactly matches this plan's expected after-state; no records were inserted. |
| `no_changes` | Baseline and expected state are identical; no records were inserted. |

If another import occurs after a successful apply, reapplying the older plan
fails as stale unless the entire catalog is again exactly its expected
after-state. Merely finding the incoming record IDs is insufficient.

All summaries include `plan_id`, source hash, baseline/expected fingerprints,
delta counts, expected import counters, and `missing_policy: "retain_history"`.
Successful apply summaries also include the observed catalog fingerprint,
record count, imported count, and an explicit `idempotent` boolean.

## File and size constraints

The database must already be an initialized v2 catalog. Plan and source inputs
must be regular local files. Direct symlinks are rejected. Database paths with
SQLite `-wal`, `-shm`, or `-journal` sidecars, including dangling sidecar
symlinks, are rejected. The sync database must use standalone `DELETE` journal
mode; sync does not change journaling settings or delete recovery files.

For an active WAL-backed source, use the catalog snapshot workflow to produce
an independent standalone copy, or close and checkpoint the source through its
own database owner before planning. Do not remove a live journal to make a
validation check pass. Treat published snapshot artifacts as immutable; create
an ordinary writable working copy when imports are intended.

Bounds are 16 MiB and 50,000 entries per export, 128 MiB per database,
100,000 combined catalog/legacy records in the baseline and expected state,
and 256 MiB per plan. These are rejection limits, not performance guarantees.
The database byte budget is checked both on entry and from SQLite's logical
`page_count × page_size` while the transaction is open. If an import would grow
past the limit, its complete transaction rolls back before commit, including
pages that have not yet been flushed to the main database file.

Plan publishing writes and flushes a temporary file in the destination
directory, then publishes it exclusively. An existing output file or symlink
is never overwritten. Validation failures leave no published partial plan.
The plan includes the complete captured export and baseline, so it should be
handled with the same access controls as those input records.

## Verification and scope

`tests/test_knowledge_sync.py` exercises deterministic plans, exact source-byte
capture, mixed deltas, preservation of historical rows and declared state,
cross-version separation, duplicate records, exact reapply behavior, stale
catalog/legacy/schema detection, malformed and internally inconsistent plans,
sidecar/symlink rejection, and no-overwrite publication. Actual SQLite triggers
and a competing connection verify rollback and the writer reservation.

Knowledge synchronization changes catalog evidence only. It does not hot-load
rules into a running policy, resolve conflicting skill definitions, establish
official game behavior, or control a game client. Consumers still select an
explicit game version and verification policy through the existing catalog
API. A future network source adapter must preserve this captured-byte,
provenance, version, and atomic-apply contract.

## 从获准的 HTTPS 导出生成离线审查计划

`knowledge fetch-plan` 读取操作者明确指定的 JSON 导出地址。上游须提供当前 catalog 导出合同。命令保存原始字节、来源端点、采集时间、许可声明、请求 URL 摘要和条目版本。URL 查询值不写入回执。命令支持固定预期 SHA-256，响应超过 16 MiB、跳转、未授权响应、结构错误或摘要不符都会终止。计划只读数据库基线。随后的 `apply-sync` 完全离线，重建差异并在事务内应用；重试保留幂等语义。

下面的可复现实验取自本仓库固定提交，数据为 14 条自写测试记录。

```bash
python -m pip install -e ".[sync]"
python -m maple_vision knowledge init artifacts/catalog.sqlite
python -m maple_vision knowledge fetch-plan artifacts/catalog.sqlite https://raw.githubusercontent.com/www1350/maple-vision-platform/91af7960c5ece68341718539e1784de5412c1106/fixtures/knowledge-domain-demo.json artifacts/fetch-plan.json --permission self_authored --license CC0-1.0 --sha256 902401e0d96daadb59b8e194a88896a35d31b1a890b19a02e8116d3269f61237
python -m maple_vision knowledge apply-sync artifacts/catalog.sqlite artifacts/fetch-plan.json
```

本例实测新增 14 条记录。小册子当前国服数据仍需要站方许可与格式映射。此接口不会把来源声明提升为真实游戏机制的核实证据。
