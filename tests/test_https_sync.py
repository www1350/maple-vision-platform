import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from maple_vision.catalog import catalog_summary, initialize_catalog
from maple_vision.cli import main
from maple_vision.knowledge_sync import HttpsExportSource, apply_sync, plan_from_source

FIXTURE = (Path(__file__).resolve().parents[1] / "fixtures/knowledge-domain-demo.json").read_bytes()
URL = "https://example.test/exports/fixture.json?grant=opaque"


def serve(monkeypatch, status=200, headers=None, content=FIXTURE):
    calls = []

    @contextmanager
    def stream(method, url, **options):
        calls.append((method, url, options))
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    status, headers=headers or {}, stream=httpx.ByteStream(content)
                )
            )
        ) as client:
            with client.stream(method, url) as response:
                yield response

    monkeypatch.setattr(httpx, "stream", stream)
    return calls


def test_fetch_plan_once_apply_offline_and_retry(tmp_path, monkeypatch, capsys):
    db = tmp_path / "catalog.sqlite"
    plan = tmp_path / "remote.json"
    initialize_catalog(db)
    before = catalog_summary(db)
    calls = serve(monkeypatch)
    assert (
        main(
            [
                "knowledge",
                "fetch-plan",
                str(db),
                URL,
                str(plan),
                "--permission",
                "self_authored",
                "--license",
                "CC0",
                "--sha256",
                hashlib.sha256(FIXTURE).hexdigest(),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == 2 and summary["adapter"] == "https_catalog_json_v1"
    assert catalog_summary(db) == before
    assert len(calls) == 1 and calls[0][2]["follow_redirects"] is False
    raw = plan.read_text()
    assert "grant=opaque" not in raw
    captured = json.loads(raw)
    assert captured["retrieval"]["endpoint"] == "https://example.test/exports/fixture.json"
    assert captured["source"]["export_utf8"].encode() == FIXTURE
    assert captured["retrieval"]["request_url_sha256"] == hashlib.sha256(URL.encode()).hexdigest()
    assert apply_sync(db, plan)["status"] == "applied"
    assert apply_sync(db, plan)["idempotent"]
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("status", "headers", "content", "pin"),
    [
        (302, {"location": "https://other.test/x"}, FIXTURE, None),
        (403, {}, b"denied", None),
        (200, {"content-length": "16777217"}, b"{}", None),
        (200, {"content-length": "5"}, b"{}", None),
        (200, {"content-encoding": "br"}, b"", None),
        (200, {}, b"<html>login</html>", None),
        (200, {}, FIXTURE, "0" * 64),
    ],
)
def test_invalid_download_does_not_publish_or_change_catalog(
    tmp_path, monkeypatch, status, headers, content, pin
):
    db = tmp_path / "catalog.sqlite"
    plan = tmp_path / "new.json"
    initialize_catalog(db)
    before = catalog_summary(db)
    serve(monkeypatch, status, headers, content)
    with pytest.raises(ValueError):
        plan_from_source(db, HttpsExportSource(URL, "explicit_permission", "allowed", pin), plan)
    assert not plan.exists() and catalog_summary(db) == before


@pytest.mark.parametrize(
    "url",
    [
        "http://example.test/file",
        "https://user:password@example.test/x",
        "https://example.test/x#fragment",
        "file:///tmp/export",
    ],
)
def test_invalid_source_never_opens_connection(monkeypatch, url):
    calls = serve(monkeypatch)
    with pytest.raises(ValueError):
        HttpsExportSource(url, "public_license", "CC0").read()
    assert not calls
