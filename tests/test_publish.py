"""Publish gate and Gist verification: offline, credential-free coverage.

The irreversible upload is intentionally *not* automated (GitHub's API models
file content as JSON, so a binary ``.nemotrace`` travels over the Gist's Git
remote with the user's own credential). These tests cover the two halves that
can be trusted and tested: the pre-flight gate that produces the runbook, and
the read-only verification that a published Gist really carries the bytes the
catalog claims.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from nemoir_runtime.publication import PublicationError
from nemoir_runtime.publish import (
    PublishCheck,
    PublishPlan,
    parse_gist_ref,
    plan_publication,
    verify_published,
)
from nemoir_runtime.trace import read_archive_entries

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "docs" / "trace" / "schema" / "test-vectors"
PUBLICATION = VECTORS / "publication"
FIXTURE = PUBLICATION / "cvxpygen-publication.nemotrace"
AUDIT = VECTORS / "audit-valid.nemotrace"
REPLAY = VECTORS / "cli" / "replay-e2e.nemotrace"

pytestmark = pytest.mark.skipif(
    not PUBLICATION.exists(), reason="publication vectors require the meta checkout"
)


@pytest.fixture
def plan() -> PublishPlan:
    return plan_publication(FIXTURE, title="CVXPYgen fake-model fixture", license_id="Apache-2.0")


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def test_plan_accepts_a_reviewed_publication_archive(plan: PublishPlan) -> None:
    entries = read_archive_entries(FIXTURE)
    integrity: dict[str, Any] = json.loads(entries["integrity.json"].decode("utf-8"))
    assert plan.content_identity == integrity["content_identity"]
    assert plan.bytes == FIXTURE.stat().st_size
    assert plan.filename == FIXTURE.name
    assert plan.workflow_id == "CvxpygenH50Autoresearch"
    assert plan.workflow_id in plan.catalog_entry["artifact"]["workflow_id"]
    assert plan.catalog_entry["artifact"]["profile"] == "publication"
    assert plan.catalog_entry["license"] == "Apache-2.0"
    assert plan.catalog_entry["gist"] is None
    assert any("permanence" in warning or "durable" in warning for warning in plan.warnings)
    runbook = "\n".join(plan.commands)
    assert "gh gist create" in runbook
    assert "git push" in runbook
    assert "publish-verify" in runbook
    assert plan.content_identity in runbook
    # The archive never travels through the JSON API path.
    assert "files[" not in runbook


def test_plan_refuses_audit_and_replay_archives() -> None:
    with pytest.raises(PublicationError, match="only an attested publication-profile"):
        plan_publication(AUDIT, title="t", license_id="CC-BY-4.0")
    with pytest.raises(PublicationError, match="only an attested publication-profile"):
        plan_publication(REPLAY, title="t", license_id="CC-BY-4.0")


def test_plan_refuses_missing_title_license_and_bad_filename(plan: PublishPlan) -> None:
    with pytest.raises(PublicationError, match="title must not be empty"):
        plan_publication(FIXTURE, title="   ", license_id="CC-BY-4.0")
    with pytest.raises(PublicationError, match="license identifier"):
        plan_publication(FIXTURE, title="t", license_id=" ")
    with pytest.raises(PublicationError, match="plain"):
        plan_publication(FIXTURE, title="t", license_id="CC-BY-4.0", filename="nested/x.nemotrace")
    assert plan.catalog_entry["artifact"]["bytes"] == FIXTURE.stat().st_size


def test_plan_refuses_an_over_budget_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemoir_runtime.publish.PUBLICATION_MAX_BYTES", 16)
    with pytest.raises(PublicationError, match="over the 16 public budget"):
        plan_publication(FIXTURE, title="t", license_id="CC-BY-4.0")


def test_plan_refuses_an_unverifiable_archive(tmp_path: Path) -> None:
    broken = tmp_path / "broken.nemotrace"
    broken.write_bytes(FIXTURE.read_bytes()[:-32])
    with pytest.raises(PublicationError):
        plan_publication(broken, title="t", license_id="CC-BY-4.0")


# ---------------------------------------------------------------------------
# gist reference parsing
# ---------------------------------------------------------------------------


def test_parse_gist_ref_accepts_ids_urls_and_pins() -> None:
    gist_id = "d50de06684ef848b59f599ebe8fc1140"
    assert parse_gist_ref(gist_id) == (gist_id, None)
    assert parse_gist_ref(f"https://gist.github.com/{gist_id}") == (gist_id, None)
    assert parse_gist_ref(f"https://gist.github.com/hkalexling/{gist_id}") == (gist_id, None)
    assert parse_gist_ref(f"{gist_id}@c393593") == (gist_id, "c393593")


@pytest.mark.parametrize(
    "value",
    ["", "not-a-gist", "https://example.com/gist/abc", f"{'a' * 32}@zzz"],
)
def test_parse_gist_ref_rejects_bad_references(value: str) -> None:
    with pytest.raises(PublicationError):
        parse_gist_ref(value)


# ---------------------------------------------------------------------------
# published-gist verification (local HTTP server)
# ---------------------------------------------------------------------------


class _GistHandler(BaseHTTPRequestHandler):
    """Serve one fake gist API + raw bytes; the path table is set per test."""

    routes: ClassVar[dict[str, tuple[str, bytes]]] = {}

    def do_GET(self) -> None:
        entry = self.routes.get(self.path)
        if entry is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"message":"Not Found"}')
            return
        content_type, body = entry
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
        return None


@pytest.fixture
def gist_server() -> Iterator[tuple[str, dict[str, tuple[str, bytes]]]]:
    routes: dict[str, tuple[str, bytes]] = {}
    handler = type("Handler", (_GistHandler,), {"routes": routes})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}", routes
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


GIST_ID = "0123456789abcdef0123456789abcdef"
REVISION = "abcdef0123456789abcdef0123456789abcdef01"


def _metadata(
    *,
    filename: str,
    raw_url: str,
    revision: str = REVISION,
    extra: dict[str, Any] | None = None,
) -> bytes:
    files: dict[str, Any] = {filename: {"filename": filename, "raw_url": raw_url, "size": 1}}
    if extra:
        files.update(extra)
    payload = {
        "id": GIST_ID,
        "description": "fixture",
        "history": [{"version": revision}],
        "files": files,
    }
    return json.dumps(payload).encode("utf-8")


def _serve_fixture(
    routes: dict[str, tuple[str, bytes]], base: str, *, filename: str | None = None
) -> None:
    name = filename if filename is not None else FIXTURE.name
    raw_path = f"/raw/{GIST_ID}/{REVISION}/{name}"
    routes[f"/gists/{GIST_ID}"] = (
        "application/json",
        _metadata(filename=name, raw_url=f"{base}{raw_path}"),
    )
    routes[f"/gists/{GIST_ID}/{REVISION}"] = (
        "application/json",
        _metadata(filename=name, raw_url=f"{base}{raw_path}"),
    )
    routes[raw_path] = ("application/octet-stream", FIXTURE.read_bytes())


def test_verify_published_accepts_the_reviewed_fixture(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, routes = gist_server
    _serve_fixture(routes, base)
    check = verify_published(
        GIST_ID,
        api_base=base,
        viewer_base="https://viewer.example",
        allow_insecure=True,
    )
    assert check.ok, check.errors
    assert check.revision == REVISION
    assert check.filename == FIXTURE.name
    assert check.bytes == FIXTURE.stat().st_size
    assert check.content_identity.startswith("sha256:")
    assert check.viewer_url == f"https://viewer.example/{GIST_ID}"
    assert check.pinned_url == f"https://viewer.example/{GIST_ID}@{REVISION}"
    assert any("mutable" in warning for warning in check.warnings)


def test_verify_published_pins_an_explicit_revision(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, routes = gist_server
    _serve_fixture(routes, base)
    check = verify_published(
        f"{GIST_ID}@{REVISION}",
        api_base=base,
        allow_insecure=True,
    )
    assert check.ok
    assert check.warnings == ()


def test_verify_published_rejects_an_identity_mismatch(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, routes = gist_server
    _serve_fixture(routes, base)
    check = verify_published(
        GIST_ID,
        api_base=base,
        allow_insecure=True,
        expect_content_identity="sha256:" + "11" * 32,
    )
    assert not check.ok
    assert any("does not match" in error for error in check.errors)


def test_verify_published_rejects_a_non_publication_upload(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, routes = gist_server
    raw_path = f"/raw/{GIST_ID}/{REVISION}/audit.nemotrace"
    routes[f"/gists/{GIST_ID}"] = (
        "application/json",
        _metadata(filename="audit.nemotrace", raw_url=f"{base}{raw_path}"),
    )
    routes[raw_path] = ("application/octet-stream", AUDIT.read_bytes())
    check = verify_published(GIST_ID, api_base=base, allow_insecure=True)
    assert not check.ok
    assert any("not 'publication'" in error for error in check.errors)


def test_verify_published_rejects_a_foreign_raw_host(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, routes = gist_server
    routes[f"/gists/{GIST_ID}"] = (
        "application/json",
        _metadata(filename=FIXTURE.name, raw_url="https://evil.example/trace.nemotrace"),
    )
    with pytest.raises(PublicationError, match="raw_url host"):
        verify_published(GIST_ID, api_base=base, allow_insecure=True)


def test_verify_published_requires_one_trace_file(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, routes = gist_server
    _serve_fixture(routes, base)
    routes[f"/gists/{GIST_ID}"] = (
        "application/json",
        _metadata(
            filename=FIXTURE.name,
            raw_url=f"{base}/raw/{GIST_ID}/{REVISION}/{FIXTURE.name}",
            extra={"other.nemotrace": {"filename": "other.nemotrace", "raw_url": "x"}},
        ),
    )
    with pytest.raises(PublicationError, match=r"exactly one \.nemotrace"):
        verify_published(GIST_ID, api_base=base, allow_insecure=True)
    # An explicit filename resolves the ambiguity.
    check = verify_published(GIST_ID, api_base=base, filename=FIXTURE.name, allow_insecure=True)
    assert check.ok


def test_verify_published_reports_a_missing_gist(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, _routes = gist_server
    with pytest.raises(PublicationError, match="HTTP 404"):
        verify_published(GIST_ID, api_base=base, allow_insecure=True)


def test_verify_published_check_reports_failure_without_raising(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, routes = gist_server
    _serve_fixture(routes, base)
    check: PublishCheck = verify_published(
        GIST_ID, api_base=base, allow_insecure=True, expect_content_identity="sha256:" + "22" * 32
    )
    assert isinstance(check.errors, tuple)
    assert check.ok is False
