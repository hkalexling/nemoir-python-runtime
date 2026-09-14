"""Publish gate and Gist verification: offline, credential-free coverage.

The irreversible upload is intentionally *not* automated (GitHub's API models
file content as JSON, so a binary ``.nemotrace`` travels over the Gist's Git
remote with the user's own credential). These tests cover the two halves that
can be trusted and tested: the pre-flight gate that produces the runbook, and
the read-only verification that a published Gist really carries the bytes the
catalog claims.
"""

from __future__ import annotations

import io
import json
import shlex
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest

from nemoir_runtime.canonical import sha256_tag, to_canonical_bytes
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
    from collections.abc import Callable, Iterator

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


@pytest.mark.parametrize(
    "filename",
    [
        "trace'; echo PWN; #.nemotrace",
        "trace with space.nemotrace",
        "trace$(id).nemotrace",
        "trace`id`.nemotrace",
        'trace".nemotrace',
        "trace.nemotrace.sh",
        ".hidden",
    ],
)
def test_plan_refuses_shell_hostile_filenames(plan: PublishPlan, filename: str) -> None:
    """A published filename lands in a copy/paste runbook: it must be inert."""
    with pytest.raises(PublicationError, match="plain"):
        plan_publication(FIXTURE, title="t", license_id="CC-BY-4.0", filename=filename)
    assert plan.filename == FIXTURE.name


def test_plan_runbook_quotes_user_controlled_tokens() -> None:
    hostile_title = 'fixture"; echo RUNBOOK_INJECTION; $(id) `id` # '
    plan = plan_publication(FIXTURE, title=hostile_title, license_id="CC-BY-4.0")
    normalized = " ".join(hostile_title.split())
    create_line = next(line for line in plan.commands if line.startswith("gh gist create"))
    tokens = shlex.split(create_line)
    # The hostile title is exactly one inert argument, not a command sequence.
    assert tokens[tokens.index("--desc") + 1] == normalized
    commit_line = next(line for line in plan.commands if line.startswith("cd nemotrace-gist"))
    commit_tokens = shlex.split(commit_line.replace(" && ", " "))
    assert f"Add {FIXTURE.name}" in commit_tokens
    copy_line = next(line for line in plan.commands if line.startswith("cp "))
    copy_tokens = shlex.split(copy_line)
    assert copy_tokens[1] == str(FIXTURE)
    assert copy_tokens[2] == f"nemotrace-gist/{FIXTURE.name}"


def test_plan_normalizes_and_bounds_the_title() -> None:
    plan = plan_publication(FIXTURE, title="  two\n\tlines  ", license_id="CC-BY-4.0")
    assert plan.title == "two lines"
    with pytest.raises(PublicationError, match="at most"):
        plan_publication(FIXTURE, title="x" * 500, license_id="CC-BY-4.0")


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
    public: bool = True,
) -> bytes:
    files: dict[str, Any] = {filename: {"filename": filename, "raw_url": raw_url, "size": 1}}
    if extra:
        files.update(extra)
    payload = {
        "id": GIST_ID,
        "description": "fixture",
        "public": public,
        "history": [{"version": revision}],
        "files": files,
    }
    return json.dumps(payload).encode("utf-8")


def _reindexed_fixture(mutate: Callable[[dict[str, Any]], None]) -> bytes:
    """Rebuild the publication fixture with a mutated manifest and fresh hashes."""
    entries = read_archive_entries(FIXTURE)
    manifest: dict[str, Any] = json.loads(entries["manifest.json"].decode("utf-8"))
    mutate(manifest)
    entries["manifest.json"] = to_canonical_bytes(manifest)
    integrity: dict[str, Any] = json.loads(entries["integrity.json"].decode("utf-8"))
    for entry in cast("list[dict[str, Any]]", integrity["entries"]):
        data = entries[cast("str", entry["path"])]
        entry["sha256"] = sha256_tag(data)
        entry["uncompressed_bytes"] = len(data)
    identity = {
        "format": "nemoir.trace.content-identity/0.1",
        "entries": sorted(
            (
                {
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "uncompressed_bytes": entry["uncompressed_bytes"],
                }
                for entry in cast("list[dict[str, Any]]", integrity["entries"])
            ),
            key=lambda item: cast("str", item["path"]),
        ),
    }
    integrity["content_identity"] = sha256_tag(to_canonical_bytes(identity))
    entries["integrity.json"] = to_canonical_bytes(integrity)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            archive.writestr(name, entries[name])
    return buffer.getvalue()


def _serve_bytes(routes: dict[str, tuple[str, bytes]], base: str, data: bytes) -> None:
    raw_path = f"/raw/{GIST_ID}/{REVISION}/{FIXTURE.name}"
    metadata = _metadata(filename=FIXTURE.name, raw_url=f"{base}{raw_path}")
    routes[f"/gists/{GIST_ID}"] = ("application/json", metadata)
    routes[f"/gists/{GIST_ID}/{REVISION}"] = ("application/json", metadata)
    routes[raw_path] = ("application/octet-stream", data)


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


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("capture.scanner.status", "failed", "scanner did not pass"),
        ("capture.scanner.ruleset", "secrets-v9", "scanner ruleset"),
        ("capture.redaction_policy", "audit-v1", "redaction policy"),
        ("capture.publication_eligible", False, "publication_eligible"),
        ("capture.attested", False, "not attested"),
        ("provenance.complete", False, "complete provenance"),
    ],
)
def test_verify_published_applies_the_strict_artifact_gate(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
    field: str,
    value: Any,
    expected: str,
) -> None:
    """A re-indexed Gist artifact must not verify as an attested publication."""
    base, routes = gist_server

    def mutate(manifest: dict[str, Any]) -> None:
        target: Any = manifest
        *parents, leaf = field.split(".")
        for key in parents:
            target = target[key]
        target[leaf] = value

    _serve_bytes(routes, base, _reindexed_fixture(mutate))
    check = verify_published(GIST_ID, api_base=base, allow_insecure=True)
    assert not check.ok
    assert any(expected in error for error in check.errors), check.errors


def test_verify_published_refuses_a_secret_gist(
    gist_server: tuple[str, dict[str, tuple[str, bytes]]],
) -> None:
    base, routes = gist_server
    _serve_fixture(routes, base)
    metadata = _metadata(
        filename=FIXTURE.name,
        raw_url=f"{base}/raw/{GIST_ID}/{REVISION}/{FIXTURE.name}",
        public=False,
    )
    routes[f"/gists/{GIST_ID}"] = ("application/json", metadata)
    routes[f"/gists/{GIST_ID}/{REVISION}"] = ("application/json", metadata)
    with pytest.raises(PublicationError, match="not public"):
        verify_published(GIST_ID, api_base=base, allow_insecure=True)
