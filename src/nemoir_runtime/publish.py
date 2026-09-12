"""Publication delivery helpers: pre-flight gate, upload plan, and verification.

Publishing is deliberately *not* a one-click operation. This module implements
the two halves that can be made trustworthy and tested offline, and leaves the
irreversible upload to the user's own Git/GitHub credential:

1. :func:`plan_publication` refuses anything that is not an attested,
   vault-free, in-budget ``publication`` archive and returns the exact upload
   runbook plus a ready catalog entry. GitHub's REST API models file content as
   JSON text, so a binary ``.nemotrace`` must travel over the Gist's Git remote
   (``docs/trace/spikes/gist-transport.md`` §7); this module never posts archive
   bytes as ``files.*.content``.
2. :func:`verify_published` re-downloads a public Gist through the documented
   metadata → pinned revision → ``raw_url`` path, re-verifies the archive, and
   returns the pinned citation URL. It is read-only and needs no credential.

The transport rules follow the Phase 0 spike: only public,
publication-profile, vault-free traces belong on Gist; secret Gists are not a
confidentiality boundary; pinned revision plus content identity is the citation
form; and the downloaded bytes are hostile input.
"""

from __future__ import annotations

import json
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from nemoir_runtime.publication import (
    PUBLICATION_REDACTION_POLICY,
    PublicationError,
)
from nemoir_runtime.trace import (
    MANIFEST_PATH,
    SCANNER_RULESET,
    read_archive_entries,
    verify_archive,
)

PUBLICATION_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_VIEWER_BASE = "https://hkalexling.github.io/nemoir-tracer"
GIST_RAW_HOST = "gist.githubusercontent.com"
GIST_ID_RE = re.compile(r"^[0-9a-f]{8,64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HTTP_TIMEOUT_SECONDS = 20.0
MAX_METADATA_BYTES = 4 * 1024 * 1024
USER_AGENT = "nemotrace-publish/0.1 (+https://github.com/hkalexling/nemoir)"


# ---------------------------------------------------------------------------
# Upload plan (pre-flight gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishPlan:
    """Everything a maintainer needs to publish, after the gate passed."""

    archive: Path
    filename: str
    bytes: int
    content_identity: str
    trace_id: str
    workflow_id: str
    ir_sha256: str
    events: int
    title: str
    license: str
    commands: tuple[str, ...]
    catalog_entry: dict[str, Any]
    readme: str
    warnings: tuple[str, ...]


def _artifact_facts(archive: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verified manifest + integrity facts, or a refusal."""
    try:
        verification = verify_archive(archive)
    except OSError as exc:
        msg = f"publication archive could not be read: {exc}"
        raise PublicationError(msg) from exc
    if not verification.ok:
        detail = "; ".join(verification.errors[:5]) or "verification failed"
        msg = f"publication archive does not verify: {detail}"
        raise PublicationError(msg)
    try:
        entries = read_archive_entries(archive)
    except (OSError, PublicationError) as exc:
        msg = f"publication archive could not be read: {exc}"
        raise PublicationError(msg) from exc
    manifest = cast("dict[str, Any]", json.loads(entries[MANIFEST_PATH].decode("utf-8")))
    integrity = cast("dict[str, Any]", json.loads(entries["integrity.json"].decode("utf-8")))
    capture = cast("dict[str, Any]", manifest.get("capture") or {})
    if capture.get("profile") != "publication":
        msg = (
            "only an attested publication-profile archive may be published; "
            f"this archive is {capture.get('profile')!r} "
            "(run scan-publication / attest-publication / prepare-publication first)"
        )
        raise PublicationError(msg)
    if capture.get("vault_present") is not False or "private/vault.enc" in entries:
        msg = "refusing to publish an archive that carries a vault"
        raise PublicationError(msg)
    if capture.get("attested") is not True:
        msg = "refusing to publish an archive that is not attested"
        raise PublicationError(msg)
    scanner = cast("dict[str, Any]", capture.get("scanner") or {})
    if scanner.get("status") != "passed":
        msg = "refusing to publish an archive whose scanner did not pass"
        raise PublicationError(msg)
    if not cast("dict[str, Any]", manifest.get("provenance") or {}).get("complete"):
        msg = "refusing to publish an archive without complete provenance"
        raise PublicationError(msg)
    size = archive.stat().st_size
    if size > PUBLICATION_MAX_BYTES:
        msg = (
            f"archive is {size} bytes, over the {PUBLICATION_MAX_BYTES} public budget; "
            "produce a smaller publication projection instead of splitting the trace"
        )
        raise PublicationError(msg)
    return manifest, integrity


def _counter(entries: dict[str, bytes]) -> int:
    return len(
        [line for line in entries["public/events.ndjson"].split(b"\n") if line.strip()]
    )


def plan_publication(
    archive: str | Path,
    *,
    title: str,
    license_id: str,
    filename: str | None = None,
    viewer_base: str = DEFAULT_VIEWER_BASE,
) -> PublishPlan:
    """Gate one archive for publication and return the upload runbook.

    Raises :class:`PublicationError` for an audit/replay archive, a vault, an
    unattested archive, a failed scanner, incomplete provenance, an
    over-budget artifact, or a missing license.
    """
    path = Path(archive)
    manifest, integrity = _artifact_facts(path)
    if not title.strip():
        msg = "publication title must not be empty"
        raise PublicationError(msg)
    if not license_id.strip():
        msg = "publication requires a license identifier (for example CC-BY-4.0 or Apache-2.0)"
        raise PublicationError(msg)
    release_name = filename if filename is not None else path.name
    if not release_name.endswith(".nemotrace") or "/" in release_name or "\\" in release_name:
        msg = "published filename must be a plain *.nemotrace name"
        raise PublicationError(msg)
    capture: dict[str, Any] = manifest.get("capture") or {}
    workflow: dict[str, Any] = manifest.get("workflow") or {}
    content_identity = str(integrity.get("content_identity", ""))
    if not _SHA256_RE.fullmatch(content_identity):
        msg = "archive integrity index has no usable content identity"
        raise PublicationError(msg)
    try:
        entries = read_archive_entries(path)
    except (OSError, PublicationError) as exc:  # pragma: no cover - verified above
        msg = f"publication archive could not be read: {exc}"
        raise PublicationError(msg) from exc
    events = _counter(entries)
    readme = (
        f"# {title}\n\n"
        f"NemoTrace publication archive: `{release_name}`\n\n"
        f"- content identity: `{content_identity}`\n"
        f"- workflow: `{workflow.get('id', 'unknown')}`\n"
        f"- redaction policy: `{capture.get('redaction_policy', PUBLICATION_REDACTION_POLICY)}`\n"
        f"- scanner: `{SCANNER_RULESET}` (passed)\n"
        f"- license: `{license_id}`\n\n"
        "Open it at "
        f"{viewer_base}/<gist-id>@<revision> (no server, client-only viewer).\n"
    )
    commands = (
        "# 1. create the Gist shell with your own credential (text only; the API",
        "#    cannot carry a binary .nemotrace safely)",
        f'gh gist create --public --desc "{title}" README.md',
        "#    -> prints https://gist.github.com/<gist-id>",
        "",
        "# 2. push the archive over the Gist's Git remote",
        "git clone https://gist.github.com/<gist-id>.git nemotrace-gist",
        f"cp {path} nemotrace-gist/{release_name}",
        f"cd nemotrace-gist && git add {release_name} && git commit -m "
        f"'Add {release_name}' && git push",
        "",
        "# 3. verify the uploaded bytes and print the pinned citation link",
        f"nemotrace publish-verify <gist-id> --filename {release_name} "
        f"--expect-content-identity {content_identity}",
    )
    catalog_entry: dict[str, Any] = {
        "id": f"{workflow.get('id', 'trace')}-{content_identity[-12:]}",
        "title": title,
        "description": f"NemoTrace publication archive ({events} events).",
        "artifact": {
            "path": release_name,
            "format": str(manifest.get("format", "nemoir.trace/0.1")),
            "profile": "publication",
            "content_identity": content_identity,
            "trace_id": str(manifest.get("trace_id", "")),
            "workflow_id": str(workflow.get("id", "")),
            "ir_sha256": str(workflow.get("ir_sha256", "")),
            "bytes": path.stat().st_size,
        },
        "gist": None,
        "tags": [],
        "license": license_id,
        "attestation": {
            "reviewer": None,
            "reviewed_at": None,
            "redaction_policy": str(
                capture.get("redaction_policy", PUBLICATION_REDACTION_POLICY)
            ),
            "scanner": SCANNER_RULESET,
            "consent": None,
        },
        "citation": {"title": title, "publisher": "NemoIR project", "url": None},
        "notes": "Fill reviewer/reviewed_at/consent and citation from the published attestation.",
    }
    warnings = (
        "A public Gist is public and durable: searchable, cached, forked, and part of Git "
        "history. Deleting it is not remediation.",
        "A secret Gist is not private storage.",
        "Publication reduces risk but does not prove that reviewed identifiers or approved "
        "scalar metrics are non-sensitive.",
        "The uploaded bytes are only trusted after publish-verify matches the content identity.",
    )
    return PublishPlan(
        archive=path,
        filename=release_name,
        bytes=path.stat().st_size,
        content_identity=content_identity,
        trace_id=str(manifest.get("trace_id", "")),
        workflow_id=str(workflow.get("id", "")),
        ir_sha256=str(workflow.get("ir_sha256", "")),
        events=events,
        title=title,
        license=license_id,
        commands=commands,
        catalog_entry=catalog_entry,
        readme=readme,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Read path (verification of a published Gist)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishCheck:
    """Result of re-downloading and verifying one published Gist artifact."""

    gist_id: str
    revision: str
    filename: str
    bytes: int
    content_identity: str
    expected_identity: str | None
    viewer_url: str
    pinned_url: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_gist_ref(value: str) -> tuple[str, str | None]:
    """Accept a bare id, or a ``gist.github.com`` URL, optionally ``@revision``."""
    candidate = value.strip()
    if not candidate:
        msg = "gist reference must not be empty"
        raise PublicationError(msg)
    if candidate.startswith(("http://", "https://")):
        parsed = urllib.parse.urlsplit(candidate)
        if parsed.netloc not in ("gist.github.com", "www.gist.github.com"):
            msg = "gist URL must be on gist.github.com"
            raise PublicationError(msg)
        parts = [part for part in parsed.path.split("/") if part]
        if not parts:
            msg = "gist URL has no id"
            raise PublicationError(msg)
        candidate = parts[-1]
    revision: str | None = None
    if "@" in candidate:
        candidate, _, revision = candidate.partition("@")
    if not GIST_ID_RE.fullmatch(candidate):
        msg = "gist id must be a hex string"
        raise PublicationError(msg)
    if revision is not None and not REVISION_RE.fullmatch(revision):
        msg = "gist revision must be a hex commit sha"
        raise PublicationError(msg)
    return candidate, revision


def _http_get(url: str, *, max_bytes: int, accept: str, allow_insecure: bool = False) -> bytes:
    """Read one URL with an explicit scheme policy.

    ``allow_insecure`` exists only for local tests and enterprise mirrors that
    the caller names themselves; the Gist path always requires https and never
    accepts a trace-supplied host.
    """
    request = urllib.request.Request(  # noqa: S310 - scheme is pinned by the caller
        url, headers={"Accept": accept, "User-Agent": USER_AGENT}
    )
    allowed_schemes = ("https", "http") if allow_insecure else ("https",)
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
            final = urllib.parse.urlsplit(str(response.geturl()))
            if final.scheme not in allowed_schemes:
                msg = "gist request redirected to a non-https URL; refusing"
                raise PublicationError(msg)
            data = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        msg = f"gist request failed with HTTP {exc.code}"
        raise PublicationError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"gist request failed: {exc.reason}"
        raise PublicationError(msg) from exc
    if len(data) > max_bytes:
        msg = f"gist response exceeds {max_bytes} bytes"
        raise PublicationError(msg)
    return data


def verify_published(
    gist_ref: str,
    *,
    filename: str | None = None,
    api_base: str = DEFAULT_API_BASE,
    viewer_base: str = DEFAULT_VIEWER_BASE,
    expect_content_identity: str | None = None,
    allow_insecure: bool = False,
) -> PublishCheck:
    """Verify one public Gist trace through the documented transport path.

    Fetches latest metadata, resolves the pinned revision's metadata, downloads
    the selected file's ``raw_url`` (host-checked), re-verifies the archive, and
    returns the pinned citation URL. Never uses a credential and never follows
    a trace-supplied URL.
    """
    gist_id, pinned = parse_gist_ref(gist_ref)
    metadata_url = f"{api_base.rstrip('/')}/gists/{gist_id}"
    if pinned is not None:
        metadata_url = f"{metadata_url}/{pinned}"
    raw_metadata = _http_get(
        metadata_url,
        max_bytes=MAX_METADATA_BYTES,
        accept="application/vnd.github+json",
        allow_insecure=allow_insecure,
    )
    metadata = cast("dict[str, Any]", json.loads(raw_metadata))
    history = cast("list[Any]", metadata.get("history") or [])
    revision = str(history[0].get("version")) if history else ""
    if not REVISION_RE.fullmatch(revision):
        msg = "gist metadata has no usable revision"
        raise PublicationError(msg)
    warnings: list[str] = []
    if pinned is None:
        warnings.append("resolved the mutable 'latest' revision; cite the pinned link instead")
    elif pinned != revision:
        msg = "pinned revision metadata does not match the requested revision"
        raise PublicationError(msg)
    files = cast("dict[str, Any]", metadata.get("files") or {})
    names = [name for name in files if name.endswith(".nemotrace")]
    if filename is not None:
        chosen = filename if filename in files else None
    elif len(names) == 1:
        chosen = names[0]
    else:
        chosen = None
    if chosen is None:
        found = ", ".join(sorted(files)) or "none"
        msg = (
            f"gist must contain exactly one .nemotrace file "
            f"(found {len(names)} trace file(s) among: {found}); pass --filename to choose"
        )

        raise PublicationError(msg)
    raw_url = str(cast("dict[str, Any]", files[chosen]).get("raw_url", ""))
    parsed_raw = urllib.parse.urlsplit(raw_url)
    raw_hosts = {GIST_RAW_HOST}
    if allow_insecure:
        # Local-test / enterprise-mirror opt-in: the caller named this origin.
        raw_hosts.add(urllib.parse.urlsplit(api_base).netloc)
    raw_schemes = ("https", "http") if allow_insecure else ("https",)
    if parsed_raw.scheme not in raw_schemes or parsed_raw.netloc not in raw_hosts:
        msg = "gist raw_url host is not gist.githubusercontent.com; refusing"
        raise PublicationError(msg)
    data = _http_get(
        raw_url,
        max_bytes=PUBLICATION_MAX_BYTES,
        accept="application/octet-stream",
        allow_insecure=allow_insecure,
    )
    errors: list[str] = []
    content_identity = ""
    with tempfile.TemporaryDirectory(prefix="nemotrace-publish-") as work:
        local = Path(work) / chosen
        local.write_bytes(data)
        report = verify_archive(local)
        if not report.ok:
            errors.extend(report.errors[:5] or ("archive verification failed",))
        else:
            entries = read_archive_entries(local)
            integrity = cast(
                "dict[str, Any]", json.loads(entries["integrity.json"].decode("utf-8"))
            )
            content_identity = str(integrity.get("content_identity", ""))
            manifest = cast(
                "dict[str, Any]", json.loads(entries[MANIFEST_PATH].decode("utf-8"))
            )
            capture: dict[str, Any] = manifest.get("capture") or {}
            if capture.get("profile") != "publication":
                errors.append(
                    f"published archive profile is {capture.get('profile')!r}, not 'publication'"
                )
            if capture.get("vault_present") is not False:
                errors.append("published archive declares a vault")
            if capture.get("attested") is not True:
                errors.append("published archive is not attested")
    if expect_content_identity is not None:
        if not _SHA256_RE.fullmatch(expect_content_identity):
            msg = "--expect-content-identity must be a sha256: tag"
            raise PublicationError(msg)
        if content_identity != expect_content_identity:
            errors.append("published content identity does not match the expected identity")
    viewer_url = f"{viewer_base.rstrip('/')}/{gist_id}"
    pinned_url = f"{viewer_url}@{revision}"
    return PublishCheck(
        gist_id=gist_id,
        revision=revision,
        filename=chosen,
        bytes=len(data),
        content_identity=content_identity,
        expected_identity=expect_content_identity,
        viewer_url=viewer_url,
        pinned_url=pinned_url,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


__all__ = [
    "DEFAULT_API_BASE",
    "DEFAULT_VIEWER_BASE",
    "PUBLICATION_MAX_BYTES",
    "PublishCheck",
    "PublishPlan",
    "parse_gist_ref",
    "plan_publication",
    "verify_published",
]
