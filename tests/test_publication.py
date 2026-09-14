"""Phase 5 publication transform: projection, attestation, and refusals.

The publication path is a fresh projection of an ``audit`` archive under the
stricter ``publication-v1`` policy (``docs/trace/redaction-policy.md`` §1).
These tests prove the security-relevant properties: the source gates, the
projection allowlist, the deterministic/fresh identity, the blocking scanner,
and that an attestation cannot cover a projection the reviewer never saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from nemoir_runtime import cli
from nemoir_runtime.canonical import parse_json_strict, sha256_tag, to_canonical_bytes
from nemoir_runtime.publication import (
    PUBLICATION_ATTESTATION_FORMAT,
    PUBLICATION_REPORT_FORMAT,
    PublicationError,
    PublicationOptions,
    attestation_from_report,
    build_attestation,
    load_attestation,
    prepare_publication,
    publication_report_path,
    scan_publication,
    write_attestation,
    write_publication_report,
)
from nemoir_runtime.trace import (
    read_archive_entries,
    verify_archive,
    write_trace_archive,
)
from tests.test_parity import _drive_fixture  # type: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "docs" / "trace" / "schema" / "test-vectors"
AUDIT_FIXTURE = VECTORS / "audit-valid.nemotrace"
CVXPYGEN_FIXTURE = VECTORS / "cvxpygen-public.nemotrace"
REPLAY_FIXTURE = VECTORS / "cli" / "replay-e2e.nemotrace"
PUBLICATION_VECTORS = VECTORS / "publication"
PUBLICATION_SOURCE = PUBLICATION_VECTORS / "source.nemotrace"
PUBLICATION_NAME = "source.nemotrace"

CONSENT = "I reviewed the disclosure report and certify this trace is safe to publish."
RECORDS = [
    "manifest.json",
    "public/workflow.graph.json",
    "public/events.ndjson",
    "public/summary.json",
    "integrity.json",
]

pytestmark = pytest.mark.skipif(
    not VECTORS.exists(), reason="publication tests require the meta checkout"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _attest(projection: Any, options: PublicationOptions) -> Any:
    return build_attestation(
        projection,
        options=options,
        reviewer="Alex Ling",
        license_id="CC-BY-4.0",
        consent=CONSENT,
        reviewed_at="2026-09-12T00:00:00.000Z",
    )


def _rebuild_archive(source: Path, target: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Rewrite an archive with mutated canonical entries and a fresh index."""
    entries = read_archive_entries(source)
    graph: dict[str, Any] = cast(
        "dict[str, Any]", parse_json_strict(entries["public/workflow.graph.json"].decode())
    )
    events: list[dict[str, Any]] = [
        cast("dict[str, Any]", parse_json_strict(line.decode()))
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip()
    ]
    document: dict[str, Any] = {
        "manifest": cast("dict[str, Any]", parse_json_strict(entries["manifest.json"].decode())),
        "graph": graph,
        "events": events,
        "summary": cast(
            "dict[str, Any]", parse_json_strict(entries["public/summary.json"].decode())
        ),
    }
    mutate(document)
    payloads: dict[str, bytes] = {
        "manifest.json": to_canonical_bytes(document["manifest"]),
        "public/workflow.graph.json": to_canonical_bytes(document["graph"]),
        "public/events.ndjson": b"".join(
            to_canonical_bytes(record) + b"\n" for record in cast("list[Any]", document["events"])
        ),
        "public/summary.json": to_canonical_bytes(document["summary"]),
    }
    index = [
        {
            "path": path,
            "media_type": (
                "application/x-ndjson" if path.endswith(".ndjson") else "application/json"
            ),
            "uncompressed_bytes": len(data),
            "sha256": sha256_tag(data),
        }
        for path, data in sorted(payloads.items())
    ]
    identity = {
        "format": "nemoir.trace.content-identity/0.1",
        "entries": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "uncompressed_bytes": item["uncompressed_bytes"],
            }
            for item in index
        ],
    }
    payloads["integrity.json"] = to_canonical_bytes(
        {
            "format": "nemoir.trace.integrity/0.1",
            "algorithm": "sha256",
            "entries": index,
            "content_identity": sha256_tag(to_canonical_bytes(identity)),
        }
    )
    write_trace_archive(target, payloads)
    return target


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    code = cli.main(argv)
    return code, capsys.readouterr().out


def _run_err(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    code = cli.main(argv)
    return code, capsys.readouterr().err


# ---------------------------------------------------------------------------
# frozen cross-runtime vectors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PUBLICATION_VECTORS.exists(), reason="publication vectors require the meta checkout"
)
def test_projection_matches_frozen_vectors() -> None:
    """The projected entries must equal the bytes the vectors freeze."""
    projection = scan_publication(PUBLICATION_SOURCE, archive_name=PUBLICATION_NAME)
    for name, entry in (
        ("expected-manifest.json", "manifest.json"),
        ("expected-graph.json", "public/workflow.graph.json"),
        ("expected-ledger.ndjson", "public/events.ndjson"),
        ("expected-summary.json", "public/summary.json"),
        ("expected-integrity.json", "integrity.json"),
    ):
        frozen = (PUBLICATION_VECTORS / name).read_bytes()
        assert projection.entries[entry] == frozen, f"{entry} differs from {name}"


def test_digests_and_stats_match_frozen_vectors() -> None:
    frozen: dict[str, Any] = cast(
        "dict[str, Any]",
        json.loads((PUBLICATION_VECTORS / "expected-digests.json").read_text(encoding="utf-8")),
    )
    projection = scan_publication(PUBLICATION_SOURCE, archive_name=PUBLICATION_NAME)
    assert projection.projection_sha256 == frozen["projection_sha256"]
    assert projection.trace_id == frozen["trace_id"]
    assert projection.content_identity == frozen["content_identity"]
    assert projection.stats.as_dict() == frozen["stats"]
    assert list(projection.findings) == frozen["findings"]
    assert projection.source.as_dict() == frozen["source"]


def test_disclosure_report_matches_frozen_vectors() -> None:
    projection = scan_publication(PUBLICATION_SOURCE, archive_name=PUBLICATION_NAME)
    report = to_canonical_bytes(projection.report(attested=False, attestation=None)) + b"\n"
    assert report == (PUBLICATION_VECTORS / "expected-report.json").read_bytes()


def test_attestation_matches_frozen_vectors() -> None:
    report: Any = json.loads(
        (PUBLICATION_VECTORS / "expected-report.json").read_text(encoding="utf-8")
    )
    attestation = attestation_from_report(
        report,
        reviewer="Alex Ling",
        license_id="CC-BY-4.0",
        consent=CONSENT,
        reviewed_at="2026-09-12T00:00:00.000Z",
    )
    document = to_canonical_bytes(attestation.as_dict()) + b"\n"
    assert document == (PUBLICATION_VECTORS / "expected-attestation.json").read_bytes()


def test_option_variants_match_frozen_vectors() -> None:
    frozen: dict[str, dict[str, Any]] = cast(
        "dict[str, dict[str, Any]]",
        json.loads(
            (PUBLICATION_VECTORS / "expected-option-digests.json").read_text(encoding="utf-8")
        ),
    )
    variants = {
        "default": PublicationOptions(),
        "allow_tool_names": PublicationOptions(allow_tool_names=("reader", "writer")),
        "keep_relative_paths": PublicationOptions(keep_relative_paths=True),
        "both": PublicationOptions(allow_tool_names=("reader",), keep_relative_paths=True),
    }
    for label, options in variants.items():
        projection = scan_publication(
            PUBLICATION_SOURCE, options=options, archive_name=PUBLICATION_NAME
        )
        expected = frozen[label]
        assert projection.projection_sha256 == expected["projection_sha256"], label
        assert projection.trace_id == expected["trace_id"], label
        assert projection.content_identity == expected["content_identity"], label
        assert projection.stats.as_dict() == expected["stats"], label


def test_prepare_from_frozen_vectors_round_trips(tmp_path: Path) -> None:
    attestation = load_attestation(PUBLICATION_VECTORS / "attestation.json")
    destination = tmp_path / "published.nemotrace"
    result = prepare_publication(PUBLICATION_SOURCE, destination, attestation=attestation)
    frozen: dict[str, Any] = cast(
        "dict[str, Any]",
        json.loads((PUBLICATION_VECTORS / "expected-digests.json").read_text(encoding="utf-8")),
    )
    assert result.content_identity == frozen["content_identity"]
    assert result.trace_id == frozen["trace_id"]
    assert result.source.content_identity == frozen["source"]["content_identity"]
    report = verify_archive(destination)
    assert report.ok, report.errors
    assert report.replayability == "playback-only"


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------


def test_scan_projects_audit_archive_without_writing(tmp_path: Path) -> None:
    projection = scan_publication(AUDIT_FIXTURE)
    assert projection.ok
    assert sorted(projection.entries) == sorted(RECORDS)
    assert projection.source.profile == "audit"
    assert projection.trace_id != projection.source.trace_id
    assert projection.projection_sha256.startswith("sha256:")
    manifest: dict[str, Any] = cast(
        "dict[str, Any]",
        parse_json_strict(projection.entries["manifest.json"].decode()),
    )
    assert manifest["capture"] == {
        "profile": "publication",
        "vault_present": False,
        "publication_eligible": True,
        "redaction_policy": "publication-v1",
        "scanner": {"status": "passed", "ruleset": "secrets-v1"},
        "attested": True,
    }
    assert manifest["trace_id"] == projection.trace_id
    source_manifest: dict[str, Any] = cast(
        "dict[str, Any]",
        parse_json_strict(read_archive_entries(AUDIT_FIXTURE)["manifest.json"].decode()),
    )
    assert manifest["workflow"] == source_manifest["workflow"]
    assert manifest["status"] == source_manifest["status"]


def test_projection_is_deterministic_regardless_of_source_path(tmp_path: Path) -> None:
    first = tmp_path / "a" / "run.nemotrace"
    second = tmp_path / "b" / "run.nemotrace"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(AUDIT_FIXTURE.read_bytes())
    second.write_bytes(AUDIT_FIXTURE.read_bytes())
    left = scan_publication(first)
    right = scan_publication(second)
    assert left.trace_id == right.trace_id
    assert left.projection_sha256 == right.projection_sha256
    assert left.entries == right.entries
    assert left.content_identity == right.content_identity


def test_projection_drops_tool_names_and_opaque_paths(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")
    default = scan_publication(source)
    assert default.stats.tool_names_removed == 16
    assert default.stats.tool_names_retained == 0
    assert default.stats.paths_opaque == 1
    ledger = default.entries["public/events.ndjson"]
    for name in (b"reader", b"writer", b"runner", b"fetcher", b"shelltool"):
        assert name not in ledger, name
    assert b"$workspace/f.txt" not in ledger
    assert b"path-1" in ledger
    allowlisted = scan_publication(
        source, options=PublicationOptions(allow_tool_names=("reader", "writer"))
    )
    assert allowlisted.stats.tool_names_retained > 0
    assert b"reader" in allowlisted.entries["public/events.ndjson"]
    assert allowlisted.projection_sha256 != default.projection_sha256
    keep_paths = scan_publication(source, options=PublicationOptions(keep_relative_paths=True))
    assert keep_paths.stats.paths_opaque == 0
    assert b"$workspace/f.txt" in keep_paths.entries["public/events.ndjson"]
    assert keep_paths.projection_sha256 != default.projection_sha256


def test_publication_rebinds_every_ledger_record_to_the_fresh_id(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")
    projection = scan_publication(source)
    for line in projection.entries["public/events.ndjson"].split(b"\n"):
        if not line.strip():
            continue
        record: dict[str, Any] = cast("dict[str, Any]", parse_json_strict(line.decode()))
        assert record["run_id"] == projection.trace_id
    assert projection.trace_id != projection.source.trace_id


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_replay_source_is_refused() -> None:
    with pytest.raises(PublicationError, match="requires an 'audit' source"):
        scan_publication(REPLAY_FIXTURE)


def test_incomplete_provenance_is_refused() -> None:
    with pytest.raises(PublicationError, match="complete compiler provenance"):
        scan_publication(CVXPYGEN_FIXTURE)


def test_interrupted_status_is_refused(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")
    interrupted = _rebuild_archive(
        source,
        tmp_path / "interrupted.nemotrace",
        lambda document: document["manifest"].update(status="interrupted"),
    )
    with pytest.raises(PublicationError, match="complete or failed source run"):
        scan_publication(interrupted)


def test_failed_status_is_publishable(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")
    failed = _rebuild_archive(
        source,
        tmp_path / "failed.nemotrace",
        lambda document: document["manifest"].update(status="failed"),
    )
    projection = scan_publication(failed)
    assert projection.source.status == "failed"
    assert projection.ok


def test_unverifiable_source_is_refused(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")
    tampered = tmp_path / "tampered.nemotrace"
    entries = read_archive_entries(source)
    entries["public/events.ndjson"] += b"\n"
    write_trace_archive(tampered, entries)
    with pytest.raises(PublicationError, match="does not verify"):
        scan_publication(tampered)


def test_ledger_value_that_cannot_be_republished_blocks_export(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")

    def poison(document: dict[str, Any]) -> None:
        first = cast("dict[str, Any]", cast("list[Any]", document["events"])[0])
        cast("dict[str, Any]", first["metadata"])["workflow_id"] = "/home/alice/private-run"

    poisoned = _rebuild_archive(source, tmp_path / "poisoned.nemotrace", poison)
    projection = scan_publication(poisoned)
    assert not projection.ok
    assert any("home_path" in finding for finding in projection.findings)
    attestation = _attest(projection, PublicationOptions())
    with pytest.raises(PublicationError, match="scanner blocked export"):
        prepare_publication(poisoned, tmp_path / "out.nemotrace", attestation=attestation)
    assert not (tmp_path / "out.nemotrace").exists()


def test_destination_must_differ_from_source(tmp_path: Path) -> None:
    source = tmp_path / "run.nemotrace"
    source.write_bytes(AUDIT_FIXTURE.read_bytes())
    projection = scan_publication(source)
    attestation = _attest(projection, PublicationOptions())
    with pytest.raises(PublicationError, match="must differ"):
        prepare_publication(source, source, attestation=attestation)


# ---------------------------------------------------------------------------
# publication-v1 nested allowlists (H-S1)
#
# A forward-compatible reader tolerates unknown fields; the publication
# transform must not. These tests poison a structurally valid audit archive
# with ordinary (non-secret-pattern) private text and prove the transform
# refuses or drops it instead of republishing it.
# ---------------------------------------------------------------------------


def _event(document: dict[str, Any], kind: str, position: int = 0) -> dict[str, Any]:
    matches = [
        cast("dict[str, Any]", event)
        for event in cast("list[Any]", document["events"])
        if cast("dict[str, Any]", event).get("kind") == kind
    ]
    return matches[position]


def test_unknown_metadata_field_is_refused(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")

    def poison(document: dict[str, Any]) -> None:
        metadata = cast("dict[str, Any]", _event(document, "run_started")["metadata"])
        metadata["private_note"] = "patient narrative: seasonal allergies"

    poisoned = _rebuild_archive(source, tmp_path / "poisoned.nemotrace", poison)
    assert verify_archive(poisoned).ok  # structurally valid, forward-compatible
    with pytest.raises(PublicationError, match="refusing to republish unknown structure"):
        scan_publication(poisoned)


def test_out_of_shape_metadata_value_is_refused(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")

    def poison(document: dict[str, Any]) -> None:
        metadata = cast("dict[str, Any]", _event(document, "run_started")["metadata"])
        metadata["duration_ms"] = "fast"  # schema: integer, minimum 0

    poisoned = _rebuild_archive(source, tmp_path / "poisoned.nemotrace", poison)
    assert verify_archive(poisoned).ok
    with pytest.raises(
        PublicationError,
        match="metadata field 'duration_ms' does not match the publication-v1 shape",
    ):
        scan_publication(poisoned)


def test_free_text_output_scalar_is_refused(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")

    def poison(document: dict[str, Any]) -> None:
        output = cast("dict[str, Any]", _event(document, "stage_completed")["output"])
        output["note"] = "patient narrative: seasonal allergies"

    poisoned = _rebuild_archive(source, tmp_path / "poisoned.nemotrace", poison)
    assert verify_archive(poisoned).ok
    with pytest.raises(
        PublicationError, match="output field 'note' does not match the publication-v1 shape"
    ):
        scan_publication(poisoned)


def test_absolute_args_path_is_refused(tmp_path: Path) -> None:
    """`path` must stay alias-relative, even if a reviewer would keep paths."""
    source = _drive_fixture(tmp_path / "src")

    def poison(document: dict[str, Any]) -> None:
        args = cast("dict[str, Any]", _event(document, "tool_call_started")["args"])
        args["path"] = "/home/alice/private/candidate.py"

    poisoned = _rebuild_archive(source, tmp_path / "poisoned.nemotrace", poison)
    assert verify_archive(poisoned).ok
    with pytest.raises(
        PublicationError, match="args field 'path' does not match the publication-v1 shape"
    ):
        scan_publication(poisoned, options=PublicationOptions(keep_relative_paths=True))


def test_unrecognized_args_name_is_dropped_with_a_pointer(tmp_path: Path) -> None:
    """Argument *names* are open tool-domain data: drop them, keep the record."""
    source = _drive_fixture(tmp_path / "src")

    def poison(document: dict[str, Any]) -> None:
        args = cast("dict[str, Any]", _event(document, "tool_call_started")["args"])
        args["private_hint"] = "seasonal allergies"

    poisoned = _rebuild_archive(source, tmp_path / "poisoned.nemotrace", poison)
    projection = scan_publication(poisoned)
    text = projection.entries["public/events.ndjson"].decode("utf-8")
    assert "seasonal allergies" not in text
    records = [
        cast("dict[str, Any]", parse_json_strict(line.decode("utf-8")))
        for line in projection.entries["public/events.ndjson"].split(b"\n")
        if line.strip()
    ]
    started = next(
        record
        for record in records
        if record["kind"] == "tool_call_started"
        and "/args/private_hint" in cast("list[str]", record["redacted_fields"])
    )
    # The location is reported; the name and value are gone from the args.
    assert "private_hint" not in cast("dict[str, Any]", started["args"])


def test_annotation_fields_on_a_plain_record_are_refused(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")

    def poison(document: dict[str, Any]) -> None:
        _event(document, "run_started")["anchor_sequence"] = 1

    poisoned = _rebuild_archive(source, tmp_path / "poisoned.nemotrace", poison)
    assert verify_archive(poisoned).ok
    with pytest.raises(PublicationError, match="carries annotation fields"):
        scan_publication(poisoned)


# ---------------------------------------------------------------------------
# attestation binding
# ---------------------------------------------------------------------------


def _renamed_workflow(document: dict[str, Any]) -> None:
    """Rename the workflow in a canonical entry set (non-blocking change)."""
    manifest = cast("dict[str, Any]", document["manifest"])
    workflow = cast("dict[str, Any]", manifest["workflow"])
    workflow["id"] = str(workflow["id"]) + "Renamed"
    graph = cast("dict[str, Any]", document["graph"])
    graph["workflow_id"] = workflow["id"]
    first = cast("dict[str, Any]", cast("list[Any]", document["events"])[0])
    cast("dict[str, Any]", first["metadata"])["workflow_id"] = workflow["id"]


def test_attestation_is_bound_to_the_reviewed_projection(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")
    projection = scan_publication(source)
    attestation = _attest(projection, PublicationOptions())
    # A source whose reviewed ledger changed is no longer covered: the digest
    # commits to the projected events and graph, so an edited run cannot ride
    # on an earlier review.
    changed = _rebuild_archive(source, tmp_path / "changed.nemotrace", _renamed_workflow)
    with pytest.raises(PublicationError, match="does not cover this projection"):
        prepare_publication(changed, tmp_path / "out.nemotrace", attestation=attestation)
    assert not (tmp_path / "out.nemotrace").exists()
    # The source identity is checked too.
    forged = build_attestation(
        projection,
        options=PublicationOptions(),
        reviewer="Alex Ling",
        license_id="CC-BY-4.0",
        consent=CONSENT,
        reviewed_at="2026-09-12T00:00:00.000Z",
    )
    object.__setattr__(forged, "source_content_identity", "sha256:" + "11" * 32)
    with pytest.raises(PublicationError, match="content identity"):
        prepare_publication(source, tmp_path / "out.nemotrace", attestation=forged)
    # Sanity: the untouched attestation does cover it.
    result = prepare_publication(source, tmp_path / "ok.nemotrace", attestation=attestation)
    assert result.destination.exists()


def test_prepare_refuses_a_projection_with_reviewed_options_that_changed(
    tmp_path: Path,
) -> None:
    """Attestation options are self-consistent, so only real changes can diverge."""
    source = _drive_fixture(tmp_path / "src")
    reviewed = scan_publication(source, options=PublicationOptions(allow_tool_names=("reader",)))
    attestation = _attest(reviewed, PublicationOptions(allow_tool_names=("reader",)))
    result = prepare_publication(source, tmp_path / "ok.nemotrace", attestation=attestation)
    assert result.destination.exists()
    default = prepare_publication(
        source,
        tmp_path / "default.nemotrace",
        attestation=_attest(scan_publication(source), PublicationOptions()),
    )
    assert default.content_identity != result.content_identity
    assert default.destination.read_bytes() != result.destination.read_bytes()


def test_attestation_cannot_cover_a_failed_scan(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")

    def poison(document: dict[str, Any]) -> None:
        first = cast("dict[str, Any]", cast("list[Any]", document["events"])[0])
        cast("dict[str, Any]", first["metadata"])["workflow_id"] = "/home/alice/private-run"

    poisoned = _rebuild_archive(source, tmp_path / "poisoned.nemotrace", poison)
    projection = scan_publication(poisoned)
    report_path = tmp_path / "poisoned-report.json"
    write_publication_report(report_path, projection.report(attested=False, attestation=None))
    report: dict[str, Any] = cast(
        "dict[str, Any]", json.loads(report_path.read_text(encoding="utf-8"))
    )
    with pytest.raises(PublicationError, match="did not pass"):
        attestation_from_report(
            report, reviewer="Alex Ling", license_id="CC-BY-4.0", consent=CONSENT
        )


def test_attestation_document_round_trips(tmp_path: Path) -> None:
    projection = scan_publication(AUDIT_FIXTURE)
    attestation = _attest(projection, PublicationOptions())
    path = write_attestation(tmp_path / "attest.json", attestation)
    loaded = load_attestation(path)
    assert loaded == attestation
    document: dict[str, Any] = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    assert document["format"] == PUBLICATION_ATTESTATION_FORMAT
    assert document["projection"]["sha256"] == projection.projection_sha256
    document["format"] = "nemoir.trace.publication-attestation/9.9"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PublicationError, match="unsupported attestation format"):
        load_attestation(path)


# ---------------------------------------------------------------------------
# prepare: write path
# ---------------------------------------------------------------------------


def test_prepare_writes_verifiable_publication_archive(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")
    projection = scan_publication(source)
    attestation = _attest(projection, PublicationOptions())
    destination = tmp_path / "published" / "run.nemotrace"
    result = prepare_publication(source, destination, attestation=attestation)
    assert result.content_identity == projection.content_identity
    assert result.compressed_bytes > 0
    report = verify_archive(destination)
    assert report.ok, report.errors
    assert report.integrity == "passed"
    assert report.structural == "passed"
    assert report.replayability == "playback-only"
    entries = read_archive_entries(destination)
    assert sorted(entries) == sorted(RECORDS)
    assert "private/vault.enc" not in entries
    manifest: dict[str, Any] = cast(
        "dict[str, Any]", parse_json_strict(entries["manifest.json"].decode())
    )
    assert manifest["capture"]["profile"] == "publication"
    assert manifest["capture"]["attested"] is True
    assert manifest["capture"]["publication_eligible"] is True
    source_manifest: dict[str, Any] = cast(
        "dict[str, Any]",
        parse_json_strict(read_archive_entries(source)["manifest.json"].decode()),
    )
    assert manifest["provenance"] == source_manifest["provenance"]
    assert manifest["workflow"]["ir_sha256"] == source_manifest["workflow"]["ir_sha256"]
    # Seeded credential from the source run must not appear anywhere.
    for path, data in entries.items():
        assert b"sk-parity-TEST-secret-9999" not in data, path
        assert b"/work/" not in data, path
    # The disclosure report is a sidecar, never an archive entry.
    assert result.report_path == publication_report_path(destination)
    sidecar: dict[str, Any] = cast(
        "dict[str, Any]", json.loads(result.report_path.read_text(encoding="utf-8"))
    )
    assert sidecar["format"] == PUBLICATION_REPORT_FORMAT
    assert sidecar["attested"] is True
    assert sidecar["scan"] == {
        "ruleset": "secrets-v1",
        "status": "passed",
        "findings": [],
        "findings_count": 0,
    }
    assert sidecar["publication"]["content_identity"] == result.content_identity
    assert sidecar["attestation"]["reviewer"] == "Alex Ling"


def test_prepare_is_byte_deterministic(tmp_path: Path) -> None:
    source = _drive_fixture(tmp_path / "src")
    projection = scan_publication(source)
    attestation = _attest(projection, PublicationOptions())
    first = prepare_publication(source, tmp_path / "one.nemotrace", attestation=attestation)
    second = prepare_publication(source, tmp_path / "two.nemotrace", attestation=attestation)
    assert first.destination.read_bytes() == second.destination.read_bytes()
    assert first.content_identity == second.content_identity


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_publication_flow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _drive_fixture(tmp_path / "src")
    report = tmp_path / "report.json"
    code, out = _run(capsys, ["scan-publication", str(source), "--report", str(report)])
    assert code == 0
    assert "scan: passed" in out
    assert "result: ok" in out
    assert f"report: {report.name}" in out
    assert report.exists()
    attestation_path = tmp_path / "attest.json"
    code, out = _run(
        capsys,
        [
            "attest-publication",
            "--report",
            str(report),
            "--reviewer",
            "Alex Ling",
            "--license",
            "CC-BY-4.0",
            "--consent",
            CONSENT,
            "--out",
            str(attestation_path),
        ],
    )
    assert code == 0
    assert "result: ok" in out
    assert attestation_path.exists()
    destination = tmp_path / "published.nemotrace"
    code, out = _run(
        capsys,
        [
            "prepare-publication",
            str(source),
            str(destination),
            "--attest",
            str(attestation_path),
        ],
    )
    assert code == 0
    assert "attested: true" in out
    assert "result: ok" in out
    assert verify_archive(destination).ok


def test_cli_refuses_replay_source(capsys: pytest.CaptureFixture[str]) -> None:
    code, err = _run_err(capsys, ["scan-publication", str(REPLAY_FIXTURE)])
    assert code == 1
    assert "requires an 'audit' source" in err


def test_cli_usage_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, err = _run_err(capsys, ["scan-publication", str(tmp_path / "missing.nemotrace")])
    assert code == 2
    assert "not found" in err
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["prepare-publication", str(AUDIT_FIXTURE), str(tmp_path / "x.nemotrace")])
    assert excinfo.value.code == 2  # --attest is required
    code, err = _run_err(
        capsys, ["scan-publication", str(AUDIT_FIXTURE), "--allow-tool-name", "not/a name"]
    )
    assert code == 2
    assert "not a safe static declaration" in err


def test_cli_prepare_refuses_stale_attestation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _drive_fixture(tmp_path / "src")
    projection = scan_publication(source)
    attestation_path = write_attestation(
        tmp_path / "attest.json", _attest(projection, PublicationOptions())
    )
    # The attested run is later edited before publishing.
    edited = _rebuild_archive(source, tmp_path / "edited.nemotrace", _renamed_workflow)
    code, err = _run_err(
        capsys,
        [
            "prepare-publication",
            str(edited),
            str(tmp_path / "out.nemotrace"),
            "--attest",
            str(attestation_path),
        ],
    )
    assert code == 1
    assert "does not cover this projection" in err
    assert not (tmp_path / "out.nemotrace").exists()


def test_cvxpygen_publication_fixture_is_attested_and_clean() -> None:
    """The reviewed showcase fixture: publication profile, attested, secret-free."""
    fixture = PUBLICATION_VECTORS / "cvxpygen-publication.nemotrace"
    if not fixture.exists():
        pytest.skip("publication fixture requires the meta checkout")
    report = verify_archive(fixture)
    assert report.ok, report.errors
    assert report.replayability == "playback-only"
    entries = read_archive_entries(fixture)
    manifest: dict[str, Any] = cast(
        "dict[str, Any]", parse_json_strict(entries["manifest.json"].decode())
    )
    assert manifest["capture"] == {
        "profile": "publication",
        "vault_present": False,
        "publication_eligible": True,
        "redaction_policy": "publication-v1",
        "scanner": {"status": "passed", "ruleset": "secrets-v1"},
        "attested": True,
    }
    assert manifest["provenance"]["complete"] is True
    assert "private/vault.enc" not in entries
    assert fixture.stat().st_size < 8 * 1024 * 1024
    for name, data in entries.items():
        assert b"/home/" not in data, name
        assert b"/Users/" not in data, name
        assert b"sk-" not in data, name
        assert b"BEGIN RSA PRIVATE KEY" not in data, name
        assert b"run_harness" not in data, name
        assert b"read_file" not in data, name
    ledger = [
        cast("dict[str, Any]", parse_json_strict(line.decode()))
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip()
    ]
    trials = [event for event in ledger if event["kind"] == "annotation"]
    assert [event["annotation"]["payload"]["verdict"] for event in trials] == [
        "rejected",
        "rejected",
        "accepted",
    ]
    assert all("tool_name" not in event for event in ledger)
    # The catalog entry must describe this exact artifact.
    catalog: dict[str, Any] = cast(
        "dict[str, Any]", json.loads((ROOT / "docs" / "trace" / "catalog.json").read_text())
    )
    entry = catalog["entries"][0]
    integrity: dict[str, Any] = cast(
        "dict[str, Any]", parse_json_strict(entries["integrity.json"].decode())
    )
    assert entry["artifact"]["content_identity"] == integrity["content_identity"]
    assert entry["artifact"]["bytes"] == fixture.stat().st_size
    assert entry["artifact"]["ir_sha256"] == manifest["workflow"]["ir_sha256"]
    assert entry["artifact"]["profile"] == "publication"
    assert entry["license"]
    assert entry["attestation"]["consent"]
