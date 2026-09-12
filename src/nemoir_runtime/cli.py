"""``nemotrace`` verifier CLI: integrity, structural, semantic, and replay checks.

Usage::

    nemotrace verify <archive> [--unlock SRC | --replay SRC]
    nemotrace scan-publication <archive> [--allow-tool-name NAME]... [--keep-relative-paths]
    nemotrace attest-publication --report PATH --reviewer NAME --license ID --consent TEXT
    nemotrace prepare-publication <archive> <destination> --attest PATH

``SRC`` is a passphrase source: ``env:VAR`` | ``file:PATH`` | ``prompt``.
Plain ``verify`` reports the public archive levels (integrity, structural,
replayability). ``--unlock`` additionally decrypts the replay vault and
evaluates semantic evidence completeness. ``--replay`` unlocks and
taped-replays the recorded path, reporting ``matched``/``diverged`` plus
divergence strings.

The command prints a stable, line-oriented report on stdout and returns
``0`` when the requested level passed, ``1`` when it failed, and ``2`` for
usage errors (missing archive, bad passphrase source). It never prints
passphrase values, vault plaintext, or stack traces. This is the
human/CI-runnable surface over the same library reports the viewer uses;
see ``docs/trace/schema/test-vectors/cli/`` for frozen golden outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

from nemoir_runtime.publication import (
    PublicationAttestation,
    PublicationError,
    PublicationOptions,
    PublicationProjection,
    PublicationResult,
    attestation_from_report,
    load_attestation,
    prepare_publication,
    publication_report_path,
    scan_publication,
    write_attestation,
    write_publication_report,
)
from nemoir_runtime.replay import ReplayReport, replay_trace
from nemoir_runtime.trace import (
    TraceError,
    VerificationReport,
    read_archive_entries,
    unlock_archive,
    verify_archive,
)

_EXIT_OK = 0
_EXIT_FAILED = 1
_EXIT_USAGE = 2
_MAX_DIAGNOSTICS = 10
_MISSING = "-"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemotrace",
        description="Verify a NemoTrace archive (integrity, structural, semantic, replay).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser(
        "verify",
        help="verify one .nemotrace archive",
        description=(
            "Verify one .nemotrace archive. Without flags only the public levels "
            "(integrity/structural/replayability) are evaluated; --unlock adds semantic "
            "vault evidence; --replay re-executes the recorded path with fixtures only."
        ),
    )
    verify.add_argument("archive", help="path to a .nemotrace archive")
    sources = verify.add_mutually_exclusive_group()
    sources.add_argument(
        "--unlock",
        metavar="SRC",
        help="unlock the replay vault: env:VAR | file:PATH | prompt",
    )
    sources.add_argument(
        "--replay",
        metavar="SRC",
        help="unlock and taped-replay: env:VAR | file:PATH | prompt",
    )
    _add_publication_parsers(subparsers)
    return parser


def _add_publication_option_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-tool-name",
        action="append",
        default=[],
        metavar="NAME",
        help="retain this static tool declaration after review (repeatable)",
    )
    parser.add_argument(
        "--keep-relative-paths",
        action="store_true",
        help="keep alias-relative paths instead of opaque path-N refs (requires review)",
    )


def _add_publication_parsers(subparsers: Any) -> None:
    scan = subparsers.add_parser(
        "scan-publication",
        help="project an audit archive for publication review (writes a report)",
        description=(
            "Project one audit profile archive under the publication-v1 policy and "
            "write a disclosure report. Nothing is published: the report shows the "
            "projection digest, the scan result, and what the transform drops. "
            "Attest with attest-publication, then write with prepare-publication."
        ),
    )
    scan.add_argument("archive", help="path to an audit .nemotrace archive")
    _add_publication_option_flags(scan)
    scan.add_argument(
        "--report",
        metavar="PATH",
        help="where to write the disclosure report (default: <archive>.publication-report.json)",
    )
    attest = subparsers.add_parser(
        "attest-publication",
        help="sign the disclosure report you reviewed",
        description=(
            "Read one disclosure report and bind the reviewer, license, and consent "
            "statement to the projection digest it contains. Cannot attest a failed "
            "scan, and cannot invent a digest."
        ),
    )
    attest.add_argument(
        "--report", required=True, metavar="PATH", help="disclosure report to attest"
    )
    attest.add_argument("--reviewer", required=True, metavar="NAME", help="reviewer name")
    attest.add_argument(
        "--license", required=True, metavar="ID", help="license for the published trace"
    )
    attest.add_argument(
        "--consent", required=True, metavar="TEXT", help="consent/attestation statement"
    )
    attest.add_argument(
        "--out",
        metavar="PATH",
        help="where to write the attestation (default: <report>.attestation.json)",
    )
    attest.add_argument(
        "--reviewed-at",
        metavar="TIMESTAMP",
        help="fixed review timestamp (default: now); use for reproducible attestations",
    )
    prepare = subparsers.add_parser(
        "prepare-publication",
        help="write one attested, vault-free publication archive",
        description=(
            "Re-project the source, re-run the blocking scan, and write the "
            "publication archive plus its disclosure report. Refuses a source that "
            "is not an audit archive, a projection the attestation does not cover, "
            "or any scanner finding."
        ),
    )
    prepare.add_argument("archive", help="path to an audit .nemotrace archive")
    prepare.add_argument("destination", help="path to write the publication archive")
    prepare.add_argument(
        "--attest", required=True, metavar="PATH", help="attestation written by attest-publication"
    )
    prepare.add_argument(
        "--report",
        metavar="PATH",
        help=(
            "where to write the disclosure report "
            "(default: <destination>.publication-report.json)"
        ),
    )


def _resolve_passphrase(spec: str) -> str:
    """Resolve ``env:VAR`` | ``file:PATH`` | ``prompt`` into a passphrase.

    Mirrors the demo launcher's resolution rules (`.strip()` for files,
    empty values refused). Raises ``ValueError`` with a human-readable
    message; the CLI maps that to a usage error without echoing the value.
    """
    if spec == "prompt":
        try:
            value = getpass.getpass("vault passphrase: ")
        except EOFError as exc:
            msg = "passphrase prompt could not read input (use env:VAR or file:PATH)"
            raise ValueError(msg) from exc
        if not value:
            msg = "passphrase prompt supplied an empty passphrase"
            raise ValueError(msg)
        return value
    if spec.startswith("env:"):
        var = spec[len("env:") :]
        if not var:
            msg = "env: requires a variable name (env:VAR)"
            raise ValueError(msg)
        value = os.environ.get(var)
        if not value:
            msg = f"{spec}: environment variable is missing or empty"
            raise ValueError(msg)
        return value
    if spec.startswith("file:"):
        raw = spec[len("file:") :]
        if not raw:
            msg = "file: requires a path (file:PATH)"
            raise ValueError(msg)
        try:
            value = Path(raw).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            msg = f"{spec}: cannot read file ({exc})"
            raise ValueError(msg) from exc
        if not value:
            msg = f"{spec}: file is empty"
            raise ValueError(msg)
        return value
    msg = f"{spec!r}: expected env:VAR | file:PATH | prompt"
    raise ValueError(msg)


def _read_entries(archive: Path) -> dict[str, bytes] | None:
    """Best-effort archive read for metadata; verification runs separately."""
    try:
        return read_archive_entries(archive)
    except (TraceError, OSError, ValueError):
        return None


def _read_manifest(entries: Mapping[str, bytes] | None) -> dict[str, Any]:
    if not entries:
        return {}
    raw = entries.get("manifest.json")
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return cast("dict[str, Any]", parsed)


def _dict_field(node: Any, key: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    mapping = cast("dict[Any, Any]", node)
    value = mapping.get(key)
    if not isinstance(value, dict):
        return {}
    return cast("dict[str, Any]", value)


def _str_field(node: Any, key: str) -> str:
    if not isinstance(node, dict):
        return _MISSING
    mapping = cast("dict[Any, Any]", node)
    value = mapping.get(key)
    return value if isinstance(value, str) and value else _MISSING


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _diagnostic_lines(label: str, values: Iterable[str]) -> list[str]:
    items = [_single_line(str(value)) for value in values]
    lines = [f"{label}: {item}" for item in items[:_MAX_DIAGNOSTICS]]
    if len(items) > _MAX_DIAGNOSTICS:
        lines.append(f"{label}_more: {len(items) - _MAX_DIAGNOSTICS}")
    return lines


def _fallback_report() -> VerificationReport:
    return VerificationReport(
        ok=False,
        content_identity=None,
        warnings=(),
        errors=("verification could not run",),
        integrity="failed",
        structural="failed",
        semantic="failed",
        replayability="none",
    )


def _unlock_report(archive: Path, passphrase: str) -> VerificationReport:
    """Levels for a replay that raised: unlock for semantic, else public verify."""
    try:
        _records, report = unlock_archive(archive, passphrase)
    except (TraceError, OSError, ValueError):
        try:
            return verify_archive(archive)
        except (TraceError, OSError, ValueError):
            return _fallback_report()
    return report


def _verify_lines(
    archive: Path,
    manifest: Mapping[str, Any],
    report: VerificationReport,
    *,
    replay_requested: bool,
    replay_report: ReplayReport | None,
    replay_failed: bool,
) -> tuple[list[str], bool]:
    workflow = _dict_field(manifest, "workflow")
    capture = _dict_field(manifest, "capture")
    errors = list(report.errors)
    if replay_failed:
        errors.append("taped replay could not run")
    # Replay status/divergence lines use the ReplayReport when available.
    if replay_failed:
        replay_state = "error"
    elif replay_report is not None and replay_report.matched:
        replay_state = "matched"
    else:
        replay_state = "diverged"
    lines = [
        f"archive: {archive.name}",
        f"format: {_str_field(manifest, 'format')}",
        f"trace_id: {_str_field(manifest, 'trace_id')}",
        f"workflow: {_str_field(workflow, 'id')}",
        f"profile: {_str_field(capture, 'profile')}",
        f"status: {_str_field(manifest, 'status')}",
        f"content_identity: {report.content_identity or _MISSING}",
        f"integrity: {report.integrity}",
        f"structural: {report.structural}",
        f"semantic: {report.semantic}",
        f"replayability: {report.replayability}",
        f"warnings: {len(report.warnings)}",
        f"errors: {len(errors)}",
    ]
    if replay_requested:
        lines += [
            f"replay: {replay_state}",
            f"replay_status: "
            f"{replay_report.replayed_status if replay_report is not None else 'unknown'}",
            f"replay_steps: {replay_report.steps if replay_report is not None else 0}",
            f"divergences: {len(replay_report.divergences) if replay_report is not None else 0}",
        ]
    lines += _diagnostic_lines("warning", report.warnings)
    lines += _diagnostic_lines("error", errors)
    if replay_requested and replay_report is not None:
        lines += _diagnostic_lines("divergence", replay_report.divergences)
    result_ok = (
        (replay_report is not None and replay_report.matched) if replay_requested else report.ok
    )
    lines.append(f"result: {'ok' if result_ok else 'failed'}")
    return lines, result_ok


def _verify_command(args: argparse.Namespace) -> int:
    archive = Path(cast("str", args.archive))
    if not archive.exists():
        print(f"error: archive not found: {archive}", file=sys.stderr)
        return _EXIT_USAGE
    if not archive.is_file():
        print(f"error: archive is not a file: {archive}", file=sys.stderr)
        return _EXIT_USAGE
    manifest = _read_manifest(_read_entries(archive))
    unlock_spec = cast("str | None", args.unlock)
    replay_spec = cast("str | None", args.replay)
    replay_requested = replay_spec is not None
    spec = replay_spec or unlock_spec
    passphrase: str | None = None
    if spec is not None:
        try:
            passphrase = _resolve_passphrase(spec)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return _EXIT_USAGE
    replay_report: ReplayReport | None = None
    replay_failed = False
    if spec is None:
        report = verify_archive(archive)
    elif passphrase is None:  # pragma: no cover - resolver raises instead
        return _EXIT_USAGE
    elif replay_requested:
        try:
            replay_report = asyncio.run(replay_trace(archive, passphrase))
        except Exception:  # unexpected replay failure is reported, not raised
            replay_failed = True
            report = _unlock_report(archive, passphrase)
        else:
            report = replay_report.verification
    else:
        try:
            _records, report = unlock_archive(archive, passphrase)
        except Exception:  # unexpected unlock failure is reported, not raised
            report = _fallback_report()
    lines, result_ok = _verify_lines(
        archive,
        manifest,
        report,
        replay_requested=replay_requested,
        replay_report=replay_report,
        replay_failed=replay_failed,
    )
    for line in lines:
        print(line)
    return _EXIT_OK if result_ok else _EXIT_FAILED


def _publication_options(args: argparse.Namespace) -> PublicationOptions:
    try:
        return PublicationOptions(
            allow_tool_names=tuple(cast("list[str]", args.allow_tool_name)),
            keep_relative_paths=bool(args.keep_relative_paths),
        )
    except PublicationError as exc:
        raise ValueError(str(exc)) from exc


def _require_archive(path: Path, what: str = "archive") -> str | None:
    """Shared existence check; returns an error message when unusable."""
    if not path.exists():
        return f"{what} not found: {path}"
    if not path.is_file():
        return f"{what} is not a file: {path}"
    return None


def _scan_lines(projection: PublicationProjection, report: Path | None) -> list[str]:
    stats = projection.stats
    lines = [
        f"archive: {projection.source.archive}",
        f"trace_id: {projection.source.trace_id}",
        f"content_identity: {projection.source.content_identity or _MISSING}",
        f"profile: {projection.source.profile}",
        f"status: {projection.source.status}",
        f"events: {stats.event_count}",
        f"stage_visits: {stats.stage_visit_count}",
        f"tool_names_removed: {stats.tool_names_removed}",
        f"tool_names_retained: {stats.tool_names_retained}",
        f"paths_opaque: {stats.paths_opaque}",
        f"projection_sha256: {projection.projection_sha256}",
        f"predicted_content_identity: {projection.content_identity}",
        f"scan: {'passed' if projection.ok else 'failed'}",
        f"findings: {len(projection.findings)}",
    ]
    lines += _diagnostic_lines("finding", projection.findings)
    lines.append(f"report: {report.name if report is not None else _MISSING}")
    lines.append(f"result: {'ok' if projection.ok else 'failed'}")
    return lines


def _scan_publication_command(args: argparse.Namespace) -> int:
    archive = Path(cast("str", args.archive))
    problem = _require_archive(archive)
    if problem is not None:
        print(f"error: {problem}", file=sys.stderr)
        return _EXIT_USAGE
    try:
        options = _publication_options(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_USAGE
    try:
        projection = scan_publication(archive, options=options)
    except (PublicationError, TraceError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_FAILED
    report_path = (
        Path(cast("str", args.report))
        if args.report is not None
        else publication_report_path(archive)
    )
    try:
        write_publication_report(
            report_path, projection.report(attested=False, attestation=None)
        )
    except OSError as exc:
        print(f"error: cannot write report: {exc}", file=sys.stderr)
        return _EXIT_FAILED
    for line in _scan_lines(projection, report_path):
        print(line)
    return _EXIT_OK if projection.ok else _EXIT_FAILED


def _attest_publication_command(args: argparse.Namespace) -> int:
    report_path = Path(cast("str", args.report))
    problem = _require_archive(report_path, "report")
    if problem is not None:
        print(f"error: {problem}", file=sys.stderr)
        return _EXIT_USAGE
    try:
        parsed: Any = json.loads(report_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"error: report is not valid JSON: {exc}", file=sys.stderr)
        return _EXIT_FAILED
    if not isinstance(parsed, dict):
        print("error: report must be a JSON object", file=sys.stderr)
        return _EXIT_FAILED
    try:
        attestation = attestation_from_report(
            cast("dict[str, Any]", parsed),
            reviewer=cast("str", args.reviewer),
            license_id=cast("str", args.license),
            consent=cast("str", args.consent),
            reviewed_at=cast("str | None", args.reviewed_at),
        )
    except PublicationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_FAILED
    out_path = (
        Path(cast("str", args.out))
        if args.out is not None
        else report_path.with_name(report_path.name + ".attestation.json")
    )
    try:
        write_attestation(out_path, attestation)
    except OSError as exc:
        print(f"error: cannot write attestation: {exc}", file=sys.stderr)
        return _EXIT_FAILED
    lines = [
        f"report: {report_path.name}",
        f"projection_sha256: {attestation.projection_sha256}",
        f"source_trace_id: {attestation.source_trace_id}",
        f"reviewer: {attestation.reviewer}",
        f"license: {attestation.license}",
        f"reviewed_at: {attestation.reviewed_at}",
        f"attestation: {out_path.name}",
        "result: ok",
    ]
    for line in lines:
        print(line)
    return _EXIT_OK


def _prepare_publication_command(args: argparse.Namespace) -> int:
    archive = Path(cast("str", args.archive))
    problem = _require_archive(archive)
    if problem is not None:
        print(f"error: {problem}", file=sys.stderr)
        return _EXIT_USAGE
    try:
        attestation: PublicationAttestation = load_attestation(
            Path(cast("str", args.attest))
        )
    except PublicationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_FAILED
    report_path = Path(cast("str", args.report)) if args.report is not None else None
    try:
        result: PublicationResult = prepare_publication(
            archive,
            Path(cast("str", args.destination)),
            attestation=attestation,
            report_path=report_path,
        )
    except (PublicationError, TraceError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_FAILED
    lines = [
        f"archive: {archive.name}",
        f"publication: {result.destination.name}",
        f"trace_id: {result.trace_id}",
        f"projection_sha256: {result.projection_sha256}",
        "scan: passed",
        "attested: true",
        f"reviewer: {attestation.reviewer}",
        f"license: {attestation.license}",
        f"events: {result.stats.event_count}",
        f"content_identity: {result.content_identity}",
        f"report: {result.report_path.name}",
        "result: ok",
    ]
    for line in lines:
        print(line)
    return _EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``nemotrace`` and ``python -m nemoir_runtime``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "scan-publication":
        return _scan_publication_command(args)
    if command == "attest-publication":
        return _attest_publication_command(args)
    if command == "prepare-publication":
        return _prepare_publication_command(args)
    return _verify_command(args)
