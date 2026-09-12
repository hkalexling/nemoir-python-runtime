"""``nemotrace`` verifier CLI: integrity, structural, semantic, and replay checks.

Usage::

    nemotrace verify <archive> [--unlock SRC | --replay SRC]

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
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``nemotrace`` and ``python -m nemoir_runtime``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _verify_command(args)
