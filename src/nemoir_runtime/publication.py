"""NemoTrace publication transform (Phase 5).

Turns one ``audit`` trace into one stricter, vault-free ``publication``
artifact (``plan.md`` §5.4, ``redaction-policy.md`` §1). Publication is a
fresh projection, never a bit flipped on an existing archive:

1. the source must verify, be an ``audit`` profile with no vault, carry
   complete compiler provenance, and be ``complete`` or ``failed``;
2. the public ledger is re-projected under ``publication-v1`` — tool names
   are dropped unless explicitly reviewed, and alias-relative paths become
   opaque ``path-N`` refs;
3. the artifact gets a fresh trace id derived from the projection digest, so
   it does not correlate with the local private run (``phase-0-decisions.md``
   §5);
4. a human reviews a disclosure report (``scan_publication``) and signs an
   attestation covering the reviewed projection digest; only then may
   ``prepare_publication`` write the archive;
5. the ``secrets-v1`` scanner runs as a blocking gate over every cleartext
   entry and filename, and the output never contains a vault.

Publication reduces risk; it cannot prove that reviewed scalar text is
non-sensitive (``redaction-policy.md`` §12). The disclosure report states the
retained identifier classes so a reviewer can judge that. Reports and
attestations are local review artifacts: they are written next to the
archives and are not part of the published bundle.

This module is stdlib-only and mirrors the TypeScript implementation in
``web/nemoir-runtime/src/publication.ts`` byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nemoir_runtime.canonical import parse_json_strict, sha256_tag, to_canonical_bytes
from nemoir_runtime.trace import (
    CONTENT_IDENTITY_FORMAT,
    EVENTS_PATH,
    GRAPH_PATH,
    INTEGRITY_PATH,
    MANIFEST_PATH,
    MAX_SAFE_INT,
    MIN_SAFE_INT,
    SCANNER_RULESET,
    SUMMARY_FORMAT,
    SUMMARY_PATH,
    TRACE_FORMAT,
    TraceError,
    format_timestamp,
    read_archive_entries,
    scan_cleartext_entries,
    verify_archive,
    write_trace_archive,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PUBLICATION_PROJECTION_FORMAT = "nemoir.trace.publication-projection/0.1"
PUBLICATION_ATTESTATION_FORMAT = "nemoir.trace.publication-attestation/0.1"
PUBLICATION_REPORT_FORMAT = "nemoir.trace.publication-report/0.1"
PUBLICATION_ID_DOMAIN = b"nemoir.trace.publication-id/0.1\x00"
PUBLICATION_REDACTION_POLICY = "publication-v1"

# Source gates. Publication accepts only the default redacted capture, with
# no vault, and refuses interrupted runs (partial evidence).
SOURCE_PROFILE = "audit"
PUBLISHABLE_STATUSES = ("complete", "failed")

_GRAPH_FORMAT = "nemoir.trace.workflow-graph/0.1"

_LEDGER_KINDS = frozenset(
    {
        "run_started",
        "stage_started",
        "model_delta",
        "model_completed",
        "model_retry",
        "tool_call_started",
        "tool_call_completed",
        "tool_call_failed",
        "policy_checked",
        "policy_denied",
        "transition_selected",
        "stage_completed",
        "run_completed",
        "run_failed",
        "annotation",
    }
)

# Strict writer allowlists. Unknown fields are refused rather than dropped:
# a source written by a newer tool must not be silently republished.
_MANIFEST_KEYS = frozenset(
    {
        "format",
        "trace_id",
        "created_at",
        "status",
        "capture",
        "workflow",
        "provenance",
        "integrity",
        "viewer",
    }
)
_CAPTURE_KEYS = frozenset(
    {
        "profile",
        "vault_present",
        "publication_eligible",
        "redaction_policy",
        "scanner",
        "attested",
    }
)
_WORKFLOW_KEYS = frozenset({"id", "ir_version", "ir_sha256", "entry", "exits"})
_PROVENANCE_KEYS = frozenset(
    {"complete", "frontend", "target", "compiler_version", "runtime", "model"}
)
# ``model`` is optional in manifest.schema.json; the rest are required.
_REQUIRED_PROVENANCE_KEYS = frozenset(
    {"complete", "frontend", "target", "compiler_version", "runtime"}
)
_RUNTIME_KEYS = frozenset({"name", "version"})
_MODEL_KEYS = frozenset({"name", "api_mode", "sampling"})
_SAMPLING_KEYS = frozenset({"temperature", "max_tokens"})
_VIEWER_KEYS = frozenset({"min_format"})
_GRAPH_KEYS = frozenset(
    {"format", "workflow_id", "entry", "exits", "nodes", "transitions", "policies"}
)
_NODE_KEYS = frozenset({"id", "execution", "capabilities", "writes"})
_WRITE_KEYS = frozenset({"name", "type", "optional"})
_TRANSITION_KEYS = frozenset({"from", "to", "priority", "guard_kind"})
_POLICY_KEYS = frozenset({"ref", "kind", "trigger_capability", "required_capabilities"})
_ATTESTATION_KEYS = frozenset(
    {
        "format",
        "source",
        "projection",
        "options",
        "reviewer",
        "reviewed_at",
        "license",
        "consent",
    }
)
_ATTESTATION_SOURCE_KEYS = frozenset({"content_identity", "trace_id"})
_ATTESTATION_PROJECTION_KEYS = frozenset({"sha256"})
_OPTIONS_KEYS = frozenset({"allow_tool_names", "keep_relative_paths"})

# Strict publication-v1 allowlists. A forward-compatible *reader* tolerates
# unknown fields (schema README §1); the publication transform must not,
# because a published artifact asserts what it contains. Sets mirror
# ``docs/trace/schema/public-event.schema.json`` exactly.
_LEDGER_KEYS = frozenset(
    {
        "kind",
        "run_id",
        "sequence",
        "timestamp",
        "stage_id",
        "stage_visit_id",
        "model_call_id",
        "tool_call_id",
        "channel",
        "text",
        "capability",
        "tool_name",
        "args",
        "output",
        "result",
        "error",
        "transition_to",
        "metadata",
        "redacted_fields",
        "anchor_sequence",
        "annotation",
    }
)
_METADATA_KEYS = frozenset(
    {
        "approval",
        "attempt",
        "byte_count",
        "category",
        "code_bytes",
        "command_id",
        "content_omitted",
        "cost_usd",
        "denied",
        "duration_ms",
        "entry",
        "error_code",
        "error_type",
        "exit_class",
        "input_bytes",
        "input_tokens",
        "max_retries",
        "method",
        "origin_class",
        "output_tokens",
        "path_ref",
        "policy_kind",
        "policy_ref",
        "priority",
        "reason",
        "required_capabilities",
        "response_bytes",
        "result_status",
        "result_type",
        "root_class",
        "status_class",
        "step_count",
        "total_tokens",
        "value_type",
        "workflow_id",
    }
)
# ``args`` names are open tool-domain data: the audit writer mirrors arbitrary
# argument names as markers for ``user.*`` / ``browser.*`` capabilities, so an
# unrecognized name is dropped with a redaction pointer instead of refusing
# an otherwise-valid source.
_PUBLIC_ARGS_KEYS = frozenset(
    {
        "path",
        "path_ref",
        "root_class",
        "content",
        "command_id",
        "method",
        "origin_class",
        "key",
        "value",
        "code",
        "input",
        "headers",
        "body",
        "question",
        "message",
        "options",
    }
)
_ARGS_MARKER_KEYS = frozenset(
    {
        "content",
        "key",
        "value",
        "code",
        "input",
        "headers",
        "body",
        "question",
        "message",
        "options",
    }
)
_METADATA_BOOLS = frozenset({"approval", "content_omitted", "denied"})
_METADATA_ENUMS: dict[str, frozenset[str]] = {
    "exit_class": frozenset({"success", "failure", "signal", "timeout"}),
    "policy_kind": frozenset({"before", "deny"}),
    "value_type": frozenset({"string", "number", "boolean", "object", "array", "null", "binary"}),
    "reason": frozenset(
        {
            "explicit_transition",
            "backward_ref_loop",
            "next_stage_required_input_available",
            "skip_next_stage_required_input_missing",
            "fallthrough",
            "other",
        }
    ),
}
_METADATA_INT_MINIMUMS: dict[str, int] = {
    "attempt": 1,
    "byte_count": 0,
    "code_bytes": 0,
    "duration_ms": 0,
    "input_bytes": 0,
    "input_tokens": 0,
    "max_retries": 0,
    "output_tokens": 0,
    "priority": 0,
    "response_bytes": 0,
    "step_count": 0,
    "total_tokens": 0,
}
_METADATA_PATTERNS: dict[str, str] = {
    "category": r"^[a-z][a-z0-9_]*$",
    "command_id": r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    "error_code": r"^[a-z][a-z0-9_]*$",
    "error_type": r"^[A-Za-z][A-Za-z0-9_]*$",
    "method": r"^[A-Z]+$",
    "origin_class": r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    "path_ref": r"^path-[1-9][0-9]*$",
    "policy_ref": r"^p-[1-9][0-9]*$",
    "result_status": r"^[a-z][a-z0-9_]*$",
    "result_type": r"^[a-z][a-z0-9_]*$",
    "root_class": r"^\$[a-z][a-z0-9_]*$",
    "status_class": r"^[1-5]xx$",
}
_METADATA_MAX_LENGTHS: dict[str, int] = {
    "category": 64,
    "command_id": 128,
    "entry": 256,
    "error_code": 64,
    "error_type": 128,
    "method": 16,
    "origin_class": 128,
    "result_status": 64,
    "result_type": 64,
    "workflow_id": 256,
}
_NON_ANNOTATION_KEYS = frozenset({"annotation", "anchor_sequence"})

# Bounds for the few `public_args` values that are plain strings rather than
# opaque markers (``public-event.schema.json#/$defs/public_args``).
_MAX_ARGS_PATH_LEN = 512
_MAX_ARGS_METHOD_LEN = 16
_MAX_ARGS_CLASS_LEN = 128
_MAX_CAPABILITY_NAME_LEN = 128

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]*$")
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MAX_TOOL_NAME_LEN = 128
_MAX_REVIEWER_LEN = 200
_MAX_LICENSE_LEN = 64
_MAX_CONSENT_LEN = 500


class PublicationError(TraceError):
    """A publication source, attestation, or scanner gate refused the export."""


# ---------------------------------------------------------------------------
# Options, attestation, and result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicationOptions:
    """Review decisions that change the ``publication-v1`` projection.

    ``allow_tool_names`` retains reviewed static host tool declarations;
    everything else is dropped from the public ledger. ``keep_relative_paths``
    keeps alias-relative ``$alias/...`` arguments instead of opaque
    ``path-N`` refs, and therefore requires explicit human review.
    """

    allow_tool_names: tuple[str, ...] = ()
    keep_relative_paths: bool = False

    def __post_init__(self) -> None:
        normalized = sorted({name for name in self.allow_tool_names if name})
        for name in normalized:
            if len(name) > _MAX_TOOL_NAME_LEN or not _TOOL_NAME_RE.fullmatch(name):
                msg = f"allowed tool name {name!r} is not a safe static declaration"
                raise PublicationError(msg)
        object.__setattr__(self, "allow_tool_names", tuple(normalized))

    def as_dict(self) -> dict[str, Any]:
        """Canonical projection-options object (part of the reviewed digest)."""
        return {
            "allow_tool_names": list(self.allow_tool_names),
            "keep_relative_paths": self.keep_relative_paths,
        }


@dataclass(frozen=True)
class PublicationSource:
    """Safe facts about the reviewed source archive (basename only)."""

    archive: str
    trace_id: str
    content_identity: str
    profile: str
    status: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "archive": self.archive,
            "trace_id": self.trace_id,
            "content_identity": self.content_identity,
            "profile": self.profile,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class PublicationStats:
    """Counts only. A report never carries a redacted value."""

    event_count: int
    stage_visit_count: int
    tool_names_removed: int
    tool_names_retained: int
    paths_opaque: int
    redaction_markers: int
    annotations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "stage_visit_count": self.stage_visit_count,
            "tool_names_removed": self.tool_names_removed,
            "tool_names_retained": self.tool_names_retained,
            "paths_opaque": self.paths_opaque,
            "redaction_markers": self.redaction_markers,
            "annotations": self.annotations,
        }


@dataclass(frozen=True)
class PublicationAttestation:
    """A human attestation bound to one reviewed projection.

    The projection digest covers the projected ledger, the rebuilt graph, and
    the projection options, so attesting one projection and publishing another
    is impossible.
    """

    reviewer: str
    reviewed_at: str
    license: str
    consent: str
    source_content_identity: str
    source_trace_id: str
    projection_sha256: str
    options: PublicationOptions

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": PUBLICATION_ATTESTATION_FORMAT,
            "source": {
                "content_identity": self.source_content_identity,
                "trace_id": self.source_trace_id,
            },
            "projection": {"sha256": self.projection_sha256},
            "options": self.options.as_dict(),
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "license": self.license,
            "consent": self.consent,
        }


@dataclass(frozen=True)
class PublicationProjection:
    """One reviewed-but-not-yet-written publication projection."""

    entries: dict[str, bytes]
    trace_id: str
    projection_sha256: str
    content_identity: str
    source: PublicationSource
    stats: PublicationStats
    findings: tuple[str, ...]
    scanned_entries: tuple[str, ...]
    options: PublicationOptions

    @property
    def ok(self) -> bool:
        """True when the projection may be attested and written."""
        return not self.findings

    def report(
        self,
        *,
        attested: bool,
        attestation: PublicationAttestation | None,
    ) -> dict[str, Any]:
        """The disclosure report a human reviews before attestation."""
        publication: dict[str, Any] = {
            "trace_id": self.trace_id,
            "projection_sha256": self.projection_sha256,
            "content_identity": self.content_identity,
            "redaction_policy": PUBLICATION_REDACTION_POLICY,
            "vault_present": False,
            "entries": list(self.scanned_entries),
        }
        return {
            "format": PUBLICATION_REPORT_FORMAT,
            "source": self.source.as_dict(),
            "publication": publication,
            "options": self.options.as_dict(),
            "scan": {
                "ruleset": SCANNER_RULESET,
                "status": "passed" if self.ok else "failed",
                "findings": list(self.findings),
                "findings_count": len(self.findings),
            },
            "stats": self.stats.as_dict(),
            "disclosure": {
                "source_profile": self.source.profile,
                "vault_copied": False,
                "identifiers_retained": [
                    "workflow_id",
                    "stage_ids",
                    "capabilities",
                    "declared_output_field_names",
                ],
                "limits": [
                    "Publication is a fresh projection of an audit ledger, not a "
                    "re-encoding of a private run.",
                    "Paths are opaque path-N references unless a reviewer opted into "
                    "relative paths.",
                    "Tool names are dropped unless a reviewer allowlisted them.",
                    "Redaction reduces risk; reviewed identifiers, declared field "
                    "names, and approved scalar metrics may still be sensitive. "
                    "Human review is mandatory.",
                ],
            },
            "attested": attested,
            "attestation": attestation.as_dict() if attestation is not None else None,
        }


@dataclass(frozen=True)
class PublicationResult:
    """Outcome of ``prepare_publication`` (no private values)."""

    destination: Path
    trace_id: str
    projection_sha256: str
    content_identity: str
    compressed_bytes: int
    source: PublicationSource
    stats: PublicationStats
    report_path: Path


# ---------------------------------------------------------------------------
# Canonical hashing helpers (mirrored in TypeScript)
# ---------------------------------------------------------------------------


def _projection_digest(
    events: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    options: PublicationOptions,
) -> str:
    """Digest of exactly what will be published (ledger, graph, options).

    Records carry their source ``run_id`` here: the publication trace id is
    derived *from* this digest, so it cannot participate in it.
    """
    projection_obj = {
        "format": PUBLICATION_PROJECTION_FORMAT,
        "options": options.as_dict(),
        "graph": dict(graph),
        "events": [dict(event) for event in events],
    }
    return sha256_tag(to_canonical_bytes(projection_obj))


def _publication_trace_id(projection_sha256: str) -> str:
    """Fresh deterministic trace id: same source + options => same artifact."""
    digest = hashlib.sha256(PUBLICATION_ID_DOMAIN + projection_sha256.encode("utf-8")).hexdigest()
    return digest[:32]


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def _dict(value: Any, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"publication {what} must be an object"
        raise PublicationError(msg)
    return cast("dict[str, Any]", cast("Any", value))


def _list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        msg = f"publication {what} must be an array"
        raise PublicationError(msg)
    return cast("list[Any]", cast("Any", value))


def _string(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"publication {what} must be a non-empty string"
        raise PublicationError(msg)
    return value


def _string_list(value: Any, what: str) -> list[str]:
    return [_string(item, f"{what} item") for item in _list(value, what)]


def _require_keys(
    node: Mapping[str, Any], allowed: frozenset[str], what: str, *, required: Sequence[str] = ()
) -> None:
    missing = [key for key in required if key not in node]
    if missing:
        msg = f"publication {what} is missing {sorted(missing)}"
        raise PublicationError(msg)
    unknown = sorted(set(node) - allowed)
    if unknown:
        msg = (
            f"publication {what} has fields this transform does not understand: "
            f"{unknown}; refusing to republish unknown structure"
        )
        raise PublicationError(msg)


def _is_marker(value: Any) -> bool:
    """True for a ``common.schema.json`` redaction marker (opaque by design)."""
    if not isinstance(value, dict):
        return False
    marker = cast("dict[Any, Any]", value)
    return isinstance(marker.get("$redacted"), dict)


def _public_scalar_ok(value: Any) -> bool:
    """The closed ``public_scalar`` set: null, boolean, finite number, or marker."""
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, int):
        return MIN_SAFE_INT <= value <= MAX_SAFE_INT
    if isinstance(value, float):
        return math.isfinite(value)
    return _is_marker(value)


def _metadata_value_ok(key: str, value: Any) -> bool:
    """Per-key shape check mirroring ``public-event.schema.json`` metadata."""
    if key in _METADATA_BOOLS:
        return isinstance(value, bool)
    enums = _METADATA_ENUMS.get(key)
    if enums is not None:
        return isinstance(value, str) and value in enums
    minimum = _METADATA_INT_MINIMUMS.get(key)
    if minimum is not None:
        return isinstance(value, int) and not isinstance(value, bool) and value >= minimum
    if key == "cost_usd":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
    if key == "required_capabilities":
        return isinstance(value, list) and all(
            isinstance(item, str) and 1 <= len(item) <= _MAX_CAPABILITY_NAME_LEN
            for item in cast("list[Any]", value)
        )
    pattern = _METADATA_PATTERNS.get(key)
    if pattern is not None:
        limit = _METADATA_MAX_LENGTHS.get(key)
        return (
            isinstance(value, str)
            and re.fullmatch(pattern, value) is not None
            and (limit is None or len(value) <= limit)
        )
    limit = _METADATA_MAX_LENGTHS.get(key)
    if limit is not None:
        return isinstance(value, str) and 1 <= len(value) <= limit
    return False  # unreachable for allowlisted keys; fail closed if a set drifts.


def _arg_value_ok(key: str, value: Any) -> bool:
    """Per-key shape check mirroring ``public-event.schema.json`` public_args."""
    if key in _ARGS_MARKER_KEYS:
        return _is_marker(value)
    if key == "path":
        return isinstance(value, str) and value.startswith("$") and len(value) <= _MAX_ARGS_PATH_LEN
    if key == "root_class":
        return isinstance(value, str) and re.fullmatch(r"^\$[a-z][a-z0-9_]*$", value) is not None
    if key == "path_ref":
        return isinstance(value, str) and re.fullmatch(r"^path-[1-9][0-9]*$", value) is not None
    if key == "method":
        return (
            isinstance(value, str)
            and re.fullmatch(r"^[A-Z]+$", value) is not None
            and len(value) <= _MAX_ARGS_METHOD_LEN
        )
    if key in ("command_id", "origin_class"):
        return (
            isinstance(value, str)
            and re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$", value) is not None
            and len(value) <= _MAX_ARGS_CLASS_LEN
        )
    return False  # unreachable for allowlisted keys; fail closed if a set drifts.


def _validate_record_shape(record: Mapping[str, Any], redacted: list[str]) -> None:
    """Enforce the ``publication-v1`` nested allowlists on one ledger record.

    The reader tolerates forward-compatible fields; publication refuses them:
    a field this transform does not classify must never reach a published
    artifact. ``args`` names are the one exception (open tool-domain names are
    dropped with a redaction pointer), and annotation payloads are already
    shape-gated by ``verify_archive``.
    """
    _require_keys(
        record,
        _LEDGER_KEYS,
        "source ledger record",
        required=["kind", "run_id", "timestamp", "redacted_fields"],
    )
    if record.get("kind") != "annotation":
        present = sorted(_NON_ANNOTATION_KEYS & set(record))
        if present:
            msg = (
                f"publication source ledger record of kind {record.get('kind')!r} "
                f"carries annotation fields {present}"
            )
            raise PublicationError(msg)
    metadata = record.get("metadata")
    if metadata is not None:
        metadata_map = _dict(metadata, "source ledger metadata")
        _require_keys(metadata_map, _METADATA_KEYS, "source ledger metadata")
        for key in sorted(metadata_map):
            if not _metadata_value_ok(key, metadata_map[key]):
                msg = (
                    f"publication source ledger metadata field {key!r} "
                    "does not match the publication-v1 shape"
                )
                raise PublicationError(msg)
    output = record.get("output")
    if output is not None:
        output_map = _dict(output, "source ledger output")
        for key in sorted(output_map):
            if not _public_scalar_ok(output_map[key]):
                msg = (
                    f"publication source ledger output field {key!r} "
                    "does not match the publication-v1 shape"
                )
                raise PublicationError(msg)
    args = record.get("args")
    if args is not None:
        args_map = _dict(args, "source ledger args")
        for key in sorted(args_map):
            if key not in _PUBLIC_ARGS_KEYS:
                redacted.append(f"/args/{key}")
                continue
            if not _arg_value_ok(key, args_map[key]):
                msg = (
                    f"publication source ledger args field {key!r} "
                    "does not match the publication-v1 shape"
                )
                raise PublicationError(msg)


# ---------------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------------


def _validate_manifest(entries: Mapping[str, bytes]) -> tuple[dict[str, Any], PublicationSource]:
    """Validate source manifest facts and return them with a source record."""
    manifest = _dict(parse_json_strict(entries[MANIFEST_PATH].decode("utf-8")), "source manifest")
    _require_keys(manifest, _MANIFEST_KEYS, "source manifest", required=sorted(_MANIFEST_KEYS))
    capture = _dict(manifest["capture"], "source manifest capture")
    _require_keys(
        capture,
        _CAPTURE_KEYS,
        "source manifest capture",
        required=["profile", "vault_present"],
    )
    profile = capture.get("profile")
    if profile != SOURCE_PROFILE:
        msg = (
            f"publication requires an '{SOURCE_PROFILE}' source archive, got {profile!r}: "
            "vault-bearing and already-published archives are refused"
        )
        raise PublicationError(msg)
    if capture.get("vault_present") is not False:
        msg = "publication refuses a source archive that declares a vault"
        raise PublicationError(msg)
    if "private/vault.enc" in entries or "private/vault.meta.json" in entries:
        msg = "publication refuses a source archive that carries vault entries"
        raise PublicationError(msg)
    status = manifest.get("status")
    if status not in PUBLISHABLE_STATUSES:
        msg = (
            f"publication requires a {' or '.join(PUBLISHABLE_STATUSES)} source run, got "
            f"{status!r} (interrupted runs carry partial evidence)"
        )
        raise PublicationError(msg)
    workflow = _dict(manifest["workflow"], "source manifest workflow")
    _require_keys(
        workflow, _WORKFLOW_KEYS, "source manifest workflow", required=sorted(_WORKFLOW_KEYS)
    )
    provenance = _dict(manifest["provenance"], "source manifest provenance")
    _require_keys(
        provenance,
        _PROVENANCE_KEYS,
        "source manifest provenance",
        required=sorted(_REQUIRED_PROVENANCE_KEYS),
    )
    runtime = _dict(provenance["runtime"], "source provenance runtime")
    _require_keys(
        runtime, _RUNTIME_KEYS, "source provenance runtime", required=sorted(_RUNTIME_KEYS)
    )
    if "model" in provenance:
        model = _dict(provenance["model"], "source provenance model")
        _require_keys(model, _MODEL_KEYS, "source provenance model", required=["name"])
        if "sampling" in model:
            _require_keys(
                _dict(model["sampling"], "source provenance sampling"),
                _SAMPLING_KEYS,
                "source provenance sampling",
            )
    viewer = _dict(manifest["viewer"], "source manifest viewer")
    _require_keys(viewer, _VIEWER_KEYS, "source manifest viewer", required=sorted(_VIEWER_KEYS))
    # Publication requires an exact IR binding (manifest.schema.json).
    if provenance.get("complete") is not True:
        msg = "publication requires complete compiler provenance (exact IR binding)"
        raise PublicationError(msg)
    ir_sha256 = workflow.get("ir_sha256")
    if not isinstance(ir_sha256, str) or not _SHA256_RE.fullmatch(ir_sha256):
        msg = "publication source manifest has no valid ir_sha256"
        raise PublicationError(msg)
    trace_id = manifest.get("trace_id")
    if not isinstance(trace_id, str) or not _TRACE_ID_RE.fullmatch(trace_id):
        msg = "publication source manifest has an invalid trace_id"
        raise PublicationError(msg)
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        msg = "publication source manifest has an invalid created_at"
        raise PublicationError(msg)
    source = PublicationSource(
        archive="",
        trace_id=trace_id,
        content_identity="",
        profile=str(profile),
        status=str(status),
        created_at=created_at,
    )
    return manifest, source


def _read_ledger(entries: Mapping[str, bytes], source_trace_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(entries[EVENTS_PATH].split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            value = parse_json_strict(line.decode("utf-8"))
        except Exception as exc:
            msg = f"publication source ledger line {number} is unparsable: {exc}"
            raise PublicationError(msg) from exc
        record = _dict(value, f"source ledger line {number}")
        kind = record.get("kind")
        if kind not in _LEDGER_KINDS:
            msg = f"publication source ledger line {number} has unknown kind {kind!r}"
            raise PublicationError(msg)
        if record.get("run_id") != source_trace_id:
            msg = f"publication source ledger line {number} is not bound to its manifest trace_id"
            raise PublicationError(msg)
        if record.get("_omit") is not None:
            msg = f"publication source ledger line {number} carries an internal omit marker"
            raise PublicationError(msg)
        records.append(record)
    if not records:
        msg = "publication source ledger is empty"
        raise PublicationError(msg)
    return records


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


class _ProjectionState:
    """Mutable state for one projection pass (counters + opaque path refs)."""

    def __init__(self, options: PublicationOptions) -> None:
        self.options = options
        self.path_refs: dict[str, str] = {}
        self.tool_names_removed = 0
        self.tool_names_retained = 0
        self.paths_opaque = 0
        self.redaction_markers = 0
        self.annotations = 0

    def path_ref(self, path: str) -> str:
        existing = self.path_refs.get(path)
        if existing is not None:
            return existing
        ref = f"path-{len(self.path_refs) + 1}"
        self.path_refs[path] = ref
        return ref


def _count_markers(node: Any) -> int:
    if isinstance(node, dict):
        mapping = cast("dict[Any, Any]", node)
        if "$redacted" in mapping:
            return 1
        return sum(_count_markers(value) for value in mapping.values())
    if isinstance(node, list):
        return sum(_count_markers(value) for value in cast("list[Any]", node))
    return 0


def _project_args(
    args: Mapping[str, Any], state: _ProjectionState, redacted: list[str]
) -> dict[str, Any]:
    """Apply the ``publication-v1`` capability projection to tool arguments.

    Unrecognized argument names were already reported by
    :func:`_validate_record_shape`; they are dropped here rather than
    refused, because ``args`` names are open tool-domain data.
    """
    projected = {key: args[key] for key in args if key in _PUBLIC_ARGS_KEYS}
    path = projected.get("path")
    if isinstance(path, str) and not state.options.keep_relative_paths:
        projected.pop("path", None)
        projected["path_ref"] = state.path_ref(path)
        redacted.append("/args/path")
        state.paths_opaque += 1
    return projected


def _project_record(
    record: Mapping[str, Any], state: _ProjectionState, run_id: str
) -> dict[str, Any]:
    """Apply ``publication-v1`` to one audit ledger record."""
    projected = {key: record[key] for key in record}
    projected["run_id"] = run_id
    fields = _string_list(record.get("redacted_fields", []), "source redacted_fields")
    _validate_record_shape(record, fields)
    tool_name = projected.get("tool_name")
    if isinstance(tool_name, str):
        if tool_name in state.options.allow_tool_names:
            state.tool_names_retained += 1
        else:
            projected.pop("tool_name", None)
            fields.append("/tool_name")
            state.tool_names_removed += 1
    args = projected.get("args")
    if isinstance(args, dict):
        projected["args"] = _project_args(cast("dict[str, Any]", cast("Any", args)), state, fields)
    if fields:
        projected["redacted_fields"] = sorted(set(fields))
    if projected.get("kind") == "annotation":
        state.annotations += 1
    state.redaction_markers += _count_markers(projected)
    return projected


def _project_graph(graph_raw: Any) -> dict[str, Any]:
    """Rebuild the public graph through a strict publication allowlist."""
    graph = _dict(graph_raw, "source workflow graph")
    _require_keys(graph, _GRAPH_KEYS, "source workflow graph", required=sorted(_GRAPH_KEYS))
    if graph.get("format") != _GRAPH_FORMAT:
        msg = f"publication source workflow graph format {graph.get('format')!r} is unsupported"
        raise PublicationError(msg)
    nodes: list[dict[str, Any]] = []
    for node_raw in _list(graph["nodes"], "source graph nodes"):
        node = _dict(node_raw, "source graph node")
        _require_keys(node, _NODE_KEYS, "source graph node", required=["id", "execution"])
        writes: list[dict[str, Any]] = []
        for write_raw in _list(node.get("writes", []), "source node writes"):
            write = _dict(write_raw, "source node write")
            _require_keys(
                write, _WRITE_KEYS, "source node write", required=["name", "type", "optional"]
            )
            writes.append(
                {
                    "name": _string(write["name"], "node write name"),
                    "type": _string(write["type"], "node write type"),
                    "optional": bool(write["optional"]),
                }
            )
        nodes.append(
            {
                "id": _string(node["id"], "node id"),
                "execution": _string(node["execution"], "node execution"),
                "capabilities": _string_list(
                    node.get("capabilities", []), "source node capabilities"
                ),
                "writes": writes,
            }
        )
    transitions: list[dict[str, Any]] = []
    for transition_raw in _list(graph["transitions"], "source graph transitions"):
        transition = _dict(transition_raw, "source graph transition")
        _require_keys(
            transition,
            _TRANSITION_KEYS,
            "source graph transition",
            required=sorted(_TRANSITION_KEYS),
        )
        transitions.append(
            {
                "from": _string(transition["from"], "transition from"),
                "to": _string(transition["to"], "transition to"),
                "priority": transition["priority"],
                "guard_kind": _string(transition["guard_kind"], "transition guard_kind"),
            }
        )
    policies: list[dict[str, Any]] = []
    for policy_raw in _list(graph["policies"], "source graph policies"):
        policy = _dict(policy_raw, "source graph policy")
        _require_keys(
            policy,
            _POLICY_KEYS,
            "source graph policy",
            required=["ref", "kind", "trigger_capability"],
        )
        policy_trigger = _string(policy["trigger_capability"], "policy trigger capability")
        entry: dict[str, Any] = {
            "ref": _string(policy["ref"], "policy ref"),
            "kind": _string(policy["kind"], "policy kind"),
            "trigger_capability": policy_trigger,
        }
        if "required_capabilities" in policy:
            entry["required_capabilities"] = _string_list(
                policy["required_capabilities"], "policy required_capabilities"
            )
        policies.append(entry)
    return {
        "format": _GRAPH_FORMAT,
        "workflow_id": _string(graph["workflow_id"], "graph workflow_id"),
        "entry": _string(graph["entry"], "graph entry"),
        "exits": _string_list(graph["exits"], "graph exits"),
        "nodes": nodes,
        "transitions": transitions,
        "policies": policies,
    }


def _build_summary(
    records: Sequence[Mapping[str, Any]],
    events_bytes: bytes,
    source_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    visits: set[str] = set()
    for record in records:
        kind = str(record.get("kind", "?"))
        kinds[kind] = kinds.get(kind, 0) + 1
        visit = record.get("stage_visit_id")
        if isinstance(visit, str):
            visits.add(visit)
    summary: dict[str, Any] = {
        "format": SUMMARY_FORMAT,
        "events_sha256": sha256_tag(events_bytes),
        "event_count": len(records),
        "stage_visit_count": len(visits),
        "duration_ms": 0,
        "counts_by_kind": kinds,
        "annotations_dropped": 0,
        "annotation_warnings": [],
    }
    if source_summary is not None:
        duration = source_summary.get("duration_ms")
        if isinstance(duration, int) and not isinstance(duration, bool) and duration >= 0:
            summary["duration_ms"] = duration
        dropped = source_summary.get("annotations_dropped")
        if isinstance(dropped, int) and not isinstance(dropped, bool) and dropped >= 0:
            summary["annotations_dropped"] = dropped
        warnings: Any = source_summary.get("annotation_warnings")
        if isinstance(warnings, list):
            summary["annotation_warnings"] = _string_list(warnings, "source annotation_warnings")
    return summary


def _build_manifest(manifest: Mapping[str, Any], trace_id: str) -> dict[str, Any]:
    workflow = _dict(manifest["workflow"], "source manifest workflow")
    provenance = _dict(manifest["provenance"], "source manifest provenance")
    published_provenance: dict[str, Any] = {
        key: provenance[key]
        for key in (
            "complete",
            "frontend",
            "target",
            "compiler_version",
            "runtime",
            "model",
        )
        if key in provenance
    }
    return {
        "format": TRACE_FORMAT,
        "trace_id": trace_id,
        "created_at": manifest["created_at"],
        "status": manifest["status"],
        "capture": {
            "profile": "publication",
            "vault_present": False,
            "publication_eligible": True,
            "redaction_policy": PUBLICATION_REDACTION_POLICY,
            "scanner": {"status": "passed", "ruleset": SCANNER_RULESET},
            "attested": True,
        },
        "workflow": {
            "id": workflow["id"],
            "ir_version": workflow["ir_version"],
            "ir_sha256": workflow["ir_sha256"],
            "entry": workflow["entry"],
            "exits": _string_list(workflow["exits"], "source workflow exits"),
        },
        "provenance": published_provenance,
        "integrity": {"algorithm": "sha256", "entry": INTEGRITY_PATH},
        "viewer": {"min_format": TRACE_FORMAT},
    }


def _build_entries(
    manifest_obj: Mapping[str, Any],
    graph: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {
        MANIFEST_PATH: to_canonical_bytes(dict(manifest_obj)),
        GRAPH_PATH: to_canonical_bytes(dict(graph)),
        EVENTS_PATH: b"".join(to_canonical_bytes(record) + b"\n" for record in events),
        SUMMARY_PATH: to_canonical_bytes(dict(summary)),
    }
    entries = [
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
    identity_obj = {
        "format": CONTENT_IDENTITY_FORMAT,
        "entries": sorted(
            (
                {
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "uncompressed_bytes": entry["uncompressed_bytes"],
                }
                for entry in entries
            ),
            key=lambda item: item["path"],
        ),
    }
    integrity_obj: dict[str, Any] = {
        "format": "nemoir.trace.integrity/0.1",
        "algorithm": "sha256",
        "entries": entries,
        "content_identity": sha256_tag(to_canonical_bytes(identity_obj)),
    }
    payloads[INTEGRITY_PATH] = to_canonical_bytes(integrity_obj)
    return payloads


def _content_identity(entries: Mapping[str, bytes]) -> str:
    integrity = _dict(
        parse_json_strict(entries[INTEGRITY_PATH].decode("utf-8")), "publication integrity"
    )
    return _string(integrity.get("content_identity"), "integrity content_identity")


def require_publication_artifact(
    manifest: Mapping[str, Any],
    entries: Mapping[str, bytes],
    *,
    size_bytes: int,
    max_bytes: int,
) -> None:
    """Strict gate for an attested, vault-free ``publication`` artifact.

    Shared by the pre-upload plan and the post-download Gist verification so
    a re-indexed, truncated, or hand-edited archive cannot pass one path and
    fail the other (``plan.md`` §7.5, §9.2). Every failure is a location-only
    reason: no field value is ever echoed.
    """
    capture = _dict(manifest.get("capture"), "publication manifest capture")
    profile = capture.get("profile")
    if profile != "publication":
        msg = (
            "only an attested publication-profile archive may be published; "
            f"this archive is {profile!r}, not 'publication'"
        )
        raise PublicationError(msg)
    if capture.get("vault_present") is not False or "private/vault.enc" in entries:
        msg = "refusing to publish an archive that carries a vault"
        raise PublicationError(msg)
    if capture.get("attested") is not True:
        msg = "refusing to publish an archive that is not attested"
        raise PublicationError(msg)
    if capture.get("publication_eligible") is not True:
        msg = "refusing to publish an archive that is not marked publication_eligible"
        raise PublicationError(msg)
    policy = capture.get("redaction_policy")
    if policy != PUBLICATION_REDACTION_POLICY:
        msg = (
            "refusing to publish an archive whose redaction policy is "
            f"{policy!r}, not {PUBLICATION_REDACTION_POLICY!r}"
        )
        raise PublicationError(msg)
    scanner = _dict(capture.get("scanner"), "publication manifest scanner")
    if scanner.get("status") != "passed":
        msg = "refusing to publish an archive whose scanner did not pass"
        raise PublicationError(msg)
    ruleset = scanner.get("ruleset")
    if ruleset != SCANNER_RULESET:
        msg = (
            "refusing to publish an archive whose scanner ruleset is "
            f"{ruleset!r}, not {SCANNER_RULESET!r}"
        )
        raise PublicationError(msg)
    provenance = _dict(manifest.get("provenance"), "publication manifest provenance")
    if provenance.get("complete") is not True:
        msg = "refusing to publish an archive without complete provenance"
        raise PublicationError(msg)
    if size_bytes > max_bytes:
        msg = (
            f"archive is {size_bytes} bytes, over the {max_bytes} public budget; "
            "produce a smaller publication projection instead of splitting the trace"
        )
        raise PublicationError(msg)


# ---------------------------------------------------------------------------
# Scan (project for review)
# ---------------------------------------------------------------------------


def scan_publication(
    source: str | Path,
    *,
    options: PublicationOptions | None = None,
    archive_name: str | None = None,
) -> PublicationProjection:
    """Project an ``audit`` archive for review without writing anything.

    Returns the exact entries ``prepare_publication`` would write, the
    projection digest an attestation must cover, and the scanner findings.
    """
    active = options if options is not None else PublicationOptions()
    source_path = Path(source)
    try:
        entries = read_archive_entries(source_path)
    except OSError as exc:
        msg = f"cannot read publication source: {exc}"
        raise PublicationError(msg) from exc
    verification = verify_archive(source_path)
    if not verification.ok:
        detail = "; ".join(verification.errors[:5]) or "verification failed"
        msg = f"publication source does not verify: {detail}"
        raise PublicationError(msg)
    manifest, source_facts = _validate_manifest(entries)
    records = _read_ledger(entries, source_facts.trace_id)
    graph = _project_graph(parse_json_strict(entries[GRAPH_PATH].decode("utf-8")))
    state = _ProjectionState(active)
    # Pass 1: publication-v1 projection, keeping the source run_id so the
    # digest can be computed before the fresh publication id exists.
    reviewed = [_project_record(record, state, source_facts.trace_id) for record in records]
    digest = _projection_digest(reviewed, graph, active)
    trace_id = _publication_trace_id(digest)
    # Pass 2: the only change is the derived identity.
    projected = [dict(record, run_id=trace_id) for record in reviewed]
    source_summary: dict[str, Any] | None = None
    if SUMMARY_PATH in entries:
        try:
            parsed: Any = parse_json_strict(entries[SUMMARY_PATH].decode("utf-8"))
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            source_summary = cast("dict[str, Any]", parsed)
    events_bytes = b"".join(to_canonical_bytes(record) + b"\n" for record in projected)
    summary = _build_summary(projected, events_bytes, source_summary)
    payloads = _build_entries(_build_manifest(manifest, trace_id), graph, projected, summary)
    findings = tuple(scan_cleartext_entries(payloads))
    visits = {str(record["stage_visit_id"]) for record in projected if "stage_visit_id" in record}
    stats = PublicationStats(
        event_count=len(projected),
        stage_visit_count=len(visits),
        tool_names_removed=state.tool_names_removed,
        tool_names_retained=state.tool_names_retained,
        paths_opaque=state.paths_opaque,
        redaction_markers=state.redaction_markers,
        annotations=state.annotations,
    )
    source_with_identity = PublicationSource(
        archive=archive_name if archive_name is not None else source_path.name,
        trace_id=source_facts.trace_id,
        content_identity=verification.content_identity or "",
        profile=source_facts.profile,
        status=source_facts.status,
        created_at=source_facts.created_at,
    )
    return PublicationProjection(
        entries=payloads,
        trace_id=trace_id,
        projection_sha256=digest,
        content_identity=_content_identity(payloads),
        source=source_with_identity,
        stats=stats,
        findings=findings,
        scanned_entries=tuple(sorted(payloads)),
        options=active,
    )


# ---------------------------------------------------------------------------
# Attestation
# ---------------------------------------------------------------------------


def build_attestation(
    projection: PublicationProjection,
    *,
    options: PublicationOptions,
    reviewer: str,
    license_id: str,
    consent: str,
    reviewed_at: str | None = None,
) -> PublicationAttestation:
    """Bind a human review to one exact projection digest."""
    reviewer_clean = reviewer.strip()
    if not reviewer_clean or len(reviewer_clean) > _MAX_REVIEWER_LEN:
        msg = f"attestation requires a reviewer name (1-{_MAX_REVIEWER_LEN} characters)"
        raise PublicationError(msg)
    license_clean = license_id.strip()
    if not license_clean or len(license_clean) > _MAX_LICENSE_LEN:
        msg = f"attestation requires a license identifier (1-{_MAX_LICENSE_LEN} characters)"
        raise PublicationError(msg)
    consent_clean = " ".join(consent.split())
    if not consent_clean or len(consent_clean) > _MAX_CONSENT_LEN:
        msg = f"attestation requires a consent statement (1-{_MAX_CONSENT_LEN} characters)"
        raise PublicationError(msg)
    moment = reviewed_at if reviewed_at is not None else format_timestamp(datetime.now(UTC))
    return PublicationAttestation(
        reviewer=reviewer_clean,
        reviewed_at=moment,
        license=license_clean,
        consent=consent_clean,
        source_content_identity=projection.source.content_identity,
        source_trace_id=projection.source.trace_id,
        projection_sha256=projection.projection_sha256,
        options=options,
    )


def attestation_from_report(
    report: Mapping[str, Any],
    *,
    reviewer: str,
    license_id: str,
    consent: str,
    reviewed_at: str | None = None,
) -> PublicationAttestation:
    """Build an attestation from the disclosure report a human just reviewed.

    Deriving the digests from the report (rather than accepting them on the
    command line) means a reviewer cannot attest a projection they have not
    been shown, and a failed scan can never be attested.
    """
    node = _dict(dict(report), "report")
    if node.get("format") != PUBLICATION_REPORT_FORMAT:
        msg = f"unsupported disclosure report format {node.get('format')!r}"
        raise PublicationError(msg)
    scan = _dict(node.get("scan"), "report scan")
    if scan.get("status") != "passed":
        findings = scan.get("findings_count", 0)
        msg = f"report scan did not pass ({findings} finding(s)); refusal cannot be attested"
        raise PublicationError(msg)
    source = _dict(node.get("source"), "report source")
    publication = _dict(node.get("publication"), "report publication")
    options_node = _dict(node.get("options"), "report options")
    _require_keys(options_node, _OPTIONS_KEYS, "report options", required=sorted(_OPTIONS_KEYS))
    keep = options_node["keep_relative_paths"]
    if not isinstance(keep, bool):
        msg = "report options keep_relative_paths must be a boolean"
        raise PublicationError(msg)
    options = PublicationOptions(
        allow_tool_names=tuple(
            _string_list(options_node["allow_tool_names"], "report allow_tool_names")
        ),
        keep_relative_paths=keep,
    )
    digest = publication.get("projection_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        msg = "report projection_sha256 is invalid"
        raise PublicationError(msg)
    reviewer_clean = reviewer.strip()
    if not reviewer_clean or len(reviewer_clean) > _MAX_REVIEWER_LEN:
        msg = f"attestation requires a reviewer name (1-{_MAX_REVIEWER_LEN} characters)"
        raise PublicationError(msg)
    license_clean = license_id.strip()
    if not license_clean or len(license_clean) > _MAX_LICENSE_LEN:
        msg = f"attestation requires a license identifier (1-{_MAX_LICENSE_LEN} characters)"
        raise PublicationError(msg)
    consent_clean = " ".join(consent.split())
    if not consent_clean or len(consent_clean) > _MAX_CONSENT_LEN:
        msg = f"attestation requires a consent statement (1-{_MAX_CONSENT_LEN} characters)"
        raise PublicationError(msg)
    moment = reviewed_at if reviewed_at is not None else format_timestamp(datetime.now(UTC))
    return PublicationAttestation(
        reviewer=reviewer_clean,
        reviewed_at=moment,
        license=license_clean,
        consent=consent_clean,
        source_content_identity=_string(
            source.get("content_identity"), "report source content_identity"
        ),
        source_trace_id=_string(source.get("trace_id"), "report source trace_id"),
        projection_sha256=digest,
        options=options,
    )


def attestation_from_dict(data: Any) -> PublicationAttestation:
    """Parse an attestation document, refusing anything malformed."""
    node = _dict(data, "attestation")
    if node.get("format") != PUBLICATION_ATTESTATION_FORMAT:
        msg = f"unsupported attestation format {node.get('format')!r}"
        raise PublicationError(msg)
    _require_keys(
        node,
        _ATTESTATION_KEYS,
        "attestation",
        required=sorted(_ATTESTATION_KEYS),
    )
    source = _dict(node["source"], "attestation source")
    _require_keys(
        source,
        _ATTESTATION_SOURCE_KEYS,
        "attestation source",
        required=sorted(_ATTESTATION_SOURCE_KEYS),
    )
    projection = _dict(node["projection"], "attestation projection")
    _require_keys(
        projection,
        _ATTESTATION_PROJECTION_KEYS,
        "attestation projection",
        required=sorted(_ATTESTATION_PROJECTION_KEYS),
    )
    options_node = _dict(node["options"], "attestation options")
    _require_keys(
        options_node,
        _OPTIONS_KEYS,
        "attestation options",
        required=sorted(_OPTIONS_KEYS),
    )
    keep = options_node["keep_relative_paths"]
    if not isinstance(keep, bool):
        msg = "attestation options keep_relative_paths must be a boolean"
        raise PublicationError(msg)
    options = PublicationOptions(
        allow_tool_names=tuple(
            _string_list(options_node["allow_tool_names"], "attestation allow_tool_names")
        ),
        keep_relative_paths=keep,
    )
    digest = projection.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        msg = "attestation projection sha256 is invalid"
        raise PublicationError(msg)
    return PublicationAttestation(
        reviewer=_string(node["reviewer"], "attestation reviewer").strip(),
        reviewed_at=_string(node["reviewed_at"], "attestation reviewed_at").strip(),
        license=_string(node["license"], "attestation license").strip(),
        consent=_string(node["consent"], "attestation consent").strip(),
        source_content_identity=_string(
            source["content_identity"], "attestation source content_identity"
        ).strip(),
        source_trace_id=_string(source["trace_id"], "attestation source trace_id").strip(),
        projection_sha256=digest,
        options=options,
    )


def load_attestation(path: str | Path) -> PublicationAttestation:
    """Read and validate one attestation JSON document."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read attestation: {exc}"
        raise PublicationError(msg) from exc
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"attestation is not valid JSON: {exc}"
        raise PublicationError(msg) from exc
    return attestation_from_dict(parsed)


def write_attestation(path: str | Path, attestation: PublicationAttestation) -> Path:
    """Write one attestation document next to (never inside) an archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(to_canonical_bytes(attestation.as_dict()) + b"\n")
    return target


def write_publication_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write one disclosure report (locations and counts only)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(to_canonical_bytes(dict(report)) + b"\n")
    return target


def publication_report_path(archive: str | Path) -> Path:
    """Sidecar report path for an archive: ``run.nemotrace.publication-report.json``."""
    path = Path(archive)
    return path.with_name(path.name + ".publication-report.json")


# ---------------------------------------------------------------------------
# Prepare (write) path
# ---------------------------------------------------------------------------


def prepare_publication(
    source: str | Path,
    destination: str | Path,
    *,
    attestation: PublicationAttestation,
    report_path: str | Path | None = None,
) -> PublicationResult:
    """Write one attested, vault-free ``publication`` archive.

    Refuses a projection whose digest, source identity, or source trace id
    does not match the attestation, and refuses any unresolved scanner
    finding. The disclosure report is written next to the artifact.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        msg = "publication destination must differ from its source archive"
        raise PublicationError(msg)
    projection = scan_publication(
        source_path,
        options=attestation.options,
        archive_name=source_path.name,
    )
    if projection.findings:
        detail = "; ".join(projection.findings[:5])
        msg = (
            f"publication scanner blocked export with {len(projection.findings)} finding(s): "
            f"{detail}"
        )
        raise PublicationError(msg)
    if projection.projection_sha256 != attestation.projection_sha256:
        msg = (
            "attestation does not cover this projection (projection_sha256 mismatch); "
            "re-run scan-publication and attest again"
        )
        raise PublicationError(msg)
    if projection.source.content_identity != attestation.source_content_identity:
        msg = "attestation source content identity does not match the source archive"
        raise PublicationError(msg)
    if projection.source.trace_id != attestation.source_trace_id:
        msg = "attestation source trace id does not match the source archive"
        raise PublicationError(msg)
    write_trace_archive(destination_path, projection.entries)
    compressed_bytes = destination_path.stat().st_size
    report = projection.report(attested=True, attestation=attestation)
    report_target = (
        Path(report_path) if report_path is not None else publication_report_path(destination_path)
    )
    write_publication_report(report_target, report)
    return PublicationResult(
        destination=destination_path,
        trace_id=projection.trace_id,
        projection_sha256=projection.projection_sha256,
        content_identity=projection.content_identity,
        compressed_bytes=compressed_bytes,
        source=projection.source,
        stats=projection.stats,
        report_path=report_target,
    )
