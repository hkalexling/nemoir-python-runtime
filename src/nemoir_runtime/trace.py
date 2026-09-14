"""NemoTrace audit/replay recorder and single-file archive (Phases 1+4).

Turns a run's live :class:`WorkflowEvent` stream plus recorder semantic
hooks into one portable ``*.nemotrace`` ZIP artifact: a redacted public
ledger, a safe workflow-graph projection, a derived summary cache, and an
integrity index binding everything to an exact compiled-IR fingerprint.

Security model (see ``docs/trace/redaction-policy.md``):

- A value is absent from cleartext unless an allowlist rule permits it.
- Redaction happens at capture time, before any journal/archive write. The
  viewer is never a redaction boundary.
- ``audit`` is the default redacted profile. ``replay`` produces the same
  public ledger plus an encrypted vault (``private/vault.enc``) holding taped
  model/tool/guard evidence for local re-execution; it requires a host
  passphrase and is never publication-eligible. There is no cleartext
  full-capture profile; ``publication`` arrives in a later phase and this
  module refuses it explicitly rather than silently producing it.
- Credentials never enter the ledger or any vault. The secret-value registry
  is defense-in-depth against echoed credentials, not the primary control.
- The final cleartext scanner blocks archive finalization on any unresolved
  finding and reports locations, never values.

Cross-language contract: ``docs/trace/schema/README.md`` is normative for
the wire form. Canonical JSON bytes come from
:mod:`nemoir_runtime.canonical` (RFC 8785); the Rust
(``nemoir-ir/src/canonical.rs``) and TypeScript (``canonical.ts``) ports must
produce byte-identical uncompressed entries for the same logical run.

Audit recording is stdlib-only so tracing never adds required runtime
dependencies. The ``replay`` vault codec (PBKDF2-HMAC-SHA-256 + AES-256-GCM)
imports the optional ``cryptography`` package lazily: importing this module
or recording ``audit`` traces never requires it.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nemoir_runtime.canonical import parse_json_strict, sha256_tag, to_canonical_bytes

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from nemoir_runtime.events import WorkflowEvent

# ---------------------------------------------------------------------------
# Format constants (mirror docs/trace/schema/README.md)
# ---------------------------------------------------------------------------

TRACE_FORMAT = "nemoir.trace/0.1"
GRAPH_FORMAT = "nemoir.trace.workflow-graph/0.1"
SUMMARY_FORMAT = "nemoir.trace.summary/0.1"
CONTENT_IDENTITY_FORMAT = "nemoir.trace.content-identity/0.1"
PROVENANCE_FORMAT = "nemoir.trace-provenance/0.1"
REDACTION_POLICY = "audit-v1"
SCANNER_RULESET = "secrets-v1"
RUNTIME_NAME = "nemoir-runtime"


def _empty_alias_map() -> dict[str, str | Path]:
    """Typed empty factory for :attr:`TraceConfig.path_aliases`."""
    return {}


MANIFEST_PATH = "manifest.json"
GRAPH_PATH = "public/workflow.graph.json"
EVENTS_PATH = "public/events.ndjson"
SUMMARY_PATH = "public/summary.json"
INTEGRITY_PATH = "integrity.json"
VAULT_ENC_PATH = "private/vault.enc"
VAULT_META_PATH = "private/vault.meta.json"
AUDIT_ENTRY_PATHS = (MANIFEST_PATH, GRAPH_PATH, EVENTS_PATH, SUMMARY_PATH)
VAULT_ENTRY_PATHS = (VAULT_ENC_PATH, VAULT_META_PATH)

# Phase 4 encrypted-vault codec (docs/trace/spikes/crypto-interop.md).
VAULT_CODEC = "PBKDF2-HMAC-SHA-256+A256GCM"
VAULT_META_FORMAT = "nemoir.trace.vault-meta/0.1"
VAULT_AAD_FORMAT = "nemoir.trace.vault-aad/0.1"
VAULT_KDF_NAME = "PBKDF2-HMAC-SHA-256"
VAULT_KDF_ITERATIONS = 600_000
VAULT_SALT_BYTES = 16
VAULT_NONCE_BYTES = 12
VAULT_DERIVED_KEY_BITS = 256
VAULT_TAG_BITS = 128
VAULT_PLAINTEXT_MEDIA_TYPE = "application/x-ndjson"
# Trace id used in vault AAD when provenance has no IR fingerprint.
VAULT_NULL_IR_SHA256 = "sha256:" + "00" * 32
# Default cap for decrypted vault plaintext (uncompressed NDJSON bytes).
DEFAULT_MAX_VAULT_BYTES = 64 * 1024 * 1024

# Deterministic ZIP profile (matches the Phase 0 fixture assembly).
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
ZIP_DEFLATE_LEVEL = 6
ZIP_UNIX_REGULAR = 0o100644

# Reader limits: local audit/replay viewer budget from schema/README.md §2.
LIMIT_COMPRESSED_BYTES = 64 * 1024 * 1024
LIMIT_UNCOMPRESSED_TOTAL_BYTES = 256 * 1024 * 1024
LIMIT_UNCOMPRESSED_ENTRY_BYTES = 192 * 1024 * 1024
LIMIT_EVENT_COUNT = 500_000
LIMIT_COMPRESSION_RATIO = 200

# JS safe-integer bounds for portable trace semantics (I-JSON).
MAX_SAFE_INT = 9007199254740991
MIN_SAFE_INT = -9007199254740991

# Allowlist bounds (audit-v1 field shapes and redaction-loop guards).
_ERROR_SUFFIX_LEN = 5
_MAX_ERROR_CODE_LEN = 64
_MAX_TOOL_NAME_LEN = 128
_MAX_METHOD_LEN = 16
_MAX_CATEGORY_LEN = 64
_MIN_SECRET_LEN = 8
_MAX_MASK_PASSES = 8


def policy_refs_for(policies: Any) -> dict[str, str]:
    """Opaque policy refs in IR declaration order (``p-1``, ``p-2``, ...).

    Single rule shared by the recorder (cleartext graph/ledger refs) and
    taped replay (resolving vault ``policy_ref`` evidence back to policy
    ids). Policies are spec objects with an ``id`` attribute."""
    refs: dict[str, str] = {}
    for index, policy in enumerate(policies or (), 1):
        policy_id = getattr(policy, "id", None)
        if isinstance(policy_id, str) and policy_id not in refs:
            refs[policy_id] = f"p-{index}"
    return refs


class TraceError(Exception):
    """Raised for trace configuration, projection, scan, or archive failures."""


# ---------------------------------------------------------------------------
# Host-supplied configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelDescriptor:
    """Safe model identity. Credentials, endpoints, and provider extras are
    never representable here by construction."""

    name: str
    api_mode: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class HostProvenance:
    """Static compiler facts supplied by the host (generated package or
    manual runtime user). ``ir_sha256=None`` marks provenance incomplete:
    the trace cannot claim IR binding or semantic verification."""

    frontend: str = "manual"
    target: str = "manual"
    compiler_version: str = "unknown"
    ir_version: str = "0.1"
    ir_sha256: str | None = None


@dataclass(frozen=True)
class VaultCapture:
    """Which private evidence a ``replay``-profile recorder retains.

    The vault is encrypted with the host passphrase, but capture is still
    conservative: credentials never enter the vault (taped replay never
    calls the provider), transport headers are dropped, unregistered
    absolute paths degrade to opaque refs, and reasoning text requires an
    explicit opt-in.
    """

    include_model_messages: bool = True
    include_tool_results: bool = True
    include_stage_snapshots: bool = True
    include_transition_policy: bool = True
    include_reasoning: bool = False
    include_ir: bool = True
    max_vault_bytes: int = DEFAULT_MAX_VAULT_BYTES


@dataclass(frozen=True)
class TraceConfig:
    """Immutable per-run trace configuration owned by the host.

    One recorder instance represents one run; generated agents use a factory
    or a fresh config per invocation, never a shared recorder.
    """

    path: Path
    profile: str = "audit"
    provenance: HostProvenance = field(default_factory=HostProvenance)
    model: ModelDescriptor | None = None
    # Replay-vault passphrase (Phase 4). Required for ``profile="replay"``
    # and rejected for ``audit``. Held in memory only, never serialized.
    # Demo hosts source this from ``env:VAR`` / ``file:PATH`` / ``prompt``.
    vault_passphrase: str | bytes | None = None
    vault_capture: VaultCapture = field(default_factory=VaultCapture)
    # alias (e.g. "$workspace") -> local root the alias stands for.
    path_aliases: Mapping[str, str | Path] = field(default_factory=_empty_alias_map)
    # Aliases whose alias-relative segments may appear in cleartext. Any
    # other alias degrades to an opaque ``$alias/path-N`` ref.
    safe_path_aliases: frozenset[str] = frozenset()
    # Exact ``"StageId.field"`` stage outputs allowed as public scalar metrics.
    approved_metrics: frozenset[str] = frozenset()
    # Known credential *values* for echo defense-in-depth. Never serialized.
    secrets: tuple[str, ...] = ()
    # Test hooks: fixed trace id and clock for byte-identical fixtures.
    trace_id: str | None = None
    clock: Callable[[], datetime] | None = None
    # Narrow synchronous host hook invoked immediately after a
    # successfully observed ``stage_completed`` (Phase 3). The hook receives
    # only ``{"stage_id", "stage_visit_id", "sequence"}`` and may return
    # ``{"namespace", "kind", "payload", "anchor_sequence"?}`` or None.
    # It must never alter workflow control flow; hook failures never break
    # the run. Trace storage/configuration stays here, never in RunOptions.
    on_stage_completed: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def runtime_version() -> str:
    """Best-effort runtime version for provenance (never raises)."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("nemoir-runtime")
    except Exception:
        return "0.0.0+unknown"


def format_timestamp(moment: datetime) -> str:
    """Canonical UTC millisecond timestamp (truncate, never round)."""
    utc = moment.astimezone(UTC)
    millis = utc.microsecond // 1000
    truncated = utc.replace(microsecond=millis * 1000)
    return truncated.strftime("%Y-%m-%dT%H:%M:%S.") + f"{millis:03d}Z"


def _camel_to_snake(name: str) -> str:
    """Split CamelCase boundaries (acronym-aware) and lowercase."""
    step = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step).lower()


def stable_error(exc: BaseException) -> tuple[str, str]:
    """Map an exception to a (stable_code, type_name) pair for cleartext.

    The raw message, traceback, and cause chain never leave this function.
    """
    type_name = type(exc).__name__
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", type_name):
        type_name = "UnknownError"
    stem = (
        type_name[:-_ERROR_SUFFIX_LEN]
        if type_name.endswith("Error") and len(type_name) > _ERROR_SUFFIX_LEN
        else type_name
    )
    code = _camel_to_snake(stem)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", code) or len(code) > _MAX_ERROR_CODE_LEN:
        code = "unknown_error"
    return code, type_name


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (bytes, bytearray)):
        return "binary"
    if isinstance(value, (list, tuple)):
        return "array"
    return "object"


def _result_type_slug(value: Any) -> str:
    """Stable public type name for a tool result (never the value)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (bytes, bytearray)):
        return "binary"
    if isinstance(value, (list, tuple)):
        return "array"
    return "object"


# ---------------------------------------------------------------------------
# Secret registry + cleartext scanner (secrets-v1)
# ---------------------------------------------------------------------------


class _SecretRegistry:
    """In-memory exact-match credential values. Destroyed with the recorder."""

    def __init__(self, secrets: tuple[str, ...]) -> None:
        long_values: set[str] = set()
        short_values: set[str] = set()
        short_patterns: list[re.Pattern[str]] = []
        for secret in secrets:
            if not secret:
                continue
            if len(secret) >= _MIN_SECRET_LEN:
                long_values.add(secret)
                long_values.add(f"Bearer {secret}")
            else:
                # Short secrets: match only complete field or token-delimited
                # occurrences per redaction-policy §4, not arbitrary substrings.
                short_values.add(secret)
                short_patterns.append(
                    re.compile(r"(?<![A-Za-z0-9_\-])" + re.escape(secret) + r"(?![A-Za-z0-9_\-])")
                )
                bearer = f"Bearer {secret}"
                short_values.add(bearer)
                short_patterns.append(
                    re.compile(r"(?<![A-Za-z0-9_\-])" + re.escape(bearer) + r"(?![A-Za-z0-9_\-])")
                )
        self._long = long_values
        self._short = short_values
        self._short_patterns = tuple(short_patterns)

    def find_hit(self, text: str) -> bool:
        for value in self._long:
            if value in text:
                return True
        if not self._short:
            return False
        # Fast pre-check: if no short value is substring, skip regex.
        if not any(s in text for s in self._short):
            return False
        return any(pattern.search(text) for pattern in self._short_patterns)


# Scanner rules that all indicate credential disclosure when matched.
_CREDENTIAL_RULES = frozenset(
    {
        "provider_key",
        "github_token",
        "cloud_key",
        "auth_header",
        "private_key",
        "credential_url",
        "credential_query",
    }
)


_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("provider_key", re.compile(r"sk-(?:ant|nemoir)[A-Za-z0-9\-_]{8,}")),
    ("provider_key", re.compile(r"sk-[A-Za-z0-9\-_]{16,}")),
    ("github_token", re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{8,}")),
    ("github_token", re.compile(r"github_pat_[A-Za-z0-9_\-]{8,}")),
    ("cloud_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("cloud_key", re.compile(r"xox[bap]-[A-Za-z0-9\-]{8,}")),
    ("auth_header", re.compile(r"Bearer\s+\S{8,}")),
    ("auth_header", re.compile(r"Basic\s+[A-Za-z0-9+/=]{8,}")),
    ("private_key", re.compile(r"-----BEGIN .*PRIVATE KEY")),
    ("credential_url", re.compile(r"://[^/\s]*:[^/\s]*@")),
    (
        "credential_query",
        re.compile(r"[?&#](?:api[_-]?key|token|secret|password)=[^&#\s]*", re.IGNORECASE),
    ),
    ("home_path", re.compile(r"/home/[^/\s]*")),
    ("home_path", re.compile(r"/Users/[^/\s]*")),
    ("home_path", re.compile(r"C:[\\/]Users[\\/][^\\/\\s]*")),
    ("home_path", re.compile(r"\\\\[A-Za-z0-9_.$\-]+\\")),
)

# Field names that must never appear in a public record (backstop: the
# projection never emits them, so any occurrence is a leak).
_PROHIBITED_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "extra_headers",
        "reasoning",
        "traceback",
        "stdout",
        "stderr",
    }
)


@dataclass(frozen=True)
class _Finding:
    rule: str
    pointer: str


def _iter_strings(node: Any, pointer: str) -> Iterator[tuple[str, str | None, str | None]]:
    """Yield (pointer, key_or_None, string) for every string in a structure."""
    if isinstance(node, str):
        yield pointer, None, node
    elif isinstance(node, dict):
        mapping = cast("dict[Any, Any]", node)
        for key, val in mapping.items():
            child: str = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            yield child, key, val if isinstance(val, str) else None
            if not isinstance(val, str):
                yield from _iter_strings(val, child)
    elif isinstance(node, (list, tuple)):
        items = cast("Sequence[Any]", node)
        for i, val in enumerate(items):
            yield from _iter_strings(val, f"{pointer}/{i}")


def _has_unsafe_int(node: Any) -> bool:
    """True if any integer value is outside JS safe-integer range."""
    if isinstance(node, bool):
        return False
    if isinstance(node, int):
        return not MIN_SAFE_INT <= node <= MAX_SAFE_INT
    if isinstance(node, float):
        return node.is_integer() and not MIN_SAFE_INT <= int(node) <= MAX_SAFE_INT
    if isinstance(node, dict):
        return any(_has_unsafe_int(v) for v in cast("dict[Any, Any]", node).values())
    if isinstance(node, (list, tuple)):
        return any(_has_unsafe_int(v) for v in cast("Sequence[Any]", node))
    return False


def _scan_strings(node: Any, pointer: str, registry: _SecretRegistry) -> list[_Finding]:
    findings: list[_Finding] = []
    for child_pointer, key, value in _iter_strings(node, pointer):
        if key is not None and key in _PROHIBITED_KEYS:
            # An already-markered value is neutralized: the key name alone
            # (declared in the workflow's public writes schema) is not a
            # leak. Reporting it would loop the mask passes until the whole
            # record is omitted — dropping milestone events taped replay
            # needs for path comparison. Raw values stay reportable below.
            if _is_redaction_marker(_node_at_pointer(node, child_pointer)):
                continue
            findings.append(_Finding(rule="prohibited_field", pointer=child_pointer))
            continue
        if value is None:
            continue
        if registry.find_hit(value):
            findings.append(_Finding(rule="registered_secret", pointer=child_pointer))
            continue
        for rule, pattern in _SECRET_PATTERNS:
            if pattern.search(value):
                findings.append(_Finding(rule=rule, pointer=child_pointer))
                break
    return findings


# Pointers whose values may be replaced by a redaction marker without
# breaking the writer schema. Anything else forces record omission.
_MASKABLE_PREFIXES = (
    "/text",
    "/result",
    "/output/",
    "/annotation/payload/",
    "/args/content",
    "/args/key",
    "/args/value",
    "/args/code",
    "/args/input",
    "/args/headers",
    "/args/body",
    "/args/question",
    "/args/message",
    "/args/options",
)


def _is_maskable(pointer: str) -> bool:
    return (
        pointer in ("/text", "/result")
        or pointer.startswith(("/output/", "/annotation/payload/"))
        or pointer in _MASKABLE_PREFIXES
    )


# ---------------------------------------------------------------------------
# Trusted autoresearch annotation validation (Phase 3)
# ---------------------------------------------------------------------------

_ANNOTATION_NAMESPACE = "nemoir.autoresearch/v1"
_ANNOTATION_KIND = "trial_finished"

_ANNOTATION_VERDICTS = frozenset({"accepted", "rejected", "inconclusive"})

_ANNOTATION_REASONS = frozenset(
    {
        "accepted",
        "no_change",
        "duplicate",
        "preflight_integrity",
        "preflight_scope",
        "preflight_static_scan",
        "preflight_build",
        "preflight_smoke",
        "preflight_sanitizer",
        "selection_correctness",
        "selection_noise",
        "selection_tail_regression",
        "confirmation_correctness",
        "confirmation_noise",
        "confirmation_tail_regression",
        "no_improvement",
        "full_sanitizer",
        "no_evaluation",
        "policy_denied",
        "tool_failed",
        "budget_exhausted",
        "other",
    }
)

_ANNOTATION_METRIC_NUMBERS = frozenset(
    {
        "candidate_median_ns",
        "incumbent_median_ns",
        "delta_ns",
        "effect_ns",
        "speedup_pct",
        "candidate_spread_pct",
        "p95_regression_pct",
        "cold_regression_pct",
    }
)

_ANNOTATION_METRIC_BOOLS = frozenset({"valid", "noise_ok", "regressions_ok"})

_ANNOTATION_REQUIRED = frozenset(
    {
        "trial_id",
        "candidate_ref",
        "verdict",
        "reason_code",
        "selection_metrics",
        "artifact_refs",
    }
)

_ANNOTATION_ALLOWED = frozenset(
    {
        "trial_id",
        "candidate_ref",
        "parent_ref",
        "candidate_digest",
        "parent_digest",
        "selection_metrics",
        "confirmation_metrics",
        "verdict",
        "reason_code",
        "source_reason_code",
        "mechanism_ref",
        "mechanism_id",
        "artifact_refs",
    }
)


def _validate_autoresearch_metrics(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"trial_finished payload has invalid {field}: must be an object"
        raise TraceError(msg)
    mapping = dict(cast("Mapping[str, Any]", value))
    for key in mapping:
        if key not in _ANNOTATION_METRIC_NUMBERS and key not in _ANNOTATION_METRIC_BOOLS:
            msg = f"trial_finished payload has unknown metrics field {key!r} in {field}"
            raise TraceError(msg)
    for key in _ANNOTATION_METRIC_NUMBERS:
        if key not in mapping:
            continue
        number = mapping[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            msg = f"trial_finished payload has invalid {field}.{key}: must be a number"
            raise TraceError(msg)
        if isinstance(number, float) and not math.isfinite(number):
            msg = f"trial_finished payload has non-finite {field}.{key}"
            raise TraceError(msg)
        if isinstance(number, int) and not MIN_SAFE_INT <= number <= MAX_SAFE_INT:
            msg = f"trial_finished payload has unsafe integer {field}.{key}"
            raise TraceError(msg)
        if isinstance(number, float) and number.is_integer():
            as_int = int(number)
            if not MIN_SAFE_INT <= as_int <= MAX_SAFE_INT:
                msg = f"trial_finished payload has unsafe integer {field}.{key}"
                raise TraceError(msg)
    for key in _ANNOTATION_METRIC_BOOLS:
        if key not in mapping:
            continue
        if not isinstance(mapping[key], bool):
            msg = f"trial_finished payload has invalid {field}.{key}: must be a boolean"
            raise TraceError(msg)
    return mapping


def _validate_autoresearch_payload(payload: Any) -> dict[str, Any]:
    """Validate a ``trial_finished`` payload and return a plain copy.

    Raises :class:`TraceError` on any shape violation. Raw prose, patches,
    paths, and digests are rejected by the adapter contract; the recorder
    enforces shape, finite numbers, safe integers, opaque-ref patterns, and
    closed field sets. Secret scanning still applies after validation.
    """
    if not isinstance(payload, Mapping):
        msg = "trial_finished payload must be an object"
        raise TraceError(msg)
    data = dict(cast("Mapping[str, Any]", payload))
    unknown = set(data) - _ANNOTATION_ALLOWED
    if unknown:
        msg = f"trial_finished payload has unknown fields {sorted(unknown)}"
        raise TraceError(msg)
    missing = _ANNOTATION_REQUIRED - set(data)
    if missing:
        msg = f"trial_finished payload is missing fields {sorted(missing)}"
        raise TraceError(msg)
    trial_id = data.get("trial_id")
    if (
        isinstance(trial_id, bool)
        or not isinstance(trial_id, int)
        or trial_id < 1
        or not MIN_SAFE_INT <= trial_id <= MAX_SAFE_INT
    ):
        msg = f"trial_finished payload has invalid trial_id {trial_id!r}"
        raise TraceError(msg)
    candidate_ref = data.get("candidate_ref")
    if not isinstance(candidate_ref, str) or not re.fullmatch(
        r"candidate-[1-9][0-9]*", candidate_ref
    ):
        msg = f"trial_finished payload has invalid candidate_ref {candidate_ref!r}"
        raise TraceError(msg)
    parent_ref = data.get("parent_ref")
    if parent_ref is not None and (
        not isinstance(parent_ref, str) or not re.fullmatch(r"candidate-[1-9][0-9]*", parent_ref)
    ):
        msg = f"trial_finished payload has invalid parent_ref {parent_ref!r}"
        raise TraceError(msg)
    for digest_key in ("candidate_digest", "parent_digest"):
        digest = data.get(digest_key)
        if digest is not None and (
            not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            msg = f"trial_finished payload has invalid {digest_key} {digest!r}"
            raise TraceError(msg)
    data["selection_metrics"] = _validate_autoresearch_metrics(
        data.get("selection_metrics"), field="selection_metrics"
    )
    confirmation = data.get("confirmation_metrics")
    if confirmation is not None:
        data["confirmation_metrics"] = _validate_autoresearch_metrics(
            confirmation, field="confirmation_metrics"
        )
    verdict = data.get("verdict")
    if verdict not in _ANNOTATION_VERDICTS:
        msg = f"trial_finished payload has invalid verdict {verdict!r}"
        raise TraceError(msg)
    reason = data.get("reason_code")
    if reason not in _ANNOTATION_REASONS:
        msg = f"trial_finished payload has invalid reason_code {reason!r}"
        raise TraceError(msg)
    source_reason = data.get("source_reason_code")
    if source_reason is not None and (
        not isinstance(source_reason, str)
        or len(source_reason) > 64
        or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", source_reason)
    ):
        msg = f"trial_finished payload has invalid source_reason_code {source_reason!r}"
        raise TraceError(msg)
    mechanism_ref = data.get("mechanism_ref")
    if mechanism_ref is not None and (
        not isinstance(mechanism_ref, str)
        or not re.fullmatch(r"mechanism-[1-9][0-9]*", mechanism_ref)
    ):
        msg = f"trial_finished payload has invalid mechanism_ref {mechanism_ref!r}"
        raise TraceError(msg)
    mechanism_id = data.get("mechanism_id")
    if mechanism_id is not None and (
        not isinstance(mechanism_id, str) or not 1 <= len(mechanism_id) <= 128
    ):
        msg = "trial_finished payload has invalid mechanism_id"
        raise TraceError(msg)
    artifact_refs = data.get("artifact_refs")
    if not isinstance(artifact_refs, (list, tuple)) or isinstance(artifact_refs, (str, bytes)):
        msg = "trial_finished payload has invalid artifact_refs: must be an array"
        raise TraceError(msg)
    refs = list(cast("list[Any] | tuple[Any, ...]", artifact_refs))
    if len(set(refs)) != len(refs):
        msg = "trial_finished payload has duplicate artifact_refs"
        raise TraceError(msg)
    for ref in refs:
        if not isinstance(ref, str) or not re.fullmatch(r"artifact-[1-9][0-9]*", ref):
            msg = f"trial_finished payload has invalid artifact_ref {ref!r}"
            raise TraceError(msg)
    data["artifact_refs"] = refs
    return data


# ---------------------------------------------------------------------------
# Phase 4 replay-vault helpers (passphrase, scrubbing, codec)
# ---------------------------------------------------------------------------


def _normalize_passphrase(value: str | bytes | None) -> bytes | None:
    """Normalize a vault passphrase to NFC UTF-8 bytes (spike §1)."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            msg = "vault passphrase must be valid UTF-8"
            raise TraceError(msg) from exc
    else:
        text = value
    if not text:
        return None
    return unicodedata.normalize("NFC", text).encode("utf-8")


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str, *, what: str, expected: int | None = None) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))
    except (ValueError, binascii.Error) as exc:
        msg = f"vault metadata has invalid base64url for {what}"
        raise TraceError(msg) from exc
    if expected is not None and len(raw) != expected:
        msg = f"vault metadata has invalid length for {what}"
        raise TraceError(msg)
    return raw


def _to_jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Convert runtime values to JCS-safe plain JSON.

    Mappings/sequences/dataclasses become plain dicts/lists; sets and
    frozensets become sorted lists; Paths/UUIDs/datetimes become strings;
    bytes become UTF-8 (replacement) strings; Enums become values.
    Rejects NaN/infinity and unsafe integers like the public ledger does.
    """
    if _depth > 64:
        msg = "vault value exceeds nesting depth"
        raise TraceError(msg)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if not MIN_SAFE_INT <= value <= MAX_SAFE_INT:
            msg = f"vault value has unsafe integer {value!r}"
            raise TraceError(msg)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = "vault value has non-finite number"
            raise TraceError(msg)
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return format_timestamp(value)
    if isinstance(value, uuid.UUID):
        return value.hex
    if isinstance(value, Enum):
        return _to_jsonable(value.value, _depth=_depth + 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_jsonable(getattr(value, f.name), _depth=_depth + 1)
            for f in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        mapping = cast("Mapping[Any, Any]", value)
        items: dict[str, Any] = {}
        for raw_key, raw_item in mapping.items():
            name = raw_key
            items[str(name) if not isinstance(name, str) else name] = _to_jsonable(
                raw_item, _depth=_depth + 1
            )
        return items
    if isinstance(value, (set, frozenset)):
        members = cast("Any", value)
        return sorted(
            (_to_jsonable(item, _depth=_depth + 1) for item in members),
            key=lambda v: json.dumps(v, sort_keys=True, default=str),
        )
    if isinstance(value, (list, tuple)):
        sequence = cast("Any", value)
        return [_to_jsonable(item, _depth=_depth + 1) for item in sequence]
    # Last resort for opaque objects: never str()/repr() provider clients or
    # credentials — reject instead of guessing.
    msg = f"vault value of type {type(value).__name__} is not serializable"
    raise TraceError(msg)


# Vault-excluded mapping keys (case-insensitive): transport credentials and
# provider options that taped replay never needs. Values become opaque
# credential markers instead of blocking finalization.
_VAULT_CREDENTIAL_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "refresh_token",
        "client_secret",
        "extra_headers",
        "default_headers",
        "headers",
    }
)

_VAULT_RECORD_TYPES = frozenset(
    {
        "run_inputs",
        "stage_snapshot",
        "model_request",
        "model_response",
        "tool_result",
        "transition_evaluation",
        "policy_evaluation",
        "private_fields",
        "full_workflow_ir",
        "annotation_private_fields",
    }
)


def _node_at_pointer(node: Any, pointer: str) -> Any:
    """Return the value at an RFC 6901 pointer, or None when absent."""
    if pointer in ("", "/"):
        return node
    current = node
    for part in pointer.split("/")[1:]:
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            branch = cast("Any", current)
            if key not in branch:
                return None
            current = branch[key]
        elif isinstance(current, list):
            sequence = cast("Any", current)
            if not key.isdigit() or int(key) >= len(sequence):
                return None
            current = sequence[int(key)]
        else:
            return None
    return current


def _is_redaction_marker(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if not isinstance(value, dict):
        return False
    branch = cast("dict[Any, Any]", value)
    return isinstance(branch.get("$redacted"), Mapping)


def _parse_ledger_line(line: bytes) -> Any:
    """Parse one ledger line, returning None when it is not valid JSON."""
    try:
        return parse_json_strict(line.decode("utf-8"))
    except Exception:
        return None


def _vault_meta_object(
    *,
    salt: bytes,
    nonce: bytes,
    aad_dict: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": VAULT_META_FORMAT,
        "codec": VAULT_CODEC,
        "passphrase": {"encoding": "UTF-8", "normalization": "NFC"},
        "kdf": {
            "name": VAULT_KDF_NAME,
            "iterations": VAULT_KDF_ITERATIONS,
            "salt_base64url": _b64url_encode(salt),
            "derived_key_bits": VAULT_DERIVED_KEY_BITS,
        },
        "cipher": {
            "name": "AES-256-GCM",
            "nonce_base64url": _b64url_encode(nonce),
            "tag_length_bits": VAULT_TAG_BITS,
            "tag_placement": "ciphertext_suffix",
        },
        "plaintext": {
            "media_type": VAULT_PLAINTEXT_MEDIA_TYPE,
            "encoding": "UTF-8",
            "compression": "none",
        },
        "aad": aad_dict,
    }


def _vault_primitives() -> tuple[Any, Any]:
    """Return ``(AESGCM, InvalidTag)`` from the optional ``trace`` extra."""
    try:
        from cryptography.exceptions import InvalidTag  # noqa: PLC0415
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "trace profile 'replay' requires the optional 'trace' extra "
            "(pip install 'nemoir-runtime[trace]'): " + str(exc)
        )
        raise TraceError(msg) from exc
    return AESGCM, InvalidTag


def _derive_vault_key(passphrase: bytes, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase, salt, VAULT_KDF_ITERATIONS, dklen=32)


def encrypt_vault_records(
    plaintext: bytes,
    passphrase: bytes,
    aad: bytes,
    *,
    salt: bytes | None = None,
    nonce: bytes | None = None,
) -> tuple[bytes, bytes, bytes]:
    """Seal vault plaintext; returns ``(salt, nonce, ciphertext+tag)``."""
    aes_gcm_cls, _ = _vault_primitives()
    salt = salt if salt is not None else os.urandom(VAULT_SALT_BYTES)
    nonce = nonce if nonce is not None else os.urandom(VAULT_NONCE_BYTES)
    if len(salt) != VAULT_SALT_BYTES or len(nonce) != VAULT_NONCE_BYTES:
        msg = "vault salt/nonce have invalid length"
        raise TraceError(msg)
    key = _derive_vault_key(passphrase, salt)
    return salt, nonce, aes_gcm_cls(key).encrypt(nonce, plaintext, aad)


def decrypt_vault_records(
    sealed: bytes,
    passphrase: bytes,
    aad: bytes,
    *,
    salt: bytes,
    nonce: bytes,
) -> bytes:
    """Open vault ciphertext; fails closed with a generic error."""
    aes_gcm_cls, invalid_tag = _vault_primitives()
    if len(salt) != VAULT_SALT_BYTES or len(nonce) != VAULT_NONCE_BYTES:
        msg = "vault unlock failed"
        raise TraceError(msg)
    if len(sealed) < 16:
        msg = "vault unlock failed"
        raise TraceError(msg)
    key = _derive_vault_key(passphrase, salt)
    try:
        return aes_gcm_cls(key).decrypt(nonce, sealed, aad)
    except invalid_tag as exc:
        # Wrong passphrase and modified ciphertext/AAD are
        # indistinguishable by design; never report which one failed.
        msg = "vault unlock failed"
        raise TraceError(msg) from exc


# ---------------------------------------------------------------------------
# Provenance resource loading (generated packages)
# ---------------------------------------------------------------------------


def load_generated_provenance(package_name: str) -> HostProvenance:
    """Load and verify a generated package's trace provenance resources.

    Reads ``workflow.json`` + ``trace-provenance.json`` via
    :mod:`importlib.resources`, recomputes the canonical IR hash, and returns
    a :class:`HostProvenance`. A hash mismatch (or any load failure) yields
    *incomplete* provenance rather than raising: the trace stays honest about
    what it cannot bind instead of crashing the run.
    """
    incomplete = HostProvenance()
    try:
        from importlib import resources  # noqa: PLC0415

        workflow_text = (
            resources.files(package_name).joinpath("workflow.json").read_text(encoding="utf-8")
        )
        provenance_text = (
            resources.files(package_name)
            .joinpath("trace-provenance.json")
            .read_text(encoding="utf-8")
        )
        workflow_value = parse_json_strict(workflow_text)
        actual = sha256_tag(to_canonical_bytes(workflow_value))
        claimed: dict[str, Any] = parse_json_strict(provenance_text)
        if claimed.get("format") != PROVENANCE_FORMAT:
            return incomplete
        if claimed.get("ir_sha256") != actual:
            return incomplete
        ir_sha256 = claimed.get("ir_sha256")
        if not isinstance(ir_sha256, str):
            return incomplete
        frontend = claimed.get("frontend")
        compiler_version = claimed.get("compiler_version")
        ir_version = claimed.get("ir_version", "0.1")
        target = claimed.get("target", "python")
        return HostProvenance(
            frontend=frontend if isinstance(frontend, str) else "unknown",
            target=target if isinstance(target, str) else "python",
            compiler_version=compiler_version if isinstance(compiler_version, str) else "unknown",
            ir_version=ir_version if isinstance(ir_version, str) else "0.1",
            ir_sha256=ir_sha256,
        )
    except Exception:
        return incomplete


def safe_model_descriptor(model: Any) -> ModelDescriptor | None:
    """Build a safe model descriptor from a host model spec (allowlist only).

    Accepts mappings (``{"name": ..., "temperature": ..., ...}``) and adapter
    objects exposing ``name``/``temperature``/``max_tokens`` attributes.
    Anything else yields None rather than a guessed descriptor.
    """
    name: Any = None
    api_mode: Any = None
    temperature: Any = None
    max_tokens: Any = None
    if isinstance(model, Mapping):
        mapping = cast("Mapping[str, Any]", model)
        name = mapping.get("name")
        api_mode = mapping.get("api", mapping.get("api_mode"))
        temperature = mapping.get("temperature")
        max_tokens = mapping.get("max_tokens")
    else:
        name = getattr(model, "name", None)
        api_mode = getattr(model, "api", None)
        temperature = getattr(model, "temperature", None)
        max_tokens = getattr(model, "max_tokens", None)
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(api_mode, str):
        api_mode = None
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        temperature = None
    else:
        temperature = float(temperature)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        max_tokens = None
    return ModelDescriptor(
        name=name[:256],
        api_mode=api_mode[:64] if api_mode else None,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Trace recorder
# ---------------------------------------------------------------------------


def _vault_record_shape_error(record: Any) -> str | None:
    """Return a shape violation for a vault record, or None when valid."""
    if not isinstance(record, dict):
        return "vault record must be an object"
    entry = cast("dict[Any, Any]", record)
    if not re.fullmatch(r"v-[1-9][0-9]*", str(entry.get("record_id", ""))):
        return f"vault record has invalid record_id {entry.get('record_id')!r}"
    if entry.get("record_type") not in _VAULT_RECORD_TYPES:
        return f"vault record has invalid record_type {entry.get('record_type')!r}"
    return None


class TraceRecorder:
    """Redacted audit recorder for one run. See module docstring for the
    security model. Obtain via :meth:`create`; exactly one ``begin_run`` then
    at most one ``finish_run`` per instance."""

    def __init__(self, config: TraceConfig) -> None:
        if config.profile not in ("audit", "replay"):
            msg = (
                f"unsupported trace profile '{config.profile}': expected "
                f"'audit' or 'replay' (publication arrives in Phase 5)"
            )
            raise TraceError(msg)
        self._vault_enabled = config.profile == "replay"
        if self._vault_enabled and config.vault_passphrase in (None, "", b""):
            msg = "trace profile 'replay' requires vault_passphrase"
            raise TraceError(msg)
        if not self._vault_enabled and config.vault_passphrase not in (None, "", b""):
            msg = "vault_passphrase requires trace profile 'replay'"
            raise TraceError(msg)
        self._vault_passphrase = _normalize_passphrase(config.vault_passphrase)
        self._config = config
        self._registry = _SecretRegistry(config.secrets)
        self._clock = config.clock or (lambda: datetime.now(tz=UTC))
        self._trace_id = config.trace_id or uuid.uuid4().hex
        if not re.fullmatch(r"[0-9a-f]{32}", self._trace_id):
            msg = f"trace id must be 32 lowercase hex, got {self._trace_id!r}"
            raise TraceError(msg)
        self._roots: list[tuple[str, Path]] = []
        for alias, root in config.path_aliases.items():
            if not alias.startswith("$"):
                msg = f"path alias must start with '$', got {alias!r}"
                raise TraceError(msg)
            try:
                resolved = Path(root).resolve(strict=False)
            except OSError as exc:
                msg = f"cannot resolve path alias root {alias!r}: {exc}"
                raise TraceError(msg) from exc
            self._roots.append((alias, resolved))
        self._roots.sort(key=lambda item: len(str(item[1])), reverse=True)
        self._begun = False
        self._finished = False
        self._begin_time: datetime | None = None
        self._manifest: Any = None
        self._policy_refs: dict[str, str] = {}
        self._events: list[dict[str, Any]] = []
        self._pending_visits: dict[str, list[str]] = {}
        self._pending_models: dict[str, list[str]] = {}
        # Last model-call id bound to a ledger record in each visit. A retry
        # emitted after model_completed consumed the pending id reuses this
        # instead of synthesizing a dangling id with no vault evidence.
        self._last_model_id: dict[str, str] = {}
        # Optional taped policy outcomes for replay (see replay.py). Maps a
        # policy id to its recorded outcomes in capture order; consumed
        # FIFO by the runtime instead of re-evaluating expressions against
        # scrubbed fixture args. None (default) means live evaluation.
        self.policy_tape: dict[str, list[str]] | None = None
        self._pending_tools: dict[str, list[str]] = {}
        self._open_tools: dict[str, list[str]] = {}
        self._current_visit: str | None = None
        self._current_stage_id: str | None = None
        self._visit_to_stage: dict[str, str] = {}
        self._visit_sequences: dict[str, list[int]] = {}
        self._annotations_dropped = 0
        self._annotation_warnings: list[str] = []
        self._model_bytes: dict[str, int] = {}
        self._model_tool_calls: dict[str, int] = {}
        self._tool_started_at: dict[str, datetime] = {}
        self._tool_result_types: dict[str, str] = {}
        self._tool_errors: dict[str, tuple[str, str]] = {}
        self._transition_evidence: list[dict[str, Any]] = []
        # Phase 4 vault state (populated only when profile == "replay").
        self._open_tool_calls: dict[str, dict[str, Any]] = {}
        self._vault_records: list[dict[str, Any]] = []
        self._vault_count = 0
        self._pending_transitions: dict[str, list[dict[str, Any]]] = {}
        self._visit_count = 0
        self._model_count = 0
        self._tool_count = 0
        self._redaction_count = 0
        self._path_ref_count = 0
        self._path_refs: dict[str, str] = {}
        self._stage_completed_count = 0
        self._event_limit_exceeded = False

    # -- construction ----------------------------------------------------

    @property
    def vault_enabled(self) -> bool:
        """True when this recorder captures an encrypted replay vault."""
        return self._vault_enabled

    def consume_taped_policy(self, policy_id: Any) -> str | None:
        """Pop the next recorded outcome for a deny-policy check, if taped.

        Taped replay sets :attr:`policy_tape` so the runtime reproduces
        recorded allow/deny outcomes instead of re-evaluating expressions
        against scrubbed fixture arguments (which would wrongly deny
        path-scoped policies). Returns ``"allowed"``/``"denied"`` or
        None when no tape covers this check (live evaluation proceeds).
        """
        tape = self.policy_tape
        if tape is None or not isinstance(policy_id, str):
            return None
        queue = tape.get(policy_id)
        if not queue:
            return None
        return queue.pop(0)

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        profile: str = "audit",
        provenance: HostProvenance | None = None,
        model: ModelDescriptor | None = None,
        path_aliases: Mapping[str, str | Path] | None = None,
        safe_path_aliases: frozenset[str] | set[str] | None = None,
        approved_metrics: frozenset[str] | set[str] | None = None,
        secrets: tuple[str, ...] | list[str] | None = None,
        trace_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
        on_stage_completed: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
        vault_passphrase: str | bytes | None = None,
        vault_capture: VaultCapture | None = None,
    ) -> TraceRecorder:
        """Create a recorder for one run writing to ``path`` on finish."""
        return cls(
            TraceConfig(
                path=Path(path),
                profile=profile,
                provenance=provenance or HostProvenance(),
                model=model,
                path_aliases=dict(path_aliases or {}),
                safe_path_aliases=frozenset(safe_path_aliases or ()),
                approved_metrics=frozenset(approved_metrics or ()),
                secrets=tuple(secrets or ()),
                trace_id=trace_id,
                clock=clock,
                on_stage_completed=on_stage_completed,
                vault_passphrase=vault_passphrase,
                vault_capture=vault_capture or VaultCapture(),
            )
        )

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def config(self) -> TraceConfig:
        """The immutable host configuration this recorder was created with."""
        return self._config

    @property
    def annotations_dropped(self) -> int:
        """Counted hook annotations lost (review item 1 completeness)."""
        return self._annotations_dropped

    @property
    def annotation_warnings(self) -> tuple[str, ...]:
        """Bounded safe warnings for dropped hook annotations."""
        return tuple(self._annotation_warnings)

    # -- run lifecycle ---------------------------------------------------

    def begin_run(self, manifest: Any) -> None:
        """Attach the workflow manifest before ``run_started`` is observed."""
        if self._begun:
            msg = "TraceRecorder.begin_run called twice: one recorder per run"
            raise TraceError(msg)
        self._begun = True
        self._begin_time = self._clock()
        self._manifest = manifest
        self._policy_refs = policy_refs_for(getattr(manifest, "policies", ()))
        self._write_partial_marker("in_progress")

    def finish_run(self, status: str) -> Path:
        """Finalize and atomically write the ``*.nemotrace`` archive."""
        if not self._begun:
            msg = "TraceRecorder.finish_run without begin_run"
            raise TraceError(msg)
        if self._finished:
            msg = "TraceRecorder.finish_run called twice"
            raise TraceError(msg)
        if status not in ("complete", "failed", "interrupted"):
            msg = f"unknown trace status {status!r}"
            raise TraceError(msg)
        if self._event_limit_exceeded or len(self._events) > LIMIT_EVENT_COUNT:
            self._remove_partial_marker()
            msg = f"trace event count limit exceeded ({LIMIT_EVENT_COUNT})"
            raise TraceError(msg)
        self._finished = True
        entries = self._build_entries(status)
        self._final_scan(entries)
        out = self._config.path
        self._write_archive(out, entries)
        self._remove_partial_marker()
        return out

    # -- semantic hooks (called by the runtime at execution boundaries) ---

    def begin_stage_visit(self, stage_id: str) -> str:
        """Assign the next run-local stage-visit id for ``stage_id``."""
        self._require_begun()
        self._visit_count += 1
        visit_id = f"s-{self._visit_count}"
        self._pending_visits.setdefault(stage_id, []).append(visit_id)
        self._current_visit = visit_id
        self._current_stage_id = stage_id
        self._visit_to_stage[visit_id] = stage_id
        return visit_id

    def begin_model_call(self, _stage_id: str = "", stage_visit_id: str | None = None) -> str:
        """Assign the next run-local model-call id."""
        self._require_begun()
        self._model_count += 1
        call_id = f"m-{self._model_count}"
        visit = stage_visit_id or self._current_visit or ""
        self._pending_models.setdefault(visit, []).append(call_id)
        return call_id

    def record_model_request(self, model_call_id: str, request: Mapping[str, Any] | None) -> None:
        """Retain a model request for the encrypted replay vault.

        The public ledger keeps only counts; messages, tool schemas, and
        output schemas enter the vault (scrubbed) when ``profile='replay'``
        and are dropped otherwise. Credentials are never retained.
        """
        self._require_begun()
        if not self._vault_enabled or request is None:
            return
        capture = self._config.vault_capture
        if not capture.include_model_messages:
            return
        payload = _to_jsonable(dict(request))
        if not isinstance(payload, dict):
            msg = "model request must be an object"
            raise TraceError(msg)
        scrubbed = self._scrub_vault_value(payload)
        self._append_vault_record(
            "model_request",
            scrubbed,
            model_call_id=model_call_id,
            stage_visit_id=self._current_visit,
        )

    def record_model_response(
        self,
        model_call_id: str,
        *,
        response_bytes: int = 0,
        tool_call_count: int = 0,
        response: Mapping[str, Any] | None = None,
    ) -> None:
        """Record safe model-response facts (counts only, never text).

        When ``profile='replay'`` and ``response`` is supplied, the full
        content/tool-calls/usage also enter the encrypted vault (reasoning
        only with an explicit opt-in).
        """
        self._require_begun()
        self._model_bytes[model_call_id] = max(0, int(response_bytes))
        self._model_tool_calls[model_call_id] = max(0, int(tool_call_count))
        if not self._vault_enabled or response is None:
            return
        capture = self._config.vault_capture
        if not capture.include_model_messages:
            return
        payload = _to_jsonable(dict(response))
        if not isinstance(payload, dict):
            msg = "model response must be an object"
            raise TraceError(msg)
        model_response = cast("dict[Any, Any]", payload)
        if not capture.include_reasoning:
            reasoning = model_response.pop("reasoning", None)
            if reasoning not in (None, "", [], {}):
                model_response["reasoning"] = self._new_marker("private_content", reasoning)
        scrubbed = self._scrub_vault_value(model_response)
        self._append_vault_record(
            "model_response",
            scrubbed,
            model_call_id=model_call_id,
            stage_visit_id=self._current_visit,
        )

    def begin_tool_call(
        self,
        _stage_id: str = "",
        stage_visit_id: str | None = None,
    ) -> str:
        """Assign the next run-local tool-call id."""
        self._require_begun()
        self._tool_count += 1
        call_id = f"t-{self._tool_count}"
        visit = stage_visit_id or self._current_visit or ""
        self._pending_tools.setdefault(visit, []).append(call_id)
        self._tool_started_at[call_id] = self._clock()
        return call_id

    def record_tool_result(
        self,
        tool_call_id: str,
        result: Any,
        args: Mapping[str, Any] | None = None,
    ) -> None:
        """Note a tool result's safe type facts (the value stays private).

        When ``profile='replay'`` the full args/result also enter the
        encrypted vault (scrubbed): replay fixtures need real values, but
        transport credentials and echoed secrets still never enter the vault.
        """
        self._require_begun()
        self._tool_errors.pop(tool_call_id, None)
        self._tool_result_types[tool_call_id] = _result_type_slug(result)
        if not self._vault_enabled:
            return
        capture = self._config.vault_capture
        if not capture.include_tool_results:
            return
        stashed = self._open_tool_calls.get(tool_call_id, {})
        effective_args = args if args is not None else stashed.get("args")
        payload: dict[str, Any] = {"result": _to_jsonable(result)}
        if effective_args is not None:
            payload["args"] = _to_jsonable(dict(effective_args))
        if stashed.get("capability") is not None:
            payload["capability"] = stashed["capability"]
        if stashed.get("tool_name") is not None:
            payload["tool_name"] = stashed["tool_name"]
        scrubbed = self._scrub_vault_value(payload)
        if not isinstance(scrubbed, dict):
            msg = "tool result vault payload must be an object"
            raise TraceError(msg)
        self._append_vault_record(
            "tool_result",
            scrubbed,
            tool_call_id=tool_call_id,
            stage_visit_id=self._current_visit,
        )

    def record_run_inputs(self, inputs: Mapping[str, Any] | None) -> None:
        """Retain workflow inputs for the encrypted replay vault."""
        self._require_begun()
        if not self._vault_enabled or inputs is None:
            return
        payload = _to_jsonable(dict(inputs))
        scrubbed = self._scrub_vault_value(payload)
        self._append_vault_record("run_inputs", scrubbed)

    def record_policy_evaluation(
        self,
        stage_visit_id: str | None,
        policy_id: Any,
        bound: Mapping[str, Any] | None,
        outcome: str,
    ) -> None:
        """Retain policy-evaluation evidence for the encrypted vault.

        ``outcome`` is ``"allowed"`` or ``"denied"``. Bound trigger-arg
        values are scrubbed (paths aliased, secrets/credentials removed).
        """
        self._require_begun()
        if not self._vault_enabled:
            return
        capture = self._config.vault_capture
        if not capture.include_transition_policy:
            return
        if outcome not in ("allowed", "denied"):
            msg = f"policy evaluation outcome must be allowed/denied, got {outcome!r}"
            raise TraceError(msg)
        ref = self._policy_ref(policy_id)
        payload: dict[str, Any] = {
            "policy_ref": ref,
            "outcome": outcome,
            "bound": self._scrub_vault_value(_to_jsonable(dict(bound or {}))),
        }
        if ref is None:
            payload["policy_ref"] = self._new_marker("private_content", policy_id)
        self._append_vault_record(
            "policy_evaluation",
            payload,
            stage_visit_id=stage_visit_id or self._current_visit,
        )

    def record_tool_error(self, tool_call_id: str, exc: BaseException) -> None:
        """Capture a tool failure's stable error taxonomy (no message)."""
        self._require_begun()
        code, type_name = stable_error(exc)
        self._tool_errors[tool_call_id] = (code, type_name)
        if not self._vault_enabled:
            return
        capture = self._config.vault_capture
        if not capture.include_tool_results:
            return
        try:
            message = str(exc)[:4000]
        except Exception:
            message = ""
        stashed = self._open_tool_calls.get(tool_call_id, {})
        payload: dict[str, Any] = {
            "error": {"code": code, "type": type_name},
            "message": self._scrub_vault_value(message),
        }
        if stashed.get("args") is not None:
            payload["args"] = self._scrub_vault_value(_to_jsonable(dict(stashed["args"])))
        if stashed.get("capability") is not None:
            payload["capability"] = stashed["capability"]
        self._append_vault_record(
            "tool_result",
            payload,
            tool_call_id=tool_call_id,
            stage_visit_id=self._current_visit,
        )

    def record_transition_evaluation(
        self,
        stage_visit_id: str,
        candidates: list[dict[str, Any]],
    ) -> None:
        """Retain guard-evaluation evidence for semantic verification.

        The audit ledger publishes only the selected transition. When
        ``profile='replay'`` the full ordered candidate list (with per-guard
        match results) enters the encrypted vault, linked to the following
        ``transition_selected`` ledger sequence for that visit.
        """
        self._require_begun()
        cleaned: list[dict[str, Any]] = []
        for raw_candidate in cast("Any", candidates):
            if not isinstance(raw_candidate, Mapping):
                msg = "transition candidate must be an object"
                raise TraceError(msg)
            candidate_mapping = cast("dict[Any, Any]", raw_candidate)
            to = candidate_mapping.get("to")
            if not isinstance(to, str):
                msg = "transition candidate requires a string 'to'"
                raise TraceError(msg)
            cleaned.append(
                {
                    "to": to,
                    "priority": max(0, int(candidate_mapping.get("priority", 0) or 0)),
                    "reason": candidate_mapping.get("reason", "other"),
                    "matched": bool(candidate_mapping.get("matched", False)),
                }
            )
        self._transition_evidence.append({"stage_visit_id": stage_visit_id, "candidates": cleaned})
        if not self._vault_enabled:
            return
        capture = self._config.vault_capture
        if not capture.include_transition_policy:
            return
        pending = self._pending_transitions.setdefault(stage_visit_id, [])
        pending.append({"candidates": cleaned})

    # -- replay-vault capture ------------------------------------------

    def _append_vault_record(
        self,
        record_type: str,
        payload: Any,
        *,
        event_sequence: int | None = None,
        stage_visit_id: str | None = None,
        model_call_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Append one supplemental vault record (capture order is stable)."""
        if record_type not in _VAULT_RECORD_TYPES:
            msg = f"unknown vault record type {record_type!r}"
            raise TraceError(msg)
        self._vault_count += 1
        record: dict[str, Any] = {
            "record_id": f"v-{self._vault_count}",
            "record_type": record_type,
            "payload": payload,
        }
        if event_sequence is not None:
            record["event_sequence"] = event_sequence
        if stage_visit_id is not None:
            record["stage_visit_id"] = stage_visit_id
        if model_call_id is not None:
            record["model_call_id"] = model_call_id
        if tool_call_id is not None:
            record["tool_call_id"] = tool_call_id
        self._vault_records.append(record)
        return record

    def _scrub_vault_value(self, value: Any) -> Any:
        """Scrub a vault-bound value before it reaches vault plaintext.

        Drops transport credentials (headers/auth/api keys) to opaque
        markers, masks registered-secret echoes, and degrades unregistered
        absolute paths to opaque refs. Unlike cleartext projection this
        never omits the enclosing record: a credential-bearing tool result
        becomes a marker payload so replay fails closed with a clear
        fixture error instead of silently shifting fixture order.
        """
        scrubbed = self._scrub_vault_node(value)
        wrapper: dict[str, Any] = {"v": scrubbed}
        for _ in range(_MAX_MASK_PASSES):
            hits = [
                f
                for f in _scan_strings(wrapper, "", self._registry)
                if f.rule == "registered_secret"
            ]
            if not hits:
                break
            pointer = hits[0].pointer
            self._set_marker(wrapper, pointer, "credential")
        else:
            wrapper["v"] = self._new_marker("credential", wrapper.get("v"))
        for _ in range(_MAX_MASK_PASSES):
            findings = [
                f
                for f in _scan_strings(wrapper, "", self._registry)
                if f.rule != "registered_secret" and not self._vault_finding_excused(wrapper, f)
            ]
            if not findings:
                break
            finding = findings[0]
            if finding.rule in _CREDENTIAL_RULES:
                reason = "credential"
            elif finding.rule == "home_path":
                reason = "absolute_path"
            else:
                reason = "unapproved_field"
            self._set_marker(wrapper, finding.pointer, reason)
        else:
            wrapper["v"] = self._new_marker("credential", wrapper.get("v"))
        wrapper.pop("_omit", None)
        return wrapper["v"]

    def _vault_finding_excused(self, record: dict[str, Any], finding: Any) -> bool:
        """Whether a scanner finding needs no action in vault plaintext.

        Already-markered values are resolved. ``prohibited_field`` findings
        for non-credential keys (``reasoning`` with opt-in capture,
        ``stdout``/``stderr`` tool evidence, ...) are legal vault contents:
        the cleartext key policy does not apply inside the encrypted vault,
        and secret/credential patterns are still enforced separately.
        """
        if _is_redaction_marker(_node_at_pointer(record, finding.pointer)):
            return True
        if finding.rule == "prohibited_field":
            key = finding.pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
            return key.lower() not in _VAULT_CREDENTIAL_KEYS
        return False

    def _scrub_vault_node(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            mapping = cast("Mapping[Any, Any]", value)
            items: dict[str, Any] = {}
            for raw_key, raw_item in mapping.items():
                node_key = raw_key
                lowered = node_key.lower() if isinstance(node_key, str) else ""
                if lowered in _VAULT_CREDENTIAL_KEYS:
                    items[node_key] = self._new_marker("credential", raw_item)
                else:
                    items[node_key] = self._scrub_vault_node(raw_item)
            return items
        if isinstance(value, list):
            sequence = cast("Any", value)
            return [self._scrub_vault_node(item) for item in sequence]
        if isinstance(value, tuple):
            pair = cast("Any", value)
            return [self._scrub_vault_node(item) for item in pair]
        if isinstance(value, str):
            return self._scrub_vault_text(value)
        return value

    def _scrub_vault_text(self, text: str) -> Any:
        """Alias registered paths; degrade unregistered absolutes to refs."""
        candidate: Path | None = None
        try:
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            candidate = candidate.resolve(strict=False)
        except OSError:
            candidate = None
        if candidate is not None:
            for alias, root in self._roots:
                try:
                    relative = candidate.relative_to(root)
                except ValueError:
                    continue
                if alias in self._config.safe_path_aliases:
                    return f"{alias}/{relative.as_posix()}"
                ref = self._path_refs.get(str(candidate))
                if ref is None:
                    self._path_ref_count += 1
                    ref = f"path-{self._path_ref_count}"
                    self._path_refs[str(candidate)] = ref
                return ref
            if candidate.is_absolute() and str(candidate) != text:
                # A bare filename that resolves under cwd is not a path
                # leak; keep the original text.
                pass
            elif len(text) > 1 and Path(text).is_absolute():
                ref = self._path_refs.get(text)
                if ref is None:
                    self._path_ref_count += 1
                    ref = f"path-{self._path_ref_count}"
                    self._path_refs[text] = ref
                return ref
        return text

    def record_annotation(
        self,
        namespace: str,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        anchor_sequence: int | None = None,
        *,
        stage_id: str | None = None,
        stage_visit_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Persist one trusted domain annotation (Phase 3).

        Known ``nemoir.autoresearch/v1`` / ``trial_finished`` payloads are
        strictly validated; unknown namespaces retain only namespace/kind
        with a redacted payload marker. Returns the persisted record, or
        None when the cleartext scanner forces omission (sequence gaps are
        valid). Raises :class:`TraceError` on malformed known payloads,
        bad anchors, or missing stage context.
        """
        self._require_begun()
        if self._event_limit_exceeded or len(self._events) >= LIMIT_EVENT_COUNT:
            self._event_limit_exceeded = True
            return None
        visit = stage_visit_id or self._current_visit
        stage = stage_id or (
            self._visit_to_stage.get(visit, self._current_stage_id) if visit is not None else None
        )
        if visit is None or stage is None:
            msg = "trace annotation requires an enclosing stage visit"
            raise TraceError(msg)
        if not re.fullmatch(r"s-[1-9][0-9]*", visit):
            msg = f"trace annotation has invalid stage_visit_id {visit!r}"
            raise TraceError(msg)
        # ``stage`` is derived from host-supplied ids: validate as untrusted.
        stage_value: Any = stage
        if not isinstance(stage_value, str) or not 1 <= len(stage_value) <= 256:
            msg = f"trace annotation has invalid stage_id {stage_value!r}"
            raise TraceError(msg)
        # ``anchor_sequence`` comes from host hooks: validate as untrusted.
        anchor_value: Any = anchor_sequence
        if anchor_value is not None and (
            isinstance(anchor_value, bool)
            or not isinstance(anchor_value, int)
            or anchor_value < 1
            or not MIN_SAFE_INT <= anchor_value <= MAX_SAFE_INT
        ):
            msg = f"trace annotation has invalid anchor_sequence {anchor_value!r}"
            raise TraceError(msg)
        if anchor_sequence is not None:
            # Anchor must belong to the declared visit when that visit has
            # observed sequences; otherwise the annotation would point at an
            # unrelated event. Hook path forces the triggering sequence, so
            # this primarily guards direct calls (which raise loudly).
            known = self._visit_sequences.get(visit, [])
            if known and anchor_sequence not in known:
                msg = (
                    f"trace annotation anchor_sequence {anchor_sequence!r} "
                    f"does not belong to visit {visit!r}"
                )
                raise TraceError(msg)
        if namespace == _ANNOTATION_NAMESPACE and kind == _ANNOTATION_KIND:
            if payload is None:
                msg = "trial_finished annotation requires a payload mapping"
                raise TraceError(msg)
            projected_payload = _validate_autoresearch_payload(payload)
            # Audit policy gate (redaction-policy §11): digests and raw
            # mechanism IDs require explicit publication review. The audit
            # profile keeps only opaque refs; reject cleartext identifiers.
            if self._config.profile == "audit":
                if projected_payload.get("candidate_digest") is not None:
                    msg = "audit profile rejects non-null candidate_digest (requires publication review)"
                    raise TraceError(msg)
                if projected_payload.get("parent_digest") is not None:
                    msg = (
                        "audit profile rejects non-null parent_digest (requires publication review)"
                    )
                    raise TraceError(msg)
                if projected_payload.get("mechanism_id") is not None:
                    msg = (
                        "audit profile rejects non-null mechanism_id (requires publication review)"
                    )
                    raise TraceError(msg)
        else:
            # Unknown namespace: retain only namespace/kind per policy §11.
            projected_payload = self._new_marker(
                "unapproved_field", payload if payload is not None else {}
            )
        record: dict[str, Any] = {
            "kind": "annotation",
            "run_id": self._trace_id,
            "timestamp": format_timestamp(self._now()),
            "stage_id": stage,
            "stage_visit_id": visit,
            "annotation": {
                "namespace": namespace,
                "kind": kind,
                "payload": projected_payload,
            },
            "redacted_fields": [],
        }
        if anchor_sequence is not None:
            record["anchor_sequence"] = anchor_sequence
        if "$redacted" in projected_payload:
            record["redacted_fields"] = ["/annotation/payload"]
        record = self._apply_registry(record)
        omitted = self._scan_and_mask(record)
        if omitted:
            return None
        record["redacted_fields"] = sorted(set(record.get("redacted_fields", [])))
        if len(self._events) >= LIMIT_EVENT_COUNT:
            self._event_limit_exceeded = True
            return None
        self._events.append(record)
        return record

    # -- live event observation ------------------------------------------

    def observe_workflow_event(self, event: WorkflowEvent) -> dict[str, Any] | None:
        """Project one live event into the redacted public ledger.

        Returns the projected record, or None when the cleartext scanner
        forces record omission (sequence gaps are valid and expected).
        """
        self._require_begun()
        if self._event_limit_exceeded or len(self._events) >= LIMIT_EVENT_COUNT:
            self._event_limit_exceeded = True
            return None
        record = self._project(event)
        if record is None:
            return None
        record = self._apply_registry(record)
        omitted = self._scan_and_mask(record)
        if omitted:
            return None
        record["redacted_fields"] = sorted(set(record.get("redacted_fields", [])))
        if len(self._events) >= LIMIT_EVENT_COUNT:
            self._event_limit_exceeded = True
            return None
        self._events.append(record)
        # Track per-visit sequences for annotation anchor validation.
        visit_seq = record.get("stage_visit_id")
        seq_num = getattr(event, "sequence", None)
        if (
            isinstance(visit_seq, str)
            and isinstance(seq_num, int)
            and not isinstance(seq_num, bool)
            and seq_num >= 1
        ):
            self._visit_sequences.setdefault(visit_seq, []).append(seq_num)
        if record.get("kind") == "stage_completed":
            self._capture_stage_snapshot(record, event)
            self._maybe_emit_stage_annotation(record, event)
        if record.get("kind") == "transition_selected":
            self._flush_transition_evidence(record, event)
        return record

    def _capture_stage_snapshot(self, record: dict[str, Any], event: WorkflowEvent) -> None:
        """Retain full stage outputs for the encrypted vault (Phase 4)."""
        if not self._vault_enabled:
            return
        if not self._config.vault_capture.include_stage_snapshots:
            return
        try:
            payload = {
                "stage_id": event.stage_id or record.get("stage_id"),
                "output": self._scrub_vault_value(_to_jsonable(dict(event.output or {}))),
            }
        except TraceError:
            payload = {
                "stage_id": event.stage_id or record.get("stage_id"),
                "output": self._new_marker("private_content", event.output or {}),
            }
        seq = getattr(event, "sequence", None)
        self._append_vault_record(
            "stage_snapshot",
            payload,
            event_sequence=seq if isinstance(seq, int) and seq >= 1 else None,
            stage_visit_id=record.get("stage_visit_id"),
        )

    def _flush_transition_evidence(self, record: dict[str, Any], event: WorkflowEvent) -> None:
        """Link pending guard evidence to its ledger sequence (Phase 4)."""
        if not self._vault_enabled:
            return
        visit = record.get("stage_visit_id")
        if not isinstance(visit, str):
            return
        pending = self._pending_transitions.pop(visit, [])
        seq = getattr(event, "sequence", None)
        seq_num = seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 1 else None
        for item in pending:
            self._append_vault_record(
                "transition_evaluation",
                {"candidates": item["candidates"]},
                event_sequence=seq_num,
                stage_visit_id=visit,
            )

    def _maybe_emit_stage_annotation(self, record: dict[str, Any], event: WorkflowEvent) -> None:
        """Invoke the host ``on_stage_completed`` hook, if configured.

        Hook failures never break the run: they are swallowed so workflow
        control flow continues with the annotation missing, but a bounded
        safe warning is recorded and surfaced via the summary. Direct
        :meth:`record_annotation` calls still raise loudly for tests.
        The hook-supplied anchor is ignored; the triggering
        ``stage_completed`` sequence is authoritative.
        """
        hook = self._config.on_stage_completed
        if hook is None or self._finished:
            return
        try:
            info: dict[str, Any] = {
                "stage_id": record.get("stage_id"),
                "stage_visit_id": record.get("stage_visit_id"),
                "sequence": getattr(event, "sequence", None),
            }
            spec = hook(info)
        except Exception:
            self._record_annotation_warning(record, "hook_failed")
            return
        if spec is None:
            return
        try:
            if not isinstance(spec, Mapping):  # type: ignore[reportUnnecessaryIsInstance]
                self._record_annotation_warning(record, "invalid_spec")
                return
            namespace = spec.get("namespace")
            kind = spec.get("kind")
            payload = spec.get("payload")
            if not isinstance(namespace, str) or not isinstance(kind, str):
                self._record_annotation_warning(record, "invalid_spec")
                return
            if not isinstance(payload, Mapping):
                self._record_annotation_warning(record, "invalid_payload")
                return
            payload = cast("Mapping[str, Any]", payload)
            # Force the anchor to the triggering sequence; never trust a
            # hook-returned anchor (prevents cross-visit mislinking).
            triggering = getattr(event, "sequence", None)
            anchor: Any = triggering if isinstance(triggering, int) and triggering >= 1 else None
            persisted = self.record_annotation(
                namespace,
                kind,
                payload,
                anchor,
                stage_id=record.get("stage_id"),
                stage_visit_id=record.get("stage_visit_id"),
            )
            if persisted is None:
                # Scanner-forced omission: same silent-loss class as a hook
                # failure, so count it (review item 1). Direct calls return
                # None to the caller; only the hook path auto-counts here.
                self._record_annotation_warning(record, "invalid_payload")
        except Exception:
            self._record_annotation_warning(record, "invalid_payload")
            return

    def _record_annotation_warning(self, record: dict[str, Any], reason: str) -> None:
        """Record a bounded safe completeness warning for a dropped hook annotation."""
        self._annotations_dropped += 1
        if len(self._annotation_warnings) >= 10:
            return
        stage = record.get("stage_id")
        stage_str = stage if isinstance(stage, str) else "unknown"
        # Safe fixed vocabulary only; never echo payload values.
        safe_reason = (
            reason
            if reason in ("hook_failed", "invalid_spec", "invalid_payload")
            else "invalid_payload"
        )
        self._annotation_warnings.append(f"{stage_str}:{safe_reason}")

    # -- projection ------------------------------------------------------

    def _now(self) -> datetime:
        return self._clock()

    def _require_begun(self) -> None:
        if not self._begun:
            msg = "TraceRecorder used before begin_run"
            raise TraceError(msg)
        if self._finished:
            msg = "TraceRecorder used after finish_run"
            raise TraceError(msg)

    def _new_marker(
        self, reason: str, value: Any, *, include_length: bool = True
    ) -> dict[str, Any]:
        self._redaction_count += 1
        inner: dict[str, Any] = {
            "token": f"r-{self._redaction_count}",
            "reason": reason,
            "value_type": _value_type(value),
        }
        if include_length:
            size: int | None = None
            if isinstance(value, str):
                try:
                    size = len(value.encode("utf-8"))
                except UnicodeEncodeError:
                    # Lone surrogates (e.g. \ud800) are not valid UTF-8;
                    # fall back to surrogatepass so projection never raises
                    # and workflow outcome is unaffected. Runtime validation
                    # will have already rejected such strings for correct
                    # I-JSON semantics, but the marker must be resilient.
                    size = len(value.encode("utf-8", errors="surrogatepass"))
            elif isinstance(value, (bytes, bytearray)):
                size = len(value)
            elif isinstance(value, (list, tuple)):
                size = len(cast("Sequence[Any]", value))
            if size is not None:
                inner["length"] = size
        return {"$redacted": inner}

    def _take_visit(self, stage_id: str) -> str:
        queue = self._pending_visits.get(stage_id)
        if queue:
            visit = queue.pop(0)
        else:
            self._visit_count += 1
            visit = f"s-{self._visit_count}"
        self._current_visit = visit
        self._current_stage_id = stage_id
        self._visit_to_stage[visit] = stage_id
        return visit

    def _peek_model(self, visit: str) -> str | None:
        queue = self._pending_models.get(visit)
        return queue[-1] if queue else None

    def _pop_model(self, visit: str) -> str:
        queue = self._pending_models.get(visit)
        if queue:
            call_id = queue.pop(0)
        else:
            self._model_count += 1
            call_id = f"m-{self._model_count}"
        self._last_model_id[visit] = call_id
        return call_id

    def _pop_tool(self, visit: str) -> str:
        """Consume the next begun-but-unstarted tool id for a started event."""
        queue = self._pending_tools.get(visit)
        if queue:
            call_id = queue.pop(0)
        else:
            self._tool_count += 1
            call_id = f"t-{self._tool_count}"
            self._tool_started_at.setdefault(call_id, self._now())
        self._open_tools.setdefault(visit, []).append(call_id)
        return call_id

    def _peek_tool(self, visit: str) -> str:
        """Resolve the tool id for a completed/failed event: the most
        recently started call in the visit (nesting is strictly LIFO)."""
        stack = self._open_tools.get(visit)
        if stack:
            return stack[-1]
        queue = self._pending_tools.get(visit)
        if queue:
            return queue.pop(0)
        self._tool_count += 1
        call_id = f"t-{self._tool_count}"
        self._tool_started_at.setdefault(call_id, self._now())
        return call_id

    def _base(self, event: WorkflowEvent, *, stage_visit_id: str | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": event.kind,
            "run_id": self._trace_id,
            "timestamp": format_timestamp(self._now()),
            "redacted_fields": [],
        }
        sequence = getattr(event, "sequence", None)
        if isinstance(sequence, int) and sequence >= 1:
            record["sequence"] = sequence
        if event.stage_id is not None:
            record["stage_id"] = event.stage_id
        if stage_visit_id is not None:
            record["stage_visit_id"] = stage_visit_id
        return record

    def _project(self, event: WorkflowEvent) -> dict[str, Any] | None:
        kind = event.kind
        if kind == "run_started":
            return self._project_run_started(event)
        if kind == "stage_started":
            return self._project_stage_started(event)
        if kind == "model_delta":
            return self._project_model_delta(event)
        if kind == "model_completed":
            return self._project_model_completed(event)
        if kind == "model_retry":
            return self._project_model_retry(event)
        if kind == "tool_call_started":
            return self._project_tool_started(event)
        if kind == "tool_call_completed":
            return self._project_tool_completed(event)
        if kind == "tool_call_failed":
            return self._project_tool_failed(event)
        if kind == "policy_checked":
            return self._project_policy_checked(event)
        if kind == "policy_denied":
            return self._project_policy_denied(event)
        if kind == "transition_selected":
            return self._project_transition_selected(event)
        if kind == "stage_completed":
            return self._project_stage_completed(event)
        if kind == "run_completed":
            return self._project_run_completed(event)
        if kind == "run_failed":
            return self._project_run_failed(event)
        # Unknown future kinds: omit from the audit ledger rather than guess.
        return None

    def _project_run_started(self, event: WorkflowEvent) -> dict[str, Any]:
        record = self._base(event)
        manifest = self._manifest
        record["metadata"] = {
            "workflow_id": getattr(manifest, "workflow_id", "unknown"),
            "entry": getattr(manifest, "entry_stage_id", "unknown"),
        }
        return record

    def _project_stage_started(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._take_visit(event.stage_id or "")
        return self._base(event, stage_visit_id=visit)

    def _project_model_delta(self, event: WorkflowEvent) -> dict[str, Any]:
        # Deltas belong to the enclosing visit; never consume the queued
        # visit id (later tool/policy/completion events in the same visit
        # still need it). Synthesize only when no visit is known.
        visit = self._current_visit
        if visit is None:
            self._visit_count += 1
            visit = f"s-{self._visit_count}"
            self._current_visit = visit
        record = self._base(event, stage_visit_id=visit)
        call_id = self._peek_model(visit)
        if call_id is None:
            self._model_count += 1
            call_id = f"m-{self._model_count}"
            self._pending_models.setdefault(visit, []).append(call_id)
        record["model_call_id"] = call_id
        if event.channel is not None:
            record["channel"] = event.channel
        text = event.text if isinstance(event.text, str) else ""
        if event.channel == "reasoning":
            record["text"] = self._new_marker("reasoning", text, include_length=False)
        else:
            record["text"] = self._new_marker("private_content", text)
        record["redacted_fields"] = ["/text"]
        return record

    def _project_model_completed(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_visit or ""
        record = self._base(event, stage_visit_id=visit)
        call_id = self._pop_model(visit)
        record["model_call_id"] = call_id
        metadata: dict[str, Any] = {"content_omitted": True}
        if call_id in self._model_bytes:
            metadata["response_bytes"] = self._model_bytes[call_id]
        record["metadata"] = metadata
        return record

    def _project_model_retry(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_visit or ""
        record = self._base(event, stage_visit_id=visit)
        # The schema requires model_call_id on every model_retry. A retry can
        # legally arrive with no pending call (e.g. models.py emits a
        # tool-error retry after model_completed already consumed the call
        # id). Reuse that consumed id: the retry announces the failure of
        # the attempt it follows, which already has vault request/response
        # evidence, so semantic verification stays complete. Synthesize a
        # fresh id only when the visit has no prior model call at all.
        # Peek-hit behavior is unchanged.
        call_id = self._peek_model(visit)
        if call_id is None:
            call_id = self._last_model_id.get(visit)
        if call_id is None:
            self._model_count += 1
            call_id = f"m-{self._model_count}"
        record["model_call_id"] = call_id
        metadata: Mapping[str, Any] = event.metadata or {}
        record["metadata"] = {
            "attempt": _safe_int(metadata.get("attempt"), 1, minimum=1),
            "max_retries": _safe_int(metadata.get("max_retries"), 0, minimum=0),
            "category": _safe_category(metadata.get("category")),
        }
        record["error"] = "model_retry"
        return record

    def _project_tool_started(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_or_new_visit()
        record = self._base(event, stage_visit_id=visit)
        call_id = self._pop_tool(visit)
        # Stash pre-redaction args + tool name for the vault (Phase 4).
        # Projection below never mutates ``event.args``.
        if self._vault_enabled:
            self._open_tool_calls[call_id] = {
                "args": event.args,
                "capability": event.capability,
                "tool_name": event.tool_name,
                "stage_visit_id": visit,
            }
        record["tool_call_id"] = call_id
        capability = event.capability or "unknown"
        record["capability"] = capability
        tool_name = event.tool_name
        if (
            isinstance(tool_name, str)
            and len(tool_name) <= _MAX_TOOL_NAME_LEN
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:\-]*", tool_name)
        ):
            record["tool_name"] = tool_name
        else:
            record.setdefault("redacted_fields", []).append("/tool_name")
        args: Mapping[str, Any] = event.args or {}
        projected, pointers = self._project_tool_args(capability, args)
        if projected:
            record["args"] = projected
        record.setdefault("redacted_fields", []).extend(pointers)
        return record

    def _current_or_new_visit(self) -> str:
        """The enclosing visit for non-started events: reuse the current
        one, synthesizing only when no visit is known (hook-free use)."""
        if self._current_visit is not None:
            return self._current_visit
        self._visit_count += 1
        self._current_visit = f"s-{self._visit_count}"
        return self._current_visit

    def _project_tool_completed(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_or_new_visit()
        record = self._base(event, stage_visit_id=visit)
        call_id = self._peek_tool(visit)
        record["tool_call_id"] = call_id
        record["capability"] = event.capability or "unknown"
        tool_name = event.tool_name
        if (
            isinstance(tool_name, str)
            and len(tool_name) <= _MAX_TOOL_NAME_LEN
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:\-]*", tool_name)
        ):
            record["tool_name"] = tool_name
        started = self._tool_started_at.get(call_id)
        metadata: dict[str, Any] = {"result_status": "ok"}
        if call_id in self._tool_result_types:
            metadata["result_type"] = self._tool_result_types[call_id]
        if started is not None:
            duration = int((self._now() - started).total_seconds() * 1000)
            metadata["duration_ms"] = max(0, duration)
        record["metadata"] = metadata
        return record

    def _project_tool_failed(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_or_new_visit()
        record = self._base(event, stage_visit_id=visit)
        call_id = self._peek_tool(visit)
        record["tool_call_id"] = call_id
        record["capability"] = event.capability or "unknown"
        tool_name = event.tool_name
        if (
            isinstance(tool_name, str)
            and len(tool_name) <= _MAX_TOOL_NAME_LEN
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:\-]*", tool_name)
        ):
            record["tool_name"] = tool_name
        _, error_type = self._tool_errors.get(call_id, ("tool_failed", "ToolExecutionError"))
        # Kind-fixed code per redaction-policy §9; taxonomy in error_type.
        record["error"] = "tool_failed"
        record["metadata"] = {"error_type": error_type}
        return record

    def _policy_ref(self, policy_id: Any) -> str | None:
        if isinstance(policy_id, str):
            return self._policy_refs.get(policy_id)
        return None

    def _project_policy_checked(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_or_new_visit()
        record = self._base(event, stage_visit_id=visit)
        record["capability"] = event.capability or "unknown"
        metadata: Mapping[str, Any] = event.metadata or {}
        kind = metadata.get("policy_kind")
        kind = kind if kind in ("before", "deny") else "deny"
        meta: dict[str, Any] = {"policy_kind": kind}
        ref = self._policy_ref(metadata.get("policy_id"))
        if ref is None:
            record.setdefault("redacted_fields", []).append("/metadata/policy_id")
        else:
            meta["policy_ref"] = ref
        if kind == "deny":
            meta["denied"] = bool(metadata.get("denied", False))
        else:
            required: Any = metadata.get("required_capabilities")
            req_items: list[Any] = (
                list(cast("Sequence[Any]", required)) if isinstance(required, (list, tuple)) else []
            )
            if req_items and all(isinstance(c, str) for c in req_items):
                meta["required_capabilities"] = sorted(set(req_items))
        record["metadata"] = meta
        return record

    def _project_policy_denied(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_or_new_visit()
        record = self._base(event, stage_visit_id=visit)
        record["capability"] = event.capability or "unknown"
        metadata: Mapping[str, Any] = event.metadata or {}
        ref = self._policy_ref(metadata.get("policy_id"))
        record["metadata"] = {}
        if ref is None:
            record.setdefault("redacted_fields", []).append("/metadata/policy_id")
        else:
            record["metadata"]["policy_ref"] = ref
        record["error"] = "policy_denied"
        return record

    _TRANSITION_REASONS = frozenset(
        {
            "explicit_transition",
            "backward_ref_loop",
            "next_stage_required_input_available",
            "skip_next_stage_required_input_missing",
            "fallthrough",
            "other",
        }
    )

    def _project_transition_selected(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_or_new_visit()
        record = self._base(event, stage_visit_id=visit)
        record["transition_to"] = event.transition_to or "unknown"
        metadata: Mapping[str, Any] = event.metadata or {}
        reason = metadata.get("reason")
        reason = reason if reason in self._TRANSITION_REASONS else "other"
        record["metadata"] = {
            "reason": reason,
            "priority": _safe_int(metadata.get("priority"), 0, minimum=0),
        }
        return record

    def _project_stage_completed(self, event: WorkflowEvent) -> dict[str, Any]:
        visit = self._current_or_new_visit()
        record = self._base(event, stage_visit_id=visit)
        self._stage_completed_count += 1
        output: Mapping[str, Any] = event.output or {}
        projected, pointers = self._project_stage_output(event.stage_id or "", output)
        record["output"] = projected
        record["redacted_fields"] = pointers
        return record

    def _project_run_completed(self, event: WorkflowEvent) -> dict[str, Any]:
        record = self._base(event)
        output = getattr(event.result, "output", None)
        record["result"] = self._new_marker(
            "private_content", output if output is not None else event.result
        )
        record["metadata"] = {"step_count": self._stage_completed_count}
        record["redacted_fields"] = ["/result"]
        return record

    def _project_run_failed(self, event: WorkflowEvent) -> dict[str, Any]:
        record = self._base(event)
        _, type_name = self._tool_errors.pop("__run__", ("run_failed", "UnknownError"))
        record["error"] = "run_failed"
        record["metadata"] = {"error_type": type_name}
        return record

    def record_run_error(self, exc: BaseException) -> None:
        """Capture the terminal failure's stable taxonomy (no message)."""
        self._require_begun()
        code, type_name = stable_error(exc)
        self._tool_errors["__run__"] = (code, type_name)

    # -- capability + output projection ----------------------------------

    def _normalize_path(self, raw: Any) -> tuple[str | None, str | None, str]:
        """Render ``raw`` as (path, path_ref, root_class) triple.

        Returns at most one of ``path``/``path_ref`` set; ``root_class`` is
        the alias when a registered root contains the value. Unregistered or
        unsafe values degrade to an opaque ``path-N`` ref.
        """
        text = str(raw)
        candidate: Path | None = None
        try:
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            candidate = candidate.resolve(strict=False)
        except OSError:
            candidate = None
        for alias, root in self._roots:
            if candidate is None:
                break
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            if alias in self._config.safe_path_aliases:
                return f"{alias}/{relative.as_posix()}", None, alias
            ref = self._path_refs.get(str(candidate))
            if ref is None:
                self._path_ref_count += 1
                ref = f"path-{self._path_ref_count}"
                self._path_refs[str(candidate)] = ref
            return None, ref, alias
        # No registered root matched: opaque marker, no alias information.
        key = text
        ref = self._path_refs.get(key)
        if ref is None:
            self._path_ref_count += 1
            ref = f"path-{self._path_ref_count}"
            self._path_refs[key] = ref
        return None, ref, ""

    def _project_tool_args(
        self, capability: str, args: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Project tool args per the capability allowlist (audit-v1 §6)."""
        redacted: list[str] = []
        if capability == "fs.read":
            return self._project_path_args(args, allow_keys=("path",), redacted=redacted)
        if capability == "fs.write":
            projected, redacted = self._project_path_args(
                args, allow_keys=("path",), redacted=redacted
            )
            if "content" in args:
                projected["content"] = self._new_marker("private_content", args["content"])
                redacted.append("/args/content")
            redacted.extend(f"/args/{key}" for key in args if key not in ("path", "content"))
            return projected, redacted
        if capability == "os.shell":
            for key in args:
                redacted.append(f"/args/{key}")
            return {}, redacted
        if capability == "http.fetch":
            projected: dict[str, Any] = {}
            method = args.get("method")
            if (
                isinstance(method, str)
                and len(method) <= _MAX_METHOD_LEN
                and re.fullmatch(r"[A-Z]+", method)
            ):
                projected["method"] = method
            elif "method" in args:
                redacted.append("/args/method")
            for key in ("url", "headers", "body"):
                if key in args:
                    if key in ("headers", "body"):
                        projected[key] = self._new_marker("private_content", args[key])
                    redacted.append(f"/args/{key}")
            redacted.extend(
                f"/args/{key}" for key in args if key not in ("method", "url", "headers", "body")
            )
            return projected, redacted
        if capability in ("user.elicit", "user.confirm"):
            projected = {key: self._new_marker("private_content", args[key]) for key in args}
            redacted.extend(f"/args/{key}" for key in args)
            return projected, redacted
        if capability in (
            "browser.storage.read",
            "browser.storage.write",
            "browser.js.run",
            "browser.js.sandbox",
        ):
            projected = {key: self._new_marker("private_content", args[key]) for key in args}
            redacted.extend(f"/args/{key}" for key in args)
            return projected, redacted
        # Unknown capability: name/status/timing only; all args private.
        if args:
            redacted.append("/args")
        return {}, redacted

    def _project_path_args(
        self,
        args: Mapping[str, Any],
        *,
        allow_keys: tuple[str, ...],
        redacted: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        projected: dict[str, Any] = {}
        for key in allow_keys:
            if key not in args:
                continue
            path, path_ref, root = self._normalize_path(args[key])
            if path is not None:
                projected["path"] = path
                if root:
                    projected["root_class"] = root
            elif path_ref is not None:
                projected["path_ref"] = path_ref
                if root:
                    projected["root_class"] = root
            else:
                redacted.append(f"/args/{key}")
        redacted.extend(f"/args/{key}" for key in args if key not in allow_keys)
        return projected, redacted

    def _project_stage_output(
        self, stage_id: str, output: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Project stage outputs: approved scalar metrics survive, all else is
        replaced by markers. Strings are never metrics."""
        projected: dict[str, Any] = {}
        redacted: list[str] = []
        writes = self._stage_writes(stage_id)
        for name in output:
            if writes is not None and name not in writes:
                redacted.append(f"/output/{name}")
                projected[name] = self._new_marker("unapproved_field", output[name])
                continue
            value = output[name]
            if f"{stage_id}.{name}" in self._config.approved_metrics and _is_metric_value(value):
                projected[name] = value
                continue
            projected[name] = self._new_marker(_output_reason(value), value)
            redacted.append(f"/output/{name}")
        return projected, redacted

    def _stage_writes(self, stage_id: str) -> set[str] | None:
        manifest = self._manifest
        stages = getattr(manifest, "stages", None)
        if stages is None:
            return None
        for stage in stages:
            if getattr(stage, "id", None) == stage_id:
                return {w.name for w in (getattr(stage, "writes", ()) or ())}
        return None

    # -- secret registry + scanner application ---------------------------

    def _apply_registry(self, record: dict[str, Any]) -> dict[str, Any]:
        """Defense-in-depth: mask fields containing registered secret values."""
        for _ in range(_MAX_MASK_PASSES):
            hits = [
                f
                for f in _scan_strings(record, "", self._registry)
                if f.rule == "registered_secret"
            ]
            if not hits:
                return record
            pointer = hits[0].pointer
            if _is_maskable(pointer):
                self._set_marker(record, pointer, "credential")
            else:
                # Structural field echoed a credential: omit the record.
                record["_omit"] = True
                return record
        record["_omit"] = True
        return record

    def _set_marker(self, record: dict[str, Any], pointer: str, reason: str) -> None:
        parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]
        current: Any = record
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = cast("Any", current[part])
            elif isinstance(current, list) and part.isdigit():
                sequence = cast("list[Any]", current)
                if int(part) >= len(sequence):
                    return
                current = sequence[int(part)]
            else:
                return
        last = parts[-1]
        if isinstance(current, dict) and last in current:
            current[last] = self._new_marker(reason, current[last])
            fields = record.setdefault("redacted_fields", [])
            if pointer not in fields:
                fields.append(pointer)
        elif isinstance(current, list) and last.isdigit():
            sequence = cast("list[Any]", current)
            if int(last) < len(sequence):
                sequence[int(last)] = self._new_marker(reason, sequence[int(last)])
                fields = record.setdefault("redacted_fields", [])
                if pointer not in fields:
                    fields.append(pointer)

    def _scan_and_mask(self, record: dict[str, Any]) -> bool:
        """Apply the secrets-v1 scanner. Returns True when the record must be
        omitted (finding in a non-maskable location)."""
        if record.pop("_omit", None):
            return True
        for _ in range(_MAX_MASK_PASSES):
            findings = [
                f
                for f in _scan_strings(record, "", self._registry)
                if f.rule != "registered_secret"
            ]
            if not findings:
                return False
            finding = findings[0]
            if not _is_maskable(finding.pointer):
                return True
            if finding.rule == "home_path":
                reason = "absolute_path"
            elif finding.rule in _CREDENTIAL_RULES:
                reason = "credential"
            else:
                reason = "unapproved_field"
            self._set_marker(record, finding.pointer, reason)
        return True

    # -- archive assembly ------------------------------------------------

    def _build_entries(self, status: str) -> dict[str, bytes]:
        manifest = self._manifest
        workflow_id = getattr(manifest, "workflow_id", "unknown")
        prov = self._config.provenance
        complete = prov.ir_sha256 is not None
        model = self._config.model
        provenance: dict[str, Any] = {
            "complete": complete,
            "frontend": prov.frontend,
            "target": (
                prov.target if prov.target in ("python", "web", "manual", "imported") else "manual"
            ),
            "compiler_version": prov.compiler_version,
            "runtime": {"name": RUNTIME_NAME, "version": runtime_version()},
        }
        if model is not None:
            model_obj: dict[str, Any] = {"name": model.name}
            if model.api_mode:
                model_obj["api_mode"] = model.api_mode
            sampling: dict[str, Any] = {}
            if model.temperature is not None:
                sampling["temperature"] = model.temperature
            if model.max_tokens is not None:
                sampling["max_tokens"] = model.max_tokens
            if sampling:
                model_obj["sampling"] = sampling
            provenance["model"] = model_obj
        exit_ids: Any = getattr(manifest, "exit_stage_ids", None)
        exits: list[str] = sorted(exit_ids) if exit_ids else []
        vault_enabled = self._vault_enabled
        if vault_enabled:
            capture: dict[str, Any] = {
                "profile": "replay",
                "vault_present": True,
                "publication_eligible": False,
                "redaction_policy": REDACTION_POLICY,
                "scanner": {"status": "passed", "ruleset": SCANNER_RULESET},
            }
        else:
            capture = {
                "profile": "audit",
                "vault_present": False,
                "publication_eligible": False,
                "redaction_policy": REDACTION_POLICY,
                "scanner": {"status": "passed", "ruleset": SCANNER_RULESET},
            }
        manifest_obj: dict[str, Any] = {
            "format": TRACE_FORMAT,
            "trace_id": self._trace_id,
            "created_at": format_timestamp(self._begin_time or self._now()),
            "status": status,
            "capture": capture,
            "workflow": {
                "id": workflow_id,
                "ir_version": getattr(prov, "ir_version", "0.1"),
                "ir_sha256": prov.ir_sha256,
                "entry": getattr(manifest, "entry_stage_id", "unknown"),
                "exits": exits,
            },
            "provenance": provenance,
            "integrity": {"algorithm": "sha256", "entry": INTEGRITY_PATH},
            "viewer": {"min_format": TRACE_FORMAT},
        }
        graph_obj = self._build_graph()
        events_bytes = b"".join(to_canonical_bytes(e) + b"\n" for e in self._events)
        events_sha = sha256_tag(events_bytes)
        kinds: dict[str, int] = {}
        visits: set[str] = set()
        for event in self._events:
            kinds[event.get("kind", "?")] = kinds.get(event.get("kind", "?"), 0) + 1
            if "stage_visit_id" in event:
                visits.add(event["stage_visit_id"])
        finish_time = self._now()
        begin_time = self._begin_time or finish_time
        summary_obj: dict[str, Any] = {
            "format": SUMMARY_FORMAT,
            "events_sha256": events_sha,
            "event_count": len(self._events),
            "stage_visit_count": len(visits),
            "duration_ms": max(0, int((finish_time - begin_time).total_seconds() * 1000)),
            "counts_by_kind": kinds,
            "annotations_dropped": self._annotations_dropped,
            "annotation_warnings": list(self._annotation_warnings),
        }
        payloads: dict[str, bytes] = {
            MANIFEST_PATH: to_canonical_bytes(manifest_obj),
            GRAPH_PATH: to_canonical_bytes(graph_obj),
            EVENTS_PATH: events_bytes,
            SUMMARY_PATH: to_canonical_bytes(summary_obj),
        }
        if vault_enabled:
            self._seal_vault_entries(payloads, manifest_obj)
        integrity_entries = [
            {
                "path": path,
                "media_type": (
                    "application/octet-stream"
                    if path == VAULT_ENC_PATH
                    else "application/x-ndjson"
                    if path.endswith(".ndjson")
                    else "application/json"
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
                        "path": e["path"],
                        "sha256": e["sha256"],
                        "uncompressed_bytes": e["uncompressed_bytes"],
                    }
                    for e in integrity_entries
                ),
                key=lambda e: e["path"],
            ),
        }
        integrity_obj: dict[str, Any] = {
            "format": "nemoir.trace.integrity/0.1",
            "algorithm": "sha256",
            "entries": integrity_entries,
            "content_identity": sha256_tag(to_canonical_bytes(identity_obj)),
        }
        payloads[INTEGRITY_PATH] = to_canonical_bytes(integrity_obj)
        return payloads

    def _manifest_snapshot_for_vault(self) -> dict[str, Any]:
        """Capture a replay-grade manifest snapshot for the vault."""
        try:
            manifest_dict = _to_jsonable(dataclasses.asdict(self._manifest))
        except Exception:
            # dataclasses.asdict fails on exotic field values; fall back to
            # the reflective converter.
            manifest_dict = _to_jsonable(self._manifest)
        if not isinstance(manifest_dict, dict):
            msg = "workflow manifest snapshot must be an object"
            raise TraceError(msg)
        prov = self._config.provenance
        return {
            "manifest": manifest_dict,
            "ir_sha256": prov.ir_sha256,
            "workflow_id": getattr(self._manifest, "workflow_id", "unknown"),
        }

    def _seal_vault_entries(self, payloads: dict[str, bytes], manifest_obj: dict[str, Any]) -> None:
        """Build, scan, encrypt, and attach vault entries (replay only)."""
        capture_cfg = self._config.vault_capture
        if self._vault_passphrase is None:
            msg = "trace profile 'replay' requires vault_passphrase"
            raise TraceError(msg)
        if capture_cfg.include_ir:
            try:
                snapshot = self._manifest_snapshot_for_vault()
            except TraceError as exc:
                msg = f"cannot snapshot workflow manifest for vault: {exc}"
                raise TraceError(msg) from exc
            self._append_vault_record("full_workflow_ir", snapshot)
        lines: list[bytes] = []
        for record in self._vault_records:
            shape_error = _vault_record_shape_error(record)
            if shape_error is not None:
                msg = shape_error
                raise TraceError(msg)
            lines.append(to_canonical_bytes(record) + b"\n")
        plaintext = b"".join(lines)
        if len(plaintext) > capture_cfg.max_vault_bytes:
            msg = (
                "vault plaintext exceeds max_vault_bytes "
                f"({len(plaintext)} > {capture_cfg.max_vault_bytes})"
            )
            raise TraceError(msg)
        # Belt-and-braces: scrubbed vault plaintext must carry no
        # credential material. Markers are exempt (values already removed).
        problems: list[str] = []
        for number, record in enumerate(self._vault_records, 1):
            for finding in _scan_strings(record, "", self._registry):
                if self._vault_finding_excused(record, finding):
                    continue
                problems.append(f"vault:{number}:{finding.pointer} [{finding.rule}]")
        if problems:
            self._remove_partial_marker()
            detail = "; ".join(problems[:10])
            msg = (
                "trace scanner blocked vault finalization with "
                f"{len(problems)} finding(s): {detail}"
            )
            raise TraceError(msg)
        manifest_bytes = payloads[MANIFEST_PATH]
        events_bytes = payloads[EVENTS_PATH]
        graph_bytes = payloads[GRAPH_PATH]
        aad_dict: dict[str, Any] = {
            "events_sha256": sha256_tag(events_bytes),
            "format": VAULT_AAD_FORMAT,
            "ir_sha256": manifest_obj["workflow"]["ir_sha256"] or VAULT_NULL_IR_SHA256,
            "manifest_sha256": sha256_tag(manifest_bytes),
            "trace_id": self._trace_id,
            "workflow_graph_sha256": sha256_tag(graph_bytes),
        }
        aad = to_canonical_bytes(aad_dict)
        salt, nonce, sealed = encrypt_vault_records(plaintext, self._vault_passphrase, aad)
        meta_obj = _vault_meta_object(salt=salt, nonce=nonce, aad_dict=aad_dict)
        payloads[VAULT_ENC_PATH] = sealed
        payloads[VAULT_META_PATH] = to_canonical_bytes(meta_obj)

    def _build_graph(self) -> dict[str, Any]:
        manifest = self._manifest
        nodes: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        for stage in getattr(manifest, "stages", ()) or ():
            execution = getattr(stage, "execution", None)
            kind = getattr(execution, "kind", "model")
            kind = kind if kind in ("model", "tool") else "model"
            nodes.append(
                {
                    "id": getattr(stage, "id", "unknown"),
                    "execution": kind,
                    "capabilities": sorted(getattr(stage, "requires", frozenset()) or ()),
                    "writes": [
                        {"name": w.name, "type": w.type, "optional": bool(w.optional)}
                        for w in (getattr(stage, "writes", ()) or ())
                    ],
                }
            )
            for trans in getattr(stage, "transitions", ()) or ():
                guard = getattr(trans, "guard", None)
                guard_kind = getattr(guard, "kind", "always")
                if guard_kind not in ("always", "has_value", "missing", "eq", "if"):
                    guard_kind = "always"
                transitions.append(
                    {
                        "from": getattr(stage, "id", "unknown"),
                        "to": getattr(trans, "to", "unknown"),
                        "priority": max(0, int(getattr(trans, "priority", 0) or 0)),
                        "guard_kind": guard_kind,
                    }
                )
        policies: list[dict[str, Any]] = []
        for index, policy in enumerate(getattr(manifest, "policies", ()) or (), 1):
            kind = getattr(policy, "kind", "deny")
            kind = kind if kind in ("before", "deny") else "deny"
            trigger = getattr(policy, "trigger", None)
            entry: dict[str, Any] = {
                "ref": f"p-{index}",
                "kind": kind,
                "trigger_capability": getattr(trigger, "capability", "unknown"),
            }
            if kind == "before":
                entry["required_capabilities"] = sorted(
                    {r.capability for r in (getattr(policy, "requires", ()) or ())}
                )
            policies.append(entry)
        return {
            "format": GRAPH_FORMAT,
            "workflow_id": getattr(manifest, "workflow_id", "unknown"),
            "entry": getattr(manifest, "entry_stage_id", "unknown"),
            "exits": sorted(getattr(manifest, "exit_stage_ids", frozenset()) or ()),
            "nodes": nodes,
            "transitions": transitions,
            "policies": policies,
        }

    def _final_scan(self, entries: dict[str, bytes]) -> None:
        """Block finalization when any cleartext entry leaks (locations only)."""
        problems = scan_cleartext_entries(entries, self._registry)
        if problems:
            self._remove_partial_marker()
            detail = "; ".join(problems[:10])
            msg = f"trace scanner blocked finalization with {len(problems)} finding(s): {detail}"
            raise TraceError(msg)

    # -- archive IO ------------------------------------------------------

    @staticmethod
    def _zip_info(name: str) -> zipfile.ZipInfo:
        return _zip_info(name)

    @classmethod
    def _write_archive(cls, path: Path, entries: dict[str, bytes]) -> None:
        _write_zip_archive(path, entries)

    def _partial_path(self) -> Path:
        return self._config.path.with_name(self._config.path.name + ".partial")

    def _write_partial_marker(self, status: str) -> None:
        try:
            marker = self._partial_path()
            if str(marker.parent) not in ("", "."):
                marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                f'{{"trace_id":{self._trace_id!r},"status":{status!r}}}\n', encoding="utf-8"
            )
        except OSError:
            pass

    def _remove_partial_marker(self) -> None:
        with contextlib.suppress(OSError):
            self._partial_path().unlink(missing_ok=True)


class NoOpTraceRecorder:
    """Zero-cost recorder used when tracing is disabled. Every hook is a
    no-op; ``observe_workflow_event`` returns None."""

    vault_enabled = False
    # Taped policy outcomes for replay (same contract as
    # TraceRecorder.consume_taped_policy). Replay sets this on its own
    # instance; the class default keeps the shared NO_OP singleton tape-free.
    policy_tape: dict[str, list[str]] | None = None

    def consume_taped_policy(self, policy_id: Any) -> str | None:
        """Pop the next recorded outcome for a deny-policy check, if taped."""
        tape = self.policy_tape
        if tape is None or not isinstance(policy_id, str):
            return None
        queue = tape.get(policy_id)
        if not queue:
            return None
        return queue.pop(0)

    def begin_run(self, _manifest: Any) -> None:
        return None

    def finish_run(self, _status: str) -> None:
        return None

    def begin_stage_visit(self, _stage_id: str) -> str:
        return ""

    def begin_model_call(self, _stage_id: str = "", _stage_visit_id: str | None = None) -> str:
        return ""

    def record_model_response(self, _model_call_id: str, **_kwargs: Any) -> None:
        return None

    def begin_tool_call(self, _stage_id: str = "", _stage_visit_id: str | None = None) -> str:
        return ""

    def record_tool_result(self, _tool_call_id: str, _result: Any, **_kwargs: Any) -> None:
        return None

    def record_tool_error(self, _tool_call_id: str, _exc: BaseException) -> None:
        return None

    def record_model_request(self, _model_call_id: str, _request: Any) -> None:
        return None

    def record_run_inputs(self, _inputs: Any) -> None:
        return None

    def record_policy_evaluation(
        self, _stage_visit_id: Any, _policy_id: Any, _bound: Any, _outcome: str
    ) -> None:
        return None

    def record_run_error(self, _exc: BaseException) -> None:
        return None

    def record_transition_evaluation(
        self, _stage_visit_id: str, _candidates: list[dict[str, Any]]
    ) -> None:
        return None

    def record_annotation(
        self,
        _namespace: str,
        _kind: str,
        _payload: Mapping[str, Any] | None = None,
        _anchor_sequence: int | None = None,
        *,
        _stage_id: str | None = None,
        _stage_visit_id: str | None = None,
    ) -> None:
        return None

    def observe_workflow_event(self, _event: WorkflowEvent) -> None:
        return None


NO_OP = NoOpTraceRecorder()


def resolve_recorder(
    value: TraceRecorder | NoOpTraceRecorder | None,
) -> TraceRecorder | NoOpTraceRecorder:
    """Normalize an optional recorder to a non-None recorder interface."""
    return value if value is not None else NO_OP


TraceValue = TraceRecorder | TraceConfig | Callable[[], "TraceRecorder | None"] | None
"""Host trace values accepted by generated agents: an existing recorder,
an immutable config, a per-run factory, or None (tracing disabled)."""


def resolve_trace_recorder(
    value: TraceValue,
    *,
    default_provenance: HostProvenance | None = None,
    default_model: ModelDescriptor | None = None,
) -> TraceRecorder | None:
    """Resolve a host trace value to a fresh per-run recorder (or None).

    ``TraceConfig`` values adopt ``default_provenance``/``default_model``
    for fields the host left unspecified, so generated agents transparently
    bind their own compiled IR without the host restating it. An existing
    :class:`TraceRecorder` is used as-is (one recorder per run; reuse
    across runs raises at ``begin_run``). Factories are called once per
    resolution. Anything else raises :class:`TraceError`.
    """
    if value is None:
        return None
    if isinstance(value, TraceRecorder):
        return value
    if isinstance(value, TraceConfig):
        provenance = value.provenance
        if (
            provenance.ir_sha256 is None
            and default_provenance is not None
            and default_provenance.ir_sha256 is not None
        ):
            provenance = default_provenance
        model = value.model if value.model is not None else default_model
        if provenance is value.provenance and model is value.model:
            return TraceRecorder(value)
        return TraceRecorder(replace(value, provenance=provenance, model=model))
    if callable(value):
        # Untyped factories may return anything at runtime; validate.
        recorder: Any = value()
        if recorder is not None and not isinstance(recorder, TraceRecorder):
            msg = f"trace factory must return TraceRecorder or None, got {type(recorder).__name__}"
            raise TraceError(msg)
        return recorder
    msg = f"unsupported trace value: {type(value).__name__}"
    raise TraceError(msg)


# ---------------------------------------------------------------------------
# Archive reading + verification (stdlib-only structural checks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    content_identity: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    # Phase 4 verification levels (plan.md §7.6). ``verify_archive`` fills
    # integrity/structural/replayability without a passphrase; ``semantic``
    # is evaluated by :func:`unlock_archive` / taped replay, which hold the
    # vault evidence. Values: integrity/structural ``passed`` (+ structural
    # ``passed-with-warnings``) or ``failed``; semantic ``not-evaluated``,
    # ``passed``, or ``failed``; replayability ``taped-replay`` (vault
    # present), ``playback-only``, or ``none`` (verification failed).
    integrity: str = "failed"
    structural: str = "failed"
    semantic: str = "not-evaluated"
    replayability: str = "none"


def scan_cleartext_entries(
    entries: Mapping[str, bytes], registry: _SecretRegistry | None = None
) -> list[str]:
    """Scan cleartext entries for ``secrets-v1`` findings and unsafe integers.

    Returns location-only problem strings (``entry:pointer [rule]``) and never
    the matched value. Ciphertext (``private/vault.enc``) is skipped: it is
    pseudorandom and its plaintext was scanned before encryption. Entry names
    are scanned too. ``registry`` defaults to an empty registry, which is the
    correct posture for a transform that has no capture-time secret values
    (publication relies on capture-time redaction plus these detector rules).
    """
    active = registry if registry is not None else _SecretRegistry(())
    problems: list[str] = []
    for path, data in entries.items():
        if path == VAULT_ENC_PATH:
            # Ciphertext is pseudorandom; scanning it is meaningless.
            # Vault plaintext was scanned before encryption.
            continue
        if path.endswith(".ndjson"):
            lines = data.split(b"\n")
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    value = parse_json_strict(line.decode("utf-8"))
                except Exception as exc:
                    problems.append(f"{path}:{number}: unparsable ({exc})")
                    continue
                problems.extend(
                    f"{path}:{number}:{finding.pointer} [{finding.rule}]"
                    for finding in _scan_strings(value, "", active)
                )
                if _has_unsafe_int(value):
                    problems.append(f"{path}:{number} [unsafe_integer]")
        else:
            try:
                value = parse_json_strict(data.decode("utf-8"))
            except Exception as exc:
                problems.append(f"{path}: unparsable ({exc})")
                continue
            problems.extend(
                f"{path}:{finding.pointer} [{finding.rule}]"
                for finding in _scan_strings(value, "", active)
            )
            if _has_unsafe_int(value):
                problems.append(f"{path} [unsafe_integer]")
        if _scan_strings({"name": path}, "/name", active):
            problems.append(f"{path}: filename finding")
    return problems


def _zip_info(name: str) -> zipfile.ZipInfo:
    """Deterministic ZIP entry metadata for the NemoTrace container profile."""
    info = zipfile.ZipInfo(filename=name, date_time=ZIP_EPOCH)
    # Ciphertext is incompressible and must not be compressed before
    # or after encryption (spike §1); everything else is DEFLATE. The
    # deflate level is passed to `writestr` because `ZipInfo` only gained a
    # public `compress_level` attribute in Python 3.13; the
    # `compresslevel` keyword is the version-safe public API.
    if name == VAULT_ENC_PATH:
        info.compress_type = zipfile.ZIP_STORED
    else:
        info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = ZIP_UNIX_REGULAR << 16
    return info


def _write_zip_archive(path: Path, entries: Mapping[str, bytes]) -> None:
    """Deterministic ZIP write with an atomic rename (shared with the recorder)."""
    path = Path(path)
    if path.parent != Path() and str(path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(tmp, "w") as archive:
            for name in sorted(entries):
                archive.writestr(_zip_info(name), entries[name], compresslevel=ZIP_DEFLATE_LEVEL)
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def write_trace_archive(path: str | Path, entries: Mapping[str, bytes]) -> None:
    """Write one deterministic ``*.nemotrace`` ZIP atomically.

    Same ZIP profile as the recorder (sorted entries, fixed metadata,
    DEFLATE-6 except the STORE vault ciphertext); used by the publication
    transform, which rebuilds every entry from an audited source archive.
    """
    _write_zip_archive(Path(path), entries)


def read_archive_entries(path: str | Path) -> dict[str, bytes]:
    """Read and structurally validate a ``*.nemotrace`` container.

    Enforces entry allowlist, size/ratio budgets, and hash-not-yet semantics:
    returns uncompressed bytes keyed by entry name. Raises :class:`TraceError`
    on any structural violation.
    """
    path = Path(path)
    size = path.stat().st_size
    if size > LIMIT_COMPRESSED_BYTES:
        msg = f"trace archive exceeds {LIMIT_COMPRESSED_BYTES} compressed bytes"
        raise TraceError(msg)
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.comment:
                msg = "trace archive must not carry a comment"
                raise TraceError(msg)
            names = archive.namelist()
            if sorted(names) != names or len(set(names)) != len(names):
                msg = "trace archive entries must be sorted and unique"
                raise TraceError(msg)
            allowed = set(AUDIT_ENTRY_PATHS) | set(VAULT_ENTRY_PATHS) | {INTEGRITY_PATH}
            unknown = [n for n in names if n not in allowed]
            if unknown:
                msg = f"trace archive has unexpected entries: {unknown}"
                raise TraceError(msg)
            total = 0
            out: dict[str, bytes] = {}
            for info in archive.infolist():
                if info.is_dir():
                    msg = f"trace archive must not contain directories: {info.filename}"
                    raise TraceError(msg)
                # JSON/NDJSON entries are always DEFLATE; only the Phase 4
                # vault ciphertext is STORE (it is already pseudorandom).
                if info.filename == VAULT_ENC_PATH:
                    if info.compress_type != zipfile.ZIP_STORED:
                        msg = f"trace archive entry {info.filename} must use STORE"
                        raise TraceError(msg)
                elif info.compress_type != zipfile.ZIP_DEFLATED:
                    msg = f"trace archive entry {info.filename} must use DEFLATE"
                    raise TraceError(msg)
                if info.file_size > LIMIT_UNCOMPRESSED_ENTRY_BYTES:
                    msg = f"trace entry {info.filename} exceeds size budget"
                    raise TraceError(msg)
                ratio = info.file_size / max(info.compress_size, 1)
                if info.compress_size > 0 and ratio > LIMIT_COMPRESSION_RATIO:
                    msg = f"trace entry {info.filename} exceeds compression-ratio budget"
                    raise TraceError(msg)
                total += info.file_size
                if total > LIMIT_UNCOMPRESSED_TOTAL_BYTES:
                    msg = "trace archive exceeds uncompressed budget"
                    raise TraceError(msg)
                with archive.open(info) as handle:
                    out[info.filename] = handle.read()
            return out
    except zipfile.BadZipFile as exc:
        msg = f"not a valid trace archive: {exc}"
        raise TraceError(msg) from exc


def verify_archive(path: str | Path) -> VerificationReport:
    """Verify hashes, content identity, and summary consistency."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        entries = read_archive_entries(path)
    except TraceError as exc:
        return VerificationReport(ok=False, content_identity=None, warnings=(), errors=(str(exc),))
    errors.extend(
        f"missing required entry {required}"
        for required in (MANIFEST_PATH, GRAPH_PATH, EVENTS_PATH, INTEGRITY_PATH)
        if required not in entries
    )
    if errors:
        return VerificationReport(
            ok=False,
            content_identity=None,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )
    try:
        integrity: dict[str, Any] = parse_json_strict(entries[INTEGRITY_PATH].decode("utf-8"))
    except Exception as exc:
        return VerificationReport(
            ok=False,
            content_identity=None,
            warnings=(),
            errors=(f"integrity.json unparsable: {exc}",),
        )
    indexed: dict[str, dict[str, Any]] = {}
    raw_entries: Any = integrity.get("entries", [])  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(raw_entries, list):
        entry_list = cast("list[Any]", raw_entries)
        for idx, item in enumerate(entry_list):
            if not isinstance(item, dict):
                errors.append(f"integrity.json: entry {idx} must be an object")
                continue
            entry_map = cast("dict[Any, Any]", item)
            item_path: Any = entry_map.get("path")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            sha = entry_map.get("sha256")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            ub = entry_map.get("uncompressed_bytes")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(item_path, str):
                errors.append(f"integrity.json: entry {idx} missing or invalid path")
                continue
            if not isinstance(sha, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", sha):
                errors.append(f"integrity.json: entry {idx} invalid sha256")
                continue
            if not isinstance(ub, int) or ub < 0 or isinstance(ub, bool):
                errors.append(f"integrity.json: entry {idx} invalid uncompressed_bytes")
                continue
            # Only index valid entries to avoid KeyError downstream.
            if item_path not in indexed:
                indexed[item_path] = {"sha256": sha, "uncompressed_bytes": ub, "path": item_path}
            else:
                errors.append(f"integrity.json: duplicate path {item_path!r}")
    else:
        errors.append("integrity.json: entries must be an array")
    for name, data in entries.items():
        if name == INTEGRITY_PATH:
            continue
        entry = indexed.get(name)  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
        if entry is None:
            errors.append(f"{name} missing from integrity index")
            continue
        if entry.get("sha256") != sha256_tag(data):  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            errors.append(f"{name} hash mismatch")
        if entry.get("uncompressed_bytes") != len(data):  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            errors.append(f"{name} length mismatch")
    # Build content-identity projection safely (no KeyError on malformed entries).
    identity_entries: list[dict[str, Any]] = []
    for entry_path, entry in indexed.items():
        sha_v = entry.get("sha256")
        ub_v = entry.get("uncompressed_bytes")
        if isinstance(sha_v, str) and isinstance(ub_v, int):
            identity_entries.append(
                {"path": entry_path, "sha256": sha_v, "uncompressed_bytes": ub_v}
            )
    identity_obj: dict[str, Any] = {
        "format": CONTENT_IDENTITY_FORMAT,
        "entries": sorted(identity_entries, key=lambda item: item["path"]),  # type: ignore[reportUnknownArgumentType, reportUnknownLambdaType]
    }
    try:
        recomputed = sha256_tag(to_canonical_bytes(identity_obj))
    except Exception as exc:
        return VerificationReport(
            ok=False,
            content_identity=None,
            warnings=tuple(warnings),
            errors=(*errors, f"content identity failed: {exc}"),
        )
    content_identity = integrity.get("content_identity")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
    if recomputed != content_identity:
        errors.append("content identity mismatch")
    # Event count budget (hard failure, before inflation-heavy checks).
    try:
        event_lines = [line for line in entries[EVENTS_PATH].split(b"\n") if line.strip()]
        if len(event_lines) > LIMIT_EVENT_COUNT:
            errors.append(f"event count {len(event_lines)} exceeds budget {LIMIT_EVENT_COUNT}")
    except Exception as exc:
        errors.append(f"event count check failed: {exc}")
        event_lines = []
    # Summary is cache only: recompute and warn on mismatch.
    if SUMMARY_PATH in entries:
        try:
            summary: dict[str, Any] = parse_json_strict(entries[SUMMARY_PATH].decode("utf-8"))
            # Reuse event_lines counted above for consistency.
            kinds: dict[str, int] = {}
            visits: set[str] = set()
            for line in event_lines:
                event = parse_json_strict(line.decode("utf-8"))
                kinds[event.get("kind", "?")] = kinds.get(event.get("kind", "?"), 0) + 1  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if "stage_visit_id" in event:
                    visits.add(event["stage_visit_id"])
            if summary.get("event_count") != len(event_lines):  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                warnings.append("summary event_count mismatch (recomputed)")
            if summary.get("counts_by_kind") != kinds:  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                warnings.append("summary counts_by_kind mismatch (recomputed)")
            if summary.get("stage_visit_count") != len(visits):  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                warnings.append("summary stage_visit_count mismatch (recomputed)")
            if summary.get("events_sha256") != sha256_tag(entries[EVENTS_PATH]):  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                warnings.append("summary events_sha256 mismatch (recomputed)")
        except Exception as exc:
            warnings.append(f"summary check skipped: {exc}")
    manifest: dict[str, Any] | None = None
    manifest_ok = True
    try:
        manifest = parse_json_strict(entries[MANIFEST_PATH].decode("utf-8"))
        if not isinstance(manifest, dict):
            errors.append("manifest must be an object")
            manifest_ok = False
        else:
            required_manifest = {
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
            for field in required_manifest:
                if field not in manifest:
                    errors.append(f"manifest missing required field '{field}'")
                    manifest_ok = False
            extra = set(manifest.keys()) - required_manifest  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if extra:
                # Reader must tolerate forward-compatible fields (schema README §1);
                # unknown fields are warnings, not verification failures. Writers
                # remain strict via validate_phase0.py.
                warnings.append(f"manifest has unexpected fields {sorted(extra)} (ignored)")
                # Keep manifest_ok True; unknown fields do not invalidate.
            if manifest.get("format") != TRACE_FORMAT:  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                errors.append("manifest format mismatch")
                manifest_ok = False
            tid = manifest.get("trace_id")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(tid, str) or not re.fullmatch(r"[0-9a-f]{32}", tid):
                errors.append("manifest trace_id invalid")
                manifest_ok = False
            if manifest.get("status") not in ("complete", "failed", "interrupted"):  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                errors.append("manifest status invalid")
                manifest_ok = False
            cap = manifest.get("capture")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(cap, dict):
                errors.append("manifest capture invalid")
                manifest_ok = False
            wf = manifest.get("workflow")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(wf, dict):
                errors.append("manifest workflow invalid")
                manifest_ok = False
            else:
                for wf_f in ("id", "ir_version", "ir_sha256", "entry", "exits"):
                    if wf_f not in wf:
                        errors.append(f"manifest workflow missing '{wf_f}'")
                        manifest_ok = False
    except Exception as exc:
        errors.append(f"manifest unparsable: {exc}")
        manifest_ok = False
        manifest = None
    # Workflow graph validation (strict writer schema).
    try:
        graph_raw = entries[GRAPH_PATH].decode("utf-8")
        graph: Any = parse_json_strict(graph_raw)
        if not isinstance(graph, dict):
            errors.append("workflow graph must be an object")
        else:
            for req in (
                "format",
                "workflow_id",
                "entry",
                "exits",
                "nodes",
                "transitions",
                "policies",
            ):
                if req not in graph:
                    errors.append(f"workflow graph missing required field '{req}'")
            if graph.get("format") != "nemoir.trace.workflow-graph/0.1":  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                errors.append("workflow graph format mismatch")
            if "nodes" in graph and not isinstance(graph["nodes"], list):
                errors.append("workflow graph nodes must be an array")
    except Exception as exc:
        errors.append(f"workflow graph unparsable: {exc}")
        graph = None
    # Unsafe integer checks for every cleartext entry (portable I-JSON).
    try:
        if manifest is not None and _has_unsafe_int(manifest):
            errors.append("manifest contains unsafe integer")
        if isinstance(graph, dict) and _has_unsafe_int(graph):
            errors.append("workflow graph contains unsafe integer")
        if SUMMARY_PATH in entries:
            try:
                summary_obj = parse_json_strict(entries[SUMMARY_PATH].decode("utf-8"))
                if _has_unsafe_int(summary_obj):
                    errors.append("summary contains unsafe integer")
            except Exception:
                pass
        # integrity.json itself must be safe-integer clean
        if _has_unsafe_int(integrity):
            errors.append("integrity contains unsafe integer")
    except Exception:
        pass
    # Integrity index exact membership.
    expected_payloads = set(entries.keys()) - {INTEGRITY_PATH}  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
    indexed_keys = set(indexed.keys())  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
    if indexed_keys != expected_payloads:
        missing = sorted(expected_payloads - indexed_keys)
        extra_idx = sorted(indexed_keys - expected_payloads)
        if missing:
            errors.append(f"integrity index missing entries: {missing}")
        if extra_idx:
            errors.append(f"integrity index has extra entries: {extra_idx}")
    # Per-event ledger validation (strict writer schema).
    try:
        trace_id = manifest.get("trace_id") if isinstance(manifest, dict) else None  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
        allowed_kinds = {
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
        allowed_keys = {
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
        for idx, line in enumerate(event_lines, 1):
            try:
                ev: Any = parse_json_strict(line.decode("utf-8"))
            except Exception as exc:
                errors.append(f"public/events.ndjson:{idx} unparsable: {exc}")
                continue
            if not isinstance(ev, dict):
                errors.append(f"public/events.ndjson:{idx} must be an object")
                continue
            kind = ev.get("kind")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if kind not in allowed_kinds:
                errors.append(f"public/events.ndjson:{idx} invalid kind {kind!r}")
                continue
            extra_ev = set(ev.keys()) - allowed_keys  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if extra_ev:  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                # Forward-compatible reader ignores unknown fields (schema README §1).
                unexpected = sorted(extra_ev)  # type: ignore[reportUnknownArgumentType]
                warnings.append(
                    f"public/events.ndjson:{idx} has unexpected fields {unexpected} (ignored)"
                )
            rid = ev.get("run_id")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(rid, str) or not re.fullmatch(r"[0-9a-f]{32}", rid):
                errors.append(f"public/events.ndjson:{idx} invalid run_id")
            elif trace_id and rid != trace_id:
                errors.append(f"public/events.ndjson:{idx} run_id mismatch")
            ts = ev.get("timestamp")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(ts, str) or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z", ts
            ):
                errors.append(f"public/events.ndjson:{idx} invalid timestamp")
            rf = ev.get("redacted_fields")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(rf, list) or rf != sorted(set(rf)):  # type: ignore[reportUnknownArgumentType]
                errors.append(f"public/events.ndjson:{idx} invalid redacted_fields")
            if kind == "annotation":
                if "sequence" in ev:
                    errors.append(f"public/events.ndjson:{idx} annotation must not have sequence")
                if "annotation" not in ev:
                    errors.append(f"public/events.ndjson:{idx} annotation missing")
            elif "sequence" not in ev or not isinstance(ev["sequence"], int) or ev["sequence"] < 1:
                errors.append(f"public/events.ndjson:{idx} missing or invalid sequence")
            if kind in (
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
            ):
                if "stage_id" not in ev or "stage_visit_id" not in ev:
                    errors.append(f"public/events.ndjson:{idx} missing stage_id/stage_visit_id")
                else:
                    sid = ev.get("stage_id")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                    svid = ev.get("stage_visit_id")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                    if not isinstance(sid, str) or not (1 <= len(sid) <= 256):
                        errors.append(f"public/events.ndjson:{idx} invalid stage_id")
                    if not isinstance(svid, str) or not re.fullmatch(r"s-[1-9][0-9]*", svid):
                        errors.append(f"public/events.ndjson:{idx} invalid stage_visit_id")
            elif "stage_id" in ev:
                # stage_id present on non-stage event: validate type if present.
                sid = ev.get("stage_id")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(sid, str) or not (1 <= len(sid) <= 256):
                    errors.append(f"public/events.ndjson:{idx} invalid stage_id")
            # Validate other common optional string fields when present.
            if "stage_visit_id" in ev and "stage_id" not in ev:
                # stage_visit_id without stage_id is suspicious; check its shape anyway
                svid2 = ev.get("stage_visit_id")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(svid2, str) or not re.fullmatch(r"s-[1-9][0-9]*", svid2):
                    errors.append(f"public/events.ndjson:{idx} invalid stage_visit_id")
            if kind in ("model_delta", "model_completed", "model_retry"):
                mcid = ev.get("model_call_id")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(mcid, str) or not re.fullmatch(r"m-[1-9][0-9]*", mcid):
                    errors.append(f"public/events.ndjson:{idx} invalid model_call_id")
            if kind in ("tool_call_started", "tool_call_completed", "tool_call_failed"):
                tcid = ev.get("tool_call_id")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(tcid, str) or not re.fullmatch(r"t-[1-9][0-9]*", tcid):
                    errors.append(f"public/events.ndjson:{idx} invalid tool_call_id")
            # Validate remaining optional fields with type/bounds per public-event.schema.json
            if "channel" in ev:
                ch = ev.get("channel")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if ch is not None and ch not in (
                    "assistant",
                    "progress",
                    "reasoning",
                    "reasoning_summary",
                    "debug",
                ):
                    errors.append(f"public/events.ndjson:{idx} invalid channel")
            if "tool_name" in ev:
                tn = ev.get("tool_name")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if (
                    not isinstance(tn, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:\-]*", tn)
                    or len(tn) > 128
                ):
                    errors.append(f"public/events.ndjson:{idx} invalid tool_name")
            if "capability" in ev:
                cap = ev.get("capability")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(cap, str) or not (1 <= len(cap) <= 128):
                    errors.append(f"public/events.ndjson:{idx} invalid capability")
            if "error" in ev:
                err = ev.get("error")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if (
                    not isinstance(err, str)
                    or not re.fullmatch(r"[a-z][a-z0-9_]*", err)
                    or len(err) > 64
                ):
                    errors.append(f"public/events.ndjson:{idx} invalid error")
            if "transition_to" in ev:
                tr = ev.get("transition_to")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(tr, str) or not (1 <= len(tr) <= 256):
                    errors.append(f"public/events.ndjson:{idx} invalid transition_to")
            if "anchor_sequence" in ev:
                anc = ev.get("anchor_sequence")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(anc, int) or isinstance(anc, bool) or anc < 1:
                    errors.append(f"public/events.ndjson:{idx} invalid anchor_sequence")
            if "metadata" in ev:
                meta = ev.get("metadata")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(meta, dict):
                    errors.append(f"public/events.ndjson:{idx} invalid metadata")
            if "args" in ev and not isinstance(ev.get("args"), dict):  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                errors.append(f"public/events.ndjson:{idx} invalid args")
            if "output" in ev and not isinstance(ev.get("output"), dict):  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                errors.append(f"public/events.ndjson:{idx} invalid output")
            if "text" in ev:
                txt = ev.get("text")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(txt, dict) or "$redacted" not in txt:
                    errors.append(f"public/events.ndjson:{idx} invalid text")
            if "result" in ev:
                res = ev.get("result")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(res, dict) or "$redacted" not in res:
                    errors.append(f"public/events.ndjson:{idx} invalid result")
            if "annotation" in ev:
                ann = ev.get("annotation")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if (
                    not isinstance(ann, dict)
                    or "namespace" not in ann
                    or "kind" not in ann
                    or "payload" not in ann
                ):
                    errors.append(f"public/events.ndjson:{idx} invalid annotation")
                else:
                    ann_map = cast("dict[str, Any]", ann)
                    ns = ann_map.get("namespace")
                    kd = ann_map.get("kind")
                    pl = ann_map.get("payload")
                    if not isinstance(ns, str) or not (1 <= len(ns) <= 128):
                        errors.append(f"public/events.ndjson:{idx} invalid annotation namespace")
                    elif not isinstance(kd, str) or not (1 <= len(kd) <= 128):
                        errors.append(f"public/events.ndjson:{idx} invalid annotation kind")
                    elif ns == _ANNOTATION_NAMESPACE and kd == _ANNOTATION_KIND:
                        try:
                            _validate_autoresearch_payload(pl)  # type: ignore[arg-type]
                        except Exception as exc:
                            errors.append(
                                f"public/events.ndjson:{idx} invalid trial_finished payload: {exc}"
                            )
                    # Unknown annotations must be a redaction marker per policy §11.
                    elif not (isinstance(pl, dict) and "$redacted" in pl):
                        errors.append(
                            f"public/events.ndjson:{idx} unknown annotation payload must be a redaction marker"
                        )
            if _has_unsafe_int(ev):
                errors.append(f"public/events.ndjson:{idx} contains unsafe integer")
    except Exception as exc:
        errors.append(f"event validation failed: {exc}")
    _ = manifest_ok
    # -- Phase 4: capture/vault consistency (errors) ----------------------
    vault_present = False
    capture_profile = cast("Any", None)
    if isinstance(manifest, dict):
        manifest_branch = cast("dict[Any, Any]", manifest)
        cap = manifest_branch.get("capture")
        if isinstance(cap, dict):
            capture_branch = cast("Any", cap)
            vault_present = capture_branch.get("vault_present") is True
            capture_profile = capture_branch.get("profile")
    has_vault_enc = VAULT_ENC_PATH in entries
    has_vault_meta = VAULT_META_PATH in entries
    if has_vault_enc != has_vault_meta:
        errors.append("vault entries must co-occur: private/vault.enc + private/vault.meta.json")
    has_vault = has_vault_enc and has_vault_meta
    if vault_present and not has_vault:
        errors.append("manifest capture declares vault_present but vault entries are missing")
    if has_vault and not vault_present:
        errors.append("vault entries present but manifest capture has vault_present=false")
    if capture_profile == "replay" and not vault_present:
        errors.append("replay profile requires vault_present=true")
    if capture_profile == "publication" and has_vault:
        errors.append("publication profile forbids a vault")
    if capture_profile not in ("audit", "replay", "publication"):
        errors.append(f"manifest capture has invalid profile {capture_profile!r}")
    # -- Phase 4: structural path-shape check (warnings, never silent) ----
    structural_notes: list[str] = []
    try:
        graph_branch = cast("Any", graph)
        graph_nodes = cast("Any", graph_branch.get("nodes") if isinstance(graph, dict) else None)
        if isinstance(graph_nodes, list):
            node_ids: set[Any] = set()
            for node_entry in cast("Any", graph_nodes):
                node = node_entry
                if isinstance(node, dict):
                    node_ids.add(cast("dict[Any, Any]", node).get("id"))
            edges: set[Any] = set()
            graph_transitions = graph_branch.get("transitions", [])
            if isinstance(graph_transitions, list):
                for transition_entry in cast("Any", graph_transitions):
                    transition = transition_entry
                    if isinstance(transition, dict):
                        transition_mapping = cast("dict[Any, Any]", transition)
                        edges.add((transition_mapping.get("from"), transition_mapping.get("to")))
            unknown_stages: set[Any] = set()
            bad_edges: set[Any] = set()
            for line in event_lines:
                ledger_event = _parse_ledger_line(line)
                if ledger_event is None or not isinstance(ledger_event, dict):
                    continue
                ledger = cast("Any", ledger_event)
                kind = ledger.get("kind")
                if kind == "stage_started":
                    stage_name = ledger.get("stage_id")
                    if isinstance(stage_name, str) and stage_name not in node_ids:
                        unknown_stages.add(stage_name)
                if kind == "transition_selected":
                    edge = (ledger.get("stage_id"), ledger.get("transition_to"))
                    if edge not in edges:
                        bad_edges.add(edge)
            for stage in sorted(unknown_stages)[:10]:
                structural_notes.append(
                    f"ledger references stage {stage!r} absent from workflow graph"
                )
            if len(unknown_stages) > 10:
                structural_notes.append(f"... and {len(unknown_stages) - 10} more unknown stages")
            for edge in sorted(bad_edges, key=str)[:10]:
                structural_notes.append(
                    f"ledger transition {edge[0]!r}->{edge[1]!r} absent from workflow graph"
                )
            if len(bad_edges) > 10:
                structural_notes.append(f"... and {len(bad_edges) - 10} more unknown transitions")
    except Exception as exc:
        structural_notes.append(f"structural check skipped: {exc}")
    warnings.extend(structural_notes)
    # -- Phase 4: level summary -------------------------------------------
    integrity_markers = (
        "hash mismatch",
        "length mismatch",
        "content identity",
        "missing from integrity",
        "integrity index",
        "integrity.json",
        "integrity contains",
        "missing required entry",
        "unparsable",
    )
    integrity_failed = any(m in e for e in errors for m in integrity_markers)
    ledger_markers = ("events.ndjson", "workflow graph", "manifest")
    ledger_failed = any(m in e for e in errors for m in ledger_markers)
    integrity_level = "failed" if integrity_failed else "passed"
    if ledger_failed:
        structural_level = "failed"
    elif structural_notes:
        structural_level = "passed-with-warnings"
    else:
        structural_level = "passed"
    ok = not errors
    if ok and has_vault:
        replayability = "taped-replay"
    elif ok:
        replayability = "playback-only"
    else:
        replayability = "none"
    return VerificationReport(
        ok=ok,
        content_identity=content_identity if ok else None,
        warnings=tuple(warnings),
        errors=tuple(errors),
        integrity=integrity_level,
        structural=structural_level,
        semantic="not-evaluated",
        replayability=replayability,
    )


def _check_vault_meta(meta: Any) -> dict[str, Any]:
    """Validate vault metadata shape and frozen codec parameters."""
    if not isinstance(meta, dict):
        msg = "vault metadata must be an object"
        raise TraceError(msg)
    descriptor = cast("dict[Any, Any]", meta)
    if descriptor.get("format") != VAULT_META_FORMAT:
        msg = "vault metadata format mismatch"
        raise TraceError(msg)
    if descriptor.get("codec") != VAULT_CODEC:
        msg = "vault uses an unsupported codec"
        raise TraceError(msg)
    passphrase = cast("Any", descriptor.get("passphrase"))
    if not isinstance(passphrase, dict):
        msg = "vault metadata passphrase descriptor invalid"
        raise TraceError(msg)
    passphrase_format = cast("dict[Any, Any]", passphrase)
    if passphrase_format.get("encoding") != "UTF-8":
        msg = "vault metadata passphrase descriptor invalid"
        raise TraceError(msg)
    passphrase_descriptor = cast("dict[Any, Any]", passphrase)
    if passphrase_descriptor.get("normalization") != "NFC":
        msg = "vault metadata passphrase descriptor invalid"
        raise TraceError(msg)
    kdf = cast("Any", descriptor.get("kdf"))
    if not isinstance(kdf, dict):
        msg = "vault uses an unsupported KDF"
        raise TraceError(msg)
    kdf_name = cast("dict[Any, Any]", kdf)
    if kdf_name.get("name") != VAULT_KDF_NAME:
        msg = "vault uses an unsupported KDF"
        raise TraceError(msg)
    kdf_descriptor = cast("dict[Any, Any]", kdf)
    if kdf_descriptor.get("iterations") != VAULT_KDF_ITERATIONS:
        # A lower work factor must never appear conformant (spike §1).
        msg = "vault uses an unsupported KDF"
        raise TraceError(msg)
    if kdf_descriptor.get("derived_key_bits") != VAULT_DERIVED_KEY_BITS:
        msg = "vault uses an unsupported KDF"
        raise TraceError(msg)
    cipher = cast("Any", descriptor.get("cipher"))
    if not isinstance(cipher, dict):
        msg = "vault uses an unsupported cipher"
        raise TraceError(msg)
    cipher_name = cast("dict[Any, Any]", cipher)
    if cipher_name.get("name") != "AES-256-GCM":
        msg = "vault uses an unsupported cipher"
        raise TraceError(msg)
    cipher_descriptor = cast("dict[Any, Any]", cipher)
    if cipher_descriptor.get("tag_length_bits") != VAULT_TAG_BITS:
        msg = "vault uses an unsupported cipher"
        raise TraceError(msg)
    if cipher_descriptor.get("tag_placement") != "ciphertext_suffix":
        msg = "vault uses an unsupported cipher"
        raise TraceError(msg)
    plaintext = cast("Any", descriptor.get("plaintext"))
    if not isinstance(plaintext, dict):
        msg = "vault metadata plaintext descriptor invalid"
        raise TraceError(msg)
    plaintext_descriptor = cast("dict[Any, Any]", plaintext)
    if (
        plaintext_descriptor.get("media_type") != VAULT_PLAINTEXT_MEDIA_TYPE
        or plaintext_descriptor.get("encoding") != "UTF-8"
        or plaintext_descriptor.get("compression") != "none"
    ):
        msg = "vault metadata plaintext descriptor invalid"
        raise TraceError(msg)
    aad = cast("Any", descriptor.get("aad"))
    if not isinstance(aad, dict):
        msg = "vault metadata AAD descriptor invalid"
        raise TraceError(msg)
    aad_format = cast("dict[Any, Any]", aad)
    if aad_format.get("format") != VAULT_AAD_FORMAT:
        msg = "vault metadata AAD descriptor invalid"
        raise TraceError(msg)
    aad_descriptor = cast("dict[Any, Any]", aad)
    for aad_name in (
        "trace_id",
        "ir_sha256",
        "manifest_sha256",
        "events_sha256",
        "workflow_graph_sha256",
    ):
        if aad_name not in aad_descriptor:
            msg = f"vault metadata AAD missing {aad_name!r}"
            raise TraceError(msg)
    return cast("dict[str, Any]", meta)


def _expected_vault_aad(
    entries: dict[str, bytes], manifest: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    """Recompute the vault AAD from the verified public artifact."""
    workflow = manifest.get("workflow", {})
    aad_dict: dict[str, Any] = {
        "events_sha256": sha256_tag(entries[EVENTS_PATH]),
        "format": VAULT_AAD_FORMAT,
        "ir_sha256": workflow.get("ir_sha256") or VAULT_NULL_IR_SHA256,
        "manifest_sha256": sha256_tag(entries[MANIFEST_PATH]),
        "trace_id": manifest.get("trace_id"),
        "workflow_graph_sha256": sha256_tag(entries[GRAPH_PATH]),
    }
    return to_canonical_bytes(aad_dict), aad_dict


def unlock_archive(
    path: str | Path, passphrase: str | bytes
) -> tuple[list[dict[str, Any]], VerificationReport]:
    """Decrypt a replay vault and evaluate semantic evidence completeness.

    Returns ``(vault_records, report)`` where ``report.semantic`` is
    ``"passed"`` when every ledger model/tool/transition event has matching
    vault evidence, else ``"failed"`` with errors. Wrong passphrases and
    modified vaults fail closed with a generic error. Decrypted records are
    returned in memory only; callers must never persist them without an
    explicit user action.
    """
    report = verify_archive(path)
    if not report.ok:
        return [], replace(
            report,
            ok=False,
            semantic="failed",
            errors=(*report.errors, "archive verification failed"),
        )
    if report.replayability != "taped-replay":
        return [], replace(
            report,
            ok=False,
            semantic="failed",
            errors=(*report.errors, "archive has no replay vault"),
        )
    try:
        entries = read_archive_entries(path)
    except TraceError as exc:
        return [], replace(report, ok=False, semantic="failed", errors=(*report.errors, str(exc)))
    try:
        meta_raw = parse_json_strict(entries[VAULT_META_PATH].decode("utf-8"))
        manifest_raw = parse_json_strict(entries[MANIFEST_PATH].decode("utf-8"))
    except Exception as exc:
        return [], replace(
            report,
            ok=False,
            semantic="failed",
            errors=(*report.errors, f"vault metadata invalid: {exc}"),
        )
    try:
        meta = _check_vault_meta(meta_raw)
    except TraceError as exc:
        return [], replace(
            report,
            ok=False,
            semantic="failed",
            errors=(*report.errors, f"vault metadata invalid: {exc}"),
        )
    if not isinstance(manifest_raw, dict):
        return [], replace(
            report,
            ok=False,
            semantic="failed",
            errors=(*report.errors, "vault metadata invalid: manifest must be an object"),
        )
    manifest = cast("dict[Any, Any]", manifest_raw)
    meta_branch = cast("dict[Any, Any]", meta)
    kdf = cast("Any", meta_branch.get("kdf"))
    cipher = cast("Any", meta_branch.get("cipher"))
    try:
        salt = _b64url_decode(kdf["salt_base64url"], what="salt", expected=VAULT_SALT_BYTES)
        nonce = _b64url_decode(cipher["nonce_base64url"], what="nonce", expected=VAULT_NONCE_BYTES)
    except TraceError as exc:
        return [], replace(
            report,
            ok=False,
            semantic="failed",
            errors=(*report.errors, f"vault metadata invalid: {exc}"),
        )
    try:
        pw_bytes = _normalize_passphrase(passphrase)
    except TraceError:
        pw_bytes = None
    if pw_bytes is None:
        return [], replace(
            report, ok=False, semantic="failed", errors=(*report.errors, "vault unlock failed")
        )
    expected_aad, _ = _expected_vault_aad(entries, manifest)
    stored_aad = cast("Any", meta_branch.get("aad"))
    # The stored AAD descriptor must agree with the recomputed public
    # artifact before decryption is attempted.
    aad_ok = True
    for aad_name, digest in (
        ("manifest_sha256", sha256_tag(entries[MANIFEST_PATH])),
        ("events_sha256", sha256_tag(entries[EVENTS_PATH])),
        ("workflow_graph_sha256", sha256_tag(entries[GRAPH_PATH])),
    ):
        stored_descriptor = stored_aad
        if (
            not isinstance(stored_descriptor, dict)
            or cast("dict[Any, Any]", stored_descriptor).get(aad_name) != digest
        ):
            aad_ok = False
    manifest_branch = manifest
    if isinstance(stored_aad, dict):
        stored_trace = cast("dict[Any, Any]", stored_aad)
    else:
        stored_trace = cast("dict[Any, Any]", {})
    if not aad_ok or stored_trace.get("trace_id") != manifest_branch.get("trace_id"):
        return [], replace(
            report, ok=False, semantic="failed", errors=(*report.errors, "vault unlock failed")
        )
    try:
        plaintext = decrypt_vault_records(
            entries[VAULT_ENC_PATH], pw_bytes, expected_aad, salt=salt, nonce=nonce
        )
    except TraceError as exc:
        return [], replace(report, ok=False, semantic="failed", errors=(*report.errors, str(exc)))
    try:
        raw_lines = [line for line in plaintext.split(b"\n") if line.strip()]
        parsed: list[Any] = [parse_json_strict(line.decode("utf-8")) for line in raw_lines]
    except Exception as exc:
        return [], replace(
            report, ok=False, semantic="failed", errors=(*report.errors, f"vault invalid: {exc}")
        )
    if len(parsed) > LIMIT_EVENT_COUNT:
        return [], replace(
            report,
            ok=False,
            semantic="failed",
            errors=(*report.errors, "vault invalid: record count exceeds budget"),
        )
    records: list[Any] = []
    for number, raw_record in enumerate(parsed, 1):
        record = raw_record
        if not isinstance(record, dict):
            return [], replace(
                report,
                semantic="failed",
                errors=(*report.errors, f"vault invalid: record {number} must be an object"),
            )
        shape_error = _vault_record_shape_error(cast("Any", record))
        if shape_error is not None:
            return [], replace(
                report,
                semantic="failed",
                errors=(*report.errors, f"vault invalid: {shape_error}"),
            )
        records.append(record)
    semantic_errors = _check_vault_evidence(entries, records)
    semantic = "passed" if not semantic_errors else "failed"
    return records, replace(
        report,
        semantic=semantic,
        warnings=report.warnings,
        errors=(*report.errors, *semantic_errors),
        ok=report.ok and not semantic_errors,
    )


def _check_vault_evidence(entries: dict[str, bytes], records: list[dict[str, Any]]) -> list[str]:
    """Check that every replay-relevant ledger event has vault evidence."""
    problems: list[str] = []
    model_ids: set[Any] = set()
    tool_ids: set[Any] = set()
    transition_visits: set[Any] = set()
    for record in records:
        entry = cast("dict[Any, Any]", record)
        rtype = cast("Any", entry.get("record_type"))
        if rtype in ("model_request", "model_response"):
            call = cast("Any", entry.get("model_call_id"))
            if isinstance(call, str):
                model_ids.add(call)
        elif rtype == "tool_result":
            tool = cast("Any", entry.get("tool_call_id"))
            if isinstance(tool, str):
                tool_ids.add(tool)
        elif rtype == "transition_evaluation":
            seen_visit = cast("Any", entry.get("stage_visit_id"))
            if isinstance(seen_visit, str):
                transition_visits.add(seen_visit)
    for line in entries[EVENTS_PATH].split(b"\n"):
        if not line.strip():
            continue
        event = _parse_ledger_line(line)
        if not isinstance(event, dict):
            continue
        ledger = cast("Any", event)
        kind = ledger.get("kind")
        if kind in ("model_completed", "model_retry"):
            mid = ledger.get("model_call_id")
            if isinstance(mid, str) and mid not in model_ids:
                problems.append(f"vault missing model evidence for {mid}")
        elif kind in ("tool_call_completed", "tool_call_failed"):
            tid = ledger.get("tool_call_id")
            if isinstance(tid, str) and tid not in tool_ids:
                problems.append(f"vault missing tool evidence for {tid}")
        elif kind == "transition_selected":
            visit = ledger.get("stage_visit_id")
            if isinstance(visit, str) and visit not in transition_visits:
                problems.append(f"vault missing transition evidence for visit {visit}")
    return problems[:50]


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def _safe_int(value: Any, default: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value >= minimum:
        return value
    return default


def _safe_category(value: Any) -> str:
    if (
        isinstance(value, str)
        and len(value) <= _MAX_CATEGORY_LEN
        and re.fullmatch(r"[a-z][a-z0-9_]*", value)
    ):
        return value
    return "other"


def _is_metric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _output_reason(value: Any) -> str:
    if isinstance(value, Path):
        return "absolute_path"
    return "private_content"
