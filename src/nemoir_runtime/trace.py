"""NemoTrace audit recorder and single-file archive (Phase 1).

Turns a run's live :class:`WorkflowEvent` stream plus recorder semantic
hooks into one portable ``*.nemotrace`` ZIP artifact: a redacted public
ledger, a safe workflow-graph projection, a derived summary cache, and an
integrity index binding everything to an exact compiled-IR fingerprint.

Security model (see ``docs/trace/redaction-policy.md``):

- A value is absent from cleartext unless an allowlist rule permits it.
- Redaction happens at capture time, before any journal/archive write. The
  viewer is never a redaction boundary.
- ``audit`` is the only Phase 1 profile. There is no cleartext full-capture
  profile; ``replay``/``publication`` arrive in later phases and this module
  refuses them explicitly rather than silently producing them.
- Credentials never enter the ledger or any vault. The secret-value registry
  is defense-in-depth against echoed credentials, not the primary control.
- The final cleartext scanner blocks archive finalization on any unresolved
  finding and reports locations, never values.

Cross-language contract: ``docs/trace/schema/README.md`` is normative for
the wire form. Canonical JSON bytes come from
:mod:`nemoir_runtime.canonical` (RFC 8785); the Rust
(``nemoir-ir/src/canonical.rs``) and TypeScript (``canonical.ts``) ports must
produce byte-identical uncompressed entries for the same logical run.

This module is deliberately stdlib-only so tracing never adds required
runtime dependencies.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import uuid
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
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
AUDIT_ENTRY_PATHS = (MANIFEST_PATH, GRAPH_PATH, EVENTS_PATH, SUMMARY_PATH)

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
class TraceConfig:
    """Immutable per-run trace configuration owned by the host.

    One recorder instance represents one run; generated agents use a factory
    or a fresh config per invocation, never a shared recorder.
    """

    path: Path
    profile: str = "audit"
    provenance: HostProvenance = field(default_factory=HostProvenance)
    model: ModelDescriptor | None = None
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


def _iter_strings(
    node: Any, pointer: str
) -> Iterator[tuple[str, str | None, str | None]]:
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
    return pointer in ("/text", "/result") or pointer.startswith(
        ("/output/", "/annotation/payload/")
    ) or pointer in _MASKABLE_PREFIXES


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

        workflow_text = resources.files(package_name).joinpath("workflow.json").read_text(
            encoding="utf-8"
        )
        provenance_text = (
            resources.files(package_name).joinpath("trace-provenance.json").read_text(
                encoding="utf-8"
            )
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


class TraceRecorder:
    """Redacted audit recorder for one run. See module docstring for the
    security model. Obtain via :meth:`create`; exactly one ``begin_run`` then
    at most one ``finish_run`` per instance."""

    def __init__(self, config: TraceConfig) -> None:
        if config.profile != "audit":
            msg = (
                f"unsupported trace profile '{config.profile}': Phase 1 supports "
                f"only 'audit' (replay arrives in Phase 4, publication in Phase 5)"
            )
            raise TraceError(msg)
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
        self._pending_tools: dict[str, list[str]] = {}
        self._open_tools: dict[str, list[str]] = {}
        self._current_visit: str | None = None
        self._model_bytes: dict[str, int] = {}
        self._model_tool_calls: dict[str, int] = {}
        self._tool_started_at: dict[str, datetime] = {}
        self._tool_result_types: dict[str, str] = {}
        self._tool_errors: dict[str, tuple[str, str]] = {}
        self._transition_evidence: list[dict[str, Any]] = []
        self._visit_count = 0
        self._model_count = 0
        self._tool_count = 0
        self._redaction_count = 0
        self._path_ref_count = 0
        self._path_refs: dict[str, str] = {}
        self._stage_completed_count = 0
        self._event_limit_exceeded = False

    # -- construction ----------------------------------------------------

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
            )
        )

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def config(self) -> TraceConfig:
        """The immutable host configuration this recorder was created with."""
        return self._config

    # -- run lifecycle ---------------------------------------------------

    def begin_run(self, manifest: Any) -> None:
        """Attach the workflow manifest before ``run_started`` is observed."""
        if self._begun:
            msg = "TraceRecorder.begin_run called twice: one recorder per run"
            raise TraceError(msg)
        self._begun = True
        self._begin_time = self._clock()
        self._manifest = manifest
        for index, policy in enumerate(getattr(manifest, "policies", ()) or (), 1):
            policy_id = getattr(policy, "id", None)
            if isinstance(policy_id, str) and policy_id not in self._policy_refs:
                self._policy_refs[policy_id] = f"p-{index}"
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
        return visit_id

    def begin_model_call(
        self, _stage_id: str = "", stage_visit_id: str | None = None
    ) -> str:
        """Assign the next run-local model-call id."""
        self._require_begun()
        self._model_count += 1
        call_id = f"m-{self._model_count}"
        visit = stage_visit_id or self._current_visit or ""
        self._pending_models.setdefault(visit, []).append(call_id)
        return call_id

    def record_model_response(
        self,
        model_call_id: str,
        *,
        response_bytes: int = 0,
        tool_call_count: int = 0,
    ) -> None:
        """Record safe model-response facts (counts only, never text)."""
        self._require_begun()
        self._model_bytes[model_call_id] = max(0, int(response_bytes))
        self._model_tool_calls[model_call_id] = max(0, int(tool_call_count))

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

    def record_tool_result(self, tool_call_id: str, result: Any) -> None:
        """Note a tool result's safe type facts (the value stays private)."""
        self._require_begun()
        self._tool_errors.pop(tool_call_id, None)
        self._tool_result_types[tool_call_id] = _result_type_slug(result)

    def record_tool_error(self, tool_call_id: str, exc: BaseException) -> None:
        """Capture a tool failure's stable error taxonomy (no message)."""
        self._require_begun()
        self._tool_errors[tool_call_id] = stable_error(exc)

    def record_transition_evaluation(
        self,
        stage_visit_id: str,
        candidates: list[dict[str, Any]],
    ) -> None:
        """Retain guard-evaluation evidence for future semantic verification.

        Phase 1 keeps this in memory only: the audit ledger publishes the
        selected transition, while full candidate evidence ships with the
        Phase 4 vault.
        """
        self._require_begun()
        self._transition_evidence.append(
            {"stage_visit_id": stage_visit_id, "candidates": list(candidates)}
        )

    def record_annotation(
        self,
        namespace: str,
        kind: str,
        _payload: Mapping[str, Any],
        _anchor_sequence: int | None = None,
    ) -> None:
        """Trusted domain annotations land in Phase 3; refuse loudly until then."""
        self._require_begun()
        msg = (
            f"trace annotations ({namespace}/{kind}) arrive in Phase 3; "
            f"the Phase 1 audit recorder cannot persist them"
        )
        raise TraceError(msg)

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
        return record

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
        return visit

    def _peek_model(self, visit: str) -> str | None:
        queue = self._pending_models.get(visit)
        return queue[-1] if queue else None

    def _pop_model(self, visit: str) -> str:
        queue = self._pending_models.get(visit)
        if queue:
            return queue.pop(0)
        self._model_count += 1
        return f"m-{self._model_count}"

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
        call_id = self._peek_model(visit)
        if call_id is not None:
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
                list(cast("Sequence[Any]", required))
                if isinstance(required, (list, tuple))
                else []
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
                candidate = (Path.cwd() / candidate)
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
            projected = {
                key: self._new_marker("private_content", args[key]) for key in args
            }
            redacted.extend(f"/args/{key}" for key in args)
            return projected, redacted
        if capability in (
            "browser.storage.read",
            "browser.storage.write",
            "browser.js.run",
            "browser.js.sandbox",
        ):
            projected = {
                key: self._new_marker("private_content", args[key]) for key in args
            }
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
            else:
                return
        last = parts[-1]
        if isinstance(current, dict) and last in current:
            current[last] = self._new_marker(reason, current[last])
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
                prov.target
                if prov.target in ("python", "web", "manual", "imported")
                else "manual"
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
        manifest_obj: dict[str, Any] = {
            "format": TRACE_FORMAT,
            "trace_id": self._trace_id,
            "created_at": format_timestamp(self._begin_time or self._now()),
            "status": status,
            "capture": {
                "profile": "audit",
                "vault_present": False,
                "publication_eligible": False,
                "redaction_policy": REDACTION_POLICY,
                "scanner": {"status": "passed", "ruleset": SCANNER_RULESET},
            },
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
        }
        payloads: dict[str, bytes] = {
            MANIFEST_PATH: to_canonical_bytes(manifest_obj),
            GRAPH_PATH: to_canonical_bytes(graph_obj),
            EVENTS_PATH: events_bytes,
            SUMMARY_PATH: to_canonical_bytes(summary_obj),
        }
        integrity_entries = [
            {
                "path": path,
                "media_type": (
                    "application/x-ndjson"
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
            for trans in (getattr(stage, "transitions", ()) or ()):
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
        problems: list[str] = []
        for path, data in entries.items():
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
                        for finding in _scan_strings(value, "", self._registry)
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
                    for finding in _scan_strings(value, "", self._registry)
                )
                if _has_unsafe_int(value):
                    problems.append(f"{path} [unsafe_integer]")
            if _scan_strings({"name": path}, "/name", self._registry):
                problems.append(f"{path}: filename finding")
        if problems:
            self._remove_partial_marker()
            detail = "; ".join(problems[:10])
            msg = f"trace scanner blocked finalization with {len(problems)} finding(s): {detail}"
            raise TraceError(msg)

    # -- archive IO ------------------------------------------------------

    @staticmethod
    def _zip_info(name: str) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(filename=name, date_time=ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.compress_level = ZIP_DEFLATE_LEVEL
        info.create_system = 3
        info.external_attr = ZIP_UNIX_REGULAR << 16
        return info

    @classmethod
    def _write_archive(cls, path: Path, entries: dict[str, bytes]) -> None:
        path = Path(path)
        if path.parent != Path() and str(path.parent):
            path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        try:
            with zipfile.ZipFile(tmp, "w") as archive:
                for name in sorted(entries):
                    archive.writestr(cls._zip_info(name), entries[name])
            tmp.replace(path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

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

    def begin_run(self, _manifest: Any) -> None:
        return None

    def finish_run(self, _status: str) -> None:
        return None

    def begin_stage_visit(self, _stage_id: str) -> str:
        return ""

    def begin_model_call(
        self, _stage_id: str = "", _stage_visit_id: str | None = None
    ) -> str:
        return ""

    def record_model_response(self, _model_call_id: str, **_kwargs: Any) -> None:
        return None

    def begin_tool_call(
        self, _stage_id: str = "", _stage_visit_id: str | None = None
    ) -> str:
        return ""

    def record_tool_result(self, _tool_call_id: str, _result: Any) -> None:
        return None

    def record_tool_error(self, _tool_call_id: str, _exc: BaseException) -> None:
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
        _payload: Mapping[str, Any],
        _anchor_sequence: int | None = None,
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
            msg = (
                "trace factory must return TraceRecorder or None, "
                f"got {type(recorder).__name__}"
            )
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
            allowed = set(AUDIT_ENTRY_PATHS) | {INTEGRITY_PATH}
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
                # Phase 1 audit entries are always DEFLATE; STORE members
                # are rejected (the Phase 4 vault is the only STORE entry).
                if info.compress_type != zipfile.ZIP_DEFLATED:
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
            identity_entries.append({"path": entry_path, "sha256": sha_v, "uncompressed_bytes": ub_v})
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
            for req in ("format", "workflow_id", "entry", "exits", "nodes", "transitions", "policies"):
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
                warnings.append(f"public/events.ndjson:{idx} has unexpected fields {sorted(extra_ev)} (ignored)")  # type: ignore[reportUnknownArgumentType]
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
                if ch is not None and ch not in ("assistant", "progress", "reasoning", "reasoning_summary", "debug"):
                    errors.append(f"public/events.ndjson:{idx} invalid channel")
            if "tool_name" in ev:
                tn = ev.get("tool_name")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(tn, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:\-]*", tn) or len(tn) > 128:
                    errors.append(f"public/events.ndjson:{idx} invalid tool_name")
            if "capability" in ev:
                cap = ev.get("capability")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(cap, str) or not (1 <= len(cap) <= 128):
                    errors.append(f"public/events.ndjson:{idx} invalid capability")
            if "error" in ev:
                err = ev.get("error")  # type: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(err, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", err) or len(err) > 64:
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
                if not isinstance(ann, dict) or "namespace" not in ann or "kind" not in ann or "payload" not in ann:
                    errors.append(f"public/events.ndjson:{idx} invalid annotation")
            if _has_unsafe_int(ev):
                errors.append(f"public/events.ndjson:{idx} contains unsafe integer")
    except Exception as exc:
        errors.append(f"event validation failed: {exc}")
    _ = manifest_ok
    return VerificationReport(
        ok=not errors,
        content_identity=content_identity if not errors else None,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


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
