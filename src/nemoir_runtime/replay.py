"""NemoTrace taped replay (Phase 4, Option B standalone core).

Re-executes a recorded run's state-machine path with **no live effects**:
model calls are served from vault ``model_response`` fixtures and tool
calls from ``tool_result`` fixtures. There is no provider client, no real
tool handler, and no human interaction anywhere in the replay closure —
a fixture miss or a live-effect attempt fails closed with
:class:`TapedReplayError` instead of falling back to live execution.

Replay semantics (see ``docs/trace/plan.md`` §7.6):

- ``playback`` (public ledger only) navigates the record; nothing executes.
- ``taped replay`` (this module) re-executes control flow against fixtures
  and checks the replayed path against the recorded ledger.
- ``live rerun`` (out of scope) would call real providers/tools and is not
  deterministic; this module never does that.

Fidelity limits, stated honestly:

- Secret-bearing values never enter the vault, so runs whose *control flow*
  depends on a secret value (guards branching on credentials, secret-derived
  args) report a divergence instead of matching. Pass-through secrets served
  to taped tools do not affect control and replay fine.
- Comparison is capability-level for tools (stub names differ from recorded
  names by construction) and marker-aware for outputs (redacted snapshot
  fields cannot be verified and are skipped, never assumed).
- ``interrupted`` traces are refused: a partial path is evidence, not a
  replayable run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from nemoir_runtime.canonical import parse_json_strict
from nemoir_runtime.errors import ToolInvocationError
from nemoir_runtime.models import (
    ModelRequest,
    ModelResponse,
    ModelStageExecutor,
    ModelToolCall,
)
from nemoir_runtime.runtime import (
    ExprSpec,
    GuardSpec,
    InputSpec,
    PolicySpec,
    ReadSpec,
    RefSpec,
    RequiredCapabilitySpec,
    RunOptions,
    StageExecutionSpec,
    StageSpec,
    ToolRegistry,
    TransitionSpec,
    TriggerSpec,
    WorkflowManifest,
    WorkflowRuntime,
    WriteSpec,
)
from nemoir_runtime.tools import Tool, ToolContext

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from nemoir_runtime.events import WorkflowEvent
from nemoir_runtime.trace import (
    NoOpTraceRecorder,
    VerificationReport,
    policy_refs_for,
    read_archive_entries,
    unlock_archive,
)


def _policy_tape_for_replay(
    records: list[dict[str, Any]], manifest: WorkflowManifest
) -> dict[str, list[str]]:
    """Recorded deny-policy outcomes keyed by policy id, in capture order.

    The runtime consumes them FIFO instead of re-evaluating expressions
    against scrubbed fixture arguments. Vault refs resolve through the same
    declaration-order rule the recorder used when writing them."""
    refs = policy_refs_for(manifest.policies)
    id_by_ref = {ref: pid for pid, ref in refs.items()}
    tape: dict[str, list[str]] = {}
    for record in records:
        if record.get("record_type") != "policy_evaluation":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_map = cast("dict[Any, Any]", payload)
        ref = payload_map.get("policy_ref")
        outcome = payload_map.get("outcome")
        policy_id = id_by_ref.get(ref) if isinstance(ref, str) else None
        if policy_id is None or outcome not in ("allowed", "denied"):
            continue
        tape.setdefault(policy_id, []).append(outcome)
    return tape


class TapedReplayError(Exception):
    """Raised when a taped replay cannot proceed without live effects."""


@dataclass(frozen=True)
class ReplayReport:
    """Outcome of :func:`replay_trace`."""

    matched: bool
    divergences: tuple[str, ...]
    verification: VerificationReport
    steps: int = 0
    replayed_status: str = "unknown"


# ---------------------------------------------------------------------------
# Manifest reconstruction (vault full_workflow_ir -> WorkflowManifest)
# ---------------------------------------------------------------------------


def _req_dict(value: Any, what: str) -> dict[Any, Any]:
    """Narrow an Any JSON value to a dict, raising TapedReplayError otherwise."""
    if not isinstance(value, dict):
        msg = f"{what} must be an object"
        raise TapedReplayError(msg)
    return cast("dict[Any, Any]", value)


def _req_list(value: Any, what: str) -> list[Any]:
    """Narrow an Any JSON value to a list, raising TapedReplayError otherwise."""
    if not isinstance(value, list):
        msg = f"{what} must be an array"
        raise TapedReplayError(msg)
    return cast("list[Any]", value)


def _ref_from_dict(data: Any) -> RefSpec | None:
    if data is None:
        return None
    mapping = _req_dict(data, "manifest RefSpec")
    kind = mapping.get("kind")
    if kind not in ("input", "node_output", "bound"):
        msg = f"manifest RefSpec has invalid kind {kind!r}"
        raise TapedReplayError(msg)
    return RefSpec(
        kind=kind,
        name=mapping.get("name"),
        node=mapping.get("node"),
        field=mapping.get("field"),
    )


def _expr_from_dict(data: Any) -> ExprSpec | None:
    if data is None:
        return None
    mapping = _req_dict(data, "manifest ExprSpec")
    kind = mapping.get("kind")
    if kind not in ("not", "method_call", "ref", "literal", "and", "or", "compare", "binop"):
        msg = f"manifest ExprSpec has invalid kind {kind!r}"
        raise TapedReplayError(msg)
    raw_args = _req_list(mapping.get("args", []), "manifest ExprSpec args")
    raw_exprs = _req_list(mapping.get("exprs", []), "manifest ExprSpec exprs")
    return ExprSpec(
        kind=kind,
        expr=_expr_from_dict(mapping.get("expr")),
        receiver=_expr_from_dict(mapping.get("receiver")),
        method=mapping.get("method"),
        args=tuple(e for e in (_expr_from_dict(a) for a in raw_args) if e is not None),
        exprs=tuple(e for e in (_expr_from_dict(e) for e in raw_exprs) if e is not None),
        ref=_ref_from_dict(mapping.get("ref")),
        type=mapping.get("type"),
        value=mapping.get("value"),
        op=mapping.get("op"),
        left=_expr_from_dict(mapping.get("left")),
        right=_expr_from_dict(mapping.get("right")),
    )


def _guard_from_dict(data: Any) -> GuardSpec:
    mapping = _req_dict(data, "manifest GuardSpec")
    kind = mapping.get("kind")
    if kind not in ("always", "has_value", "missing", "eq", "if"):
        msg = f"manifest GuardSpec has invalid kind {kind!r}"
        raise TapedReplayError(msg)
    return GuardSpec(
        kind=kind,
        ref=_ref_from_dict(mapping.get("ref")),
        left=_expr_from_dict(mapping.get("left")),
        right=_expr_from_dict(mapping.get("right")),
        cond=_expr_from_dict(mapping.get("cond")),
    )


def _manifest_from_dict(data: Any) -> WorkflowManifest:
    """Rebuild a :class:`WorkflowManifest` from a vault snapshot dict."""
    root = _req_dict(data, "vault manifest snapshot")
    manifest_data = root.get("manifest", root)
    manifest_map = _req_dict(manifest_data, "vault manifest snapshot")
    raw_stages = _req_list(manifest_map.get("stages"), "vault manifest stages")
    raw_policies = _req_list(
        manifest_map.get("policies") or [], "vault manifest policies"
    )
    raw_inputs = _req_list(manifest_map.get("inputs") or [], "vault manifest inputs")
    raw_exits = _req_list(manifest_map.get("exit_stage_ids") or [], "vault manifest exits")
    raw_capabilities = _req_list(
        manifest_map.get("capabilities") or [], "vault manifest capabilities"
    )
    stages: list[StageSpec] = []
    for raw_stage in raw_stages:
        stage = _req_dict(raw_stage, "vault manifest stage")
        execution_map = _req_dict(
            stage.get("execution") or {}, "vault manifest execution"
        )
        exec_args_map = _req_dict(
            execution_map.get("args") or {}, "vault manifest execution args"
        )
        exec_args: dict[str, ExprSpec] = {}
        for arg_name, arg_expr in exec_args_map.items():
            parsed = _expr_from_dict(arg_expr)
            if parsed is not None:
                exec_args[str(arg_name)] = parsed
        raw_writes = _req_list(stage.get("writes") or [], "vault manifest writes")
        raw_transitions = _req_list(
            stage.get("transitions") or [], "vault manifest transitions"
        )
        raw_reads = _req_list(stage.get("reads") or [], "vault manifest reads")
        raw_requires = _req_list(stage.get("requires") or [], "vault manifest requires")
        transitions: list[TransitionSpec] = []
        for raw_transition in raw_transitions:
            transition_map = _req_dict(raw_transition, "vault manifest transition")
            transitions.append(
                TransitionSpec(
                    to=str(transition_map.get("to")),
                    priority=int(transition_map.get("priority", 0) or 0),
                    reason=str(transition_map.get("reason", "other")),
                    guard=_guard_from_dict(transition_map.get("guard")),
                )
            )
        execution_kind = execution_map.get("kind", "model")
        stages.append(
            StageSpec(
                id=str(stage.get("id")),
                prompt=str(stage.get("prompt", "") or ""),
                reads=tuple(_read_from_dict(r) for r in raw_reads),
                writes=tuple(
                    WriteSpec(
                        name=str(item.get("name")),
                        type=str(item.get("type")),
                        optional=bool(item.get("optional", False)),
                    )
                    for item in (_req_dict(w, "vault manifest write") for w in raw_writes)
                ),
                requires=frozenset(str(c) for c in raw_requires),
                transitions=tuple(transitions),
                execution=StageExecutionSpec(
                    kind=execution_kind if execution_kind in ("model", "tool") else "model",
                    capability=execution_map.get("capability"),
                    args=exec_args,
                ),
            )
        )
    policies: list[PolicySpec] = []
    for raw_policy in raw_policies:
        policy_map = _req_dict(raw_policy, "vault manifest policy")
        trigger_map = _req_dict(policy_map.get("trigger"), "vault manifest trigger")
        bind_map = _req_dict(trigger_map.get("bind") or {}, "vault manifest bind")
        raw_req_list = _req_list(policy_map.get("requires") or [], "vault manifest requires")
        requires: list[RequiredCapabilitySpec] = []
        for raw_req in raw_req_list:
            req_map = _req_dict(raw_req, "vault manifest requirement")
            req_args_map = _req_dict(req_map.get("args") or {}, "vault manifest req args")
            req_args: dict[str, RefSpec] = {}
            for req_name, req_ref in req_args_map.items():
                parsed_ref = _ref_from_dict(req_ref)
                if parsed_ref is not None:
                    req_args[str(req_name)] = parsed_ref
            requires.append(
                RequiredCapabilitySpec(
                    capability=str(req_map.get("capability")), args=req_args
                )
            )
        policy_kind = policy_map.get("kind", "deny")
        policies.append(
            PolicySpec(
                id=str(policy_map.get("id")),
                kind=policy_kind if policy_kind in ("before", "deny") else "deny",
                trigger=TriggerSpec(
                    capability=str(trigger_map.get("capability")),
                    bind={str(k): str(v) for k, v in bind_map.items()},
                ),
                requires=tuple(requires),
                condition=_expr_from_dict(policy_map.get("condition")),
            )
        )
    workflow_id = manifest_map.get("workflow_id")
    entry_stage_id = manifest_map.get("entry_stage_id")
    if not isinstance(workflow_id, str) or not isinstance(entry_stage_id, str):
        msg = "vault manifest snapshot has invalid workflow/entry ids"
        raise TapedReplayError(msg)
    return WorkflowManifest(
        workflow_id=workflow_id,
        entry_stage_id=entry_stage_id,
        exit_stage_ids=frozenset(str(e) for e in raw_exits),
        inputs=tuple(
            InputSpec(name=str(item.get("name")), type=str(item.get("type")))
            for item in (_req_dict(i, "vault manifest input") for i in raw_inputs)
        ),
        capabilities=frozenset(str(c) for c in raw_capabilities),
        policies=tuple(policies),
        stages=tuple(stages),
    )


def _read_from_dict(data: Any) -> Any:
    mapping = _req_dict(data, "vault manifest read")
    ref = _ref_from_dict(mapping.get("ref"))
    if ref is None:
        msg = "vault manifest read requires a ref"
        raise TapedReplayError(msg)
    return ReadSpec(ref=ref, optional=bool(mapping.get("optional", False)))



# ---------------------------------------------------------------------------
# Taped fixtures (no live effects by construction)
# ---------------------------------------------------------------------------

_WRITE_TYPE_MAP: dict[str, type] = {
    "string": str,
    "bool": bool,
    "number": float,
    "path": str,
    "string[]": list,
    "json": dict,
}


def _is_redaction_marker(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    marker = cast("dict[Any, Any]", value)
    return isinstance(marker.get("$redacted"), dict)


def _marker_aware_equal(replayed: Any, recorded: Any) -> bool:
    """Compare replayed values against possibly-markered vault snapshots.

    Redaction markers on the recorded side mean "could not verify": they
    never fail the comparison (the ledger already proves structure).
    """
    if _is_redaction_marker(recorded):
        return True
    if isinstance(recorded, dict):
        if not isinstance(replayed, dict):
            return False
        recorded_map = cast("dict[Any, Any]", recorded)
        replayed_map = cast("dict[Any, Any]", replayed)
        for key, expected in recorded_map.items():
            if key not in replayed_map or not _marker_aware_equal(
                replayed_map[key], expected
            ):
                return False
        return True
    if isinstance(recorded, list):
        if not isinstance(replayed, list):
            return False
        recorded_seq = cast("list[Any]", recorded)
        replayed_seq = cast("list[Any]", replayed)
        return len(replayed_seq) == len(recorded_seq) and all(
            _marker_aware_equal(a, b) for a, b in zip(replayed_seq, recorded_seq, strict=False)
        )
    return replayed == recorded


def _fixture_placeholder(marker: dict[Any, Any]) -> Any:
    """Type-correct stand-in for one redaction marker.

    The marker's ``value_type``/``length`` describe the removed value, so a
    placeholder of the same JSON type satisfies schema validation at a
    redacted position. Placeholders are never trusted: ledger comparison
    (``_marker_aware_equal``) skips every markered position, and any
    placeholder-derived value that lands in an unmarked position surfaces
    as an explicit divergence — never a silent match.
    """
    inner_any = marker.get("$redacted")
    if isinstance(inner_any, dict):
        inner = cast("dict[Any, Any]", inner_any)
    else:
        inner = cast("dict[Any, Any]", {})
    value_type = inner.get("value_type")
    length = inner.get("length")
    count = length if isinstance(length, int) and length >= 0 else 0
    if value_type == "string":
        return "?" * count
    if value_type == "number":
        return 0
    if value_type == "boolean":
        return False
    if value_type == "array":
        return []
    if value_type == "object":
        return {}
    return None


def _unmark_fixture(value: Any) -> Any:
    """Recursively replace redaction markers with typed placeholders."""
    if _is_redaction_marker(value):
        return _fixture_placeholder(cast("dict[Any, Any]", value))
    if isinstance(value, dict):
        pairs = cast("dict[Any, Any]", value)
        return {key: _unmark_fixture(item) for key, item in pairs.items()}
    if isinstance(value, list):
        items = cast("list[Any]", value)
        return [_unmark_fixture(item) for item in items]
    if isinstance(value, tuple):
        entries = cast("tuple[Any, ...]", value)
        return [_unmark_fixture(item) for item in entries]
    return value


def _infer_stub_type(value: Any) -> str:
    """A ``_WRITE_TYPE_MAP`` key accepting a recorded argument value.

    Recorded args passed live validation, so their JSON shapes are exactly
    what replay must accept. Extra known names are inert (stubs have no
    required params; normalization only checks present keys), so one global
    union across all fixtures is sound. Shapes live validation rejects
    (dict/None) fall back to ``"string"`` and fail honestly if ever served.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        items = cast("list[Any]", value)
        if all(isinstance(item, str) for item in items):
            return "string[]"
    return "string"


def _args_dicts(payload: dict[Any, Any], record_type: Any) -> list[dict[Any, Any]]:
    """Recorded argument objects for one vault record (tool or model)."""
    found: list[dict[Any, Any]] = []
    if record_type == "tool_result":
        args = payload.get("args")
        if isinstance(args, dict):
            found.append(cast("dict[Any, Any]", args))
    elif record_type == "model_response":
        raw_calls = payload.get("tool_calls")
        if isinstance(raw_calls, list):
            for raw_call in cast("list[Any]", raw_calls):
                if isinstance(raw_call, dict):
                    call_args = cast("dict[Any, Any]", raw_call).get("arguments")
                    if isinstance(call_args, dict):
                        found.append(cast("dict[Any, Any]", call_args))
    return found


def _recorded_arg_shapes(
    records: list[dict[str, Any]],
) -> dict[str, str]:
    """Global arg-name → stub-type union over tool/model fixtures."""
    shapes: dict[str, str] = {}
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_map = cast("dict[Any, Any]", payload)
        arg_sets = _args_dicts(payload_map, record.get("record_type"))
        for args in arg_sets:
            for key, value in args.items():
                if isinstance(key, str) and key not in shapes:
                    shapes[key] = _infer_stub_type(value)
    return shapes


class TapedModelAdapter:
    """Serves recorded model responses per stage; never calls a provider."""

    def __init__(self, responses_by_stage: Mapping[str, list[dict[str, Any]]]) -> None:
        self._queues: dict[str, list[dict[str, Any]]] = {
            stage: list(items) for stage, items in responses_by_stage.items()
        }
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        raw_queue = self._queues.get(request.stage_id)
        if not raw_queue:
            msg = (
                f"taped replay has no model fixture for stage "
                f"'{request.stage_id}' (call {self.calls})"
            )
            raise TapedReplayError(msg)
        queue = raw_queue
        fixture = queue.pop(0)
        tool_calls_raw = _unmark_fixture(fixture.get("tool_calls"))
        if not isinstance(tool_calls_raw, list):
            calls_list = cast("list[Any]", [])
        else:
            calls_list = cast("list[Any]", tool_calls_raw)
        tool_calls = tuple(
            ModelToolCall(
                id=str(call_map.get("id", "")),
                name=str(call_map.get("name", "")),
                arguments=dict(call_map.get("arguments", {})),
            )
            for call_map in (
                cast("dict[Any, Any]", raw_call)
                for raw_call in calls_list
                if isinstance(raw_call, dict)
            )
        )
        return ModelResponse(
            content=_unmark_fixture(fixture.get("content")),
            tool_calls=tool_calls,
            reasoning=None
            if _is_redaction_marker(fixture.get("reasoning"))
            else fixture.get("reasoning"),
        )


class TapedToolRegistry(ToolRegistry):
    """Serves recorded tool results in capture order; never executes.

    Catalog validation is disabled: taped stubs cover exactly the recorded
    capabilities (including synthetic ones), and fixtures — not handlers —
    determine outcomes. ``tool_name`` is ignored for dispatch (stubs are
    per-capability) but recorded for divergence reporting by the caller.
    """

    def __init__(
        self,
        fixtures: list[dict[str, Any]],
        *,
        stub_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._fixtures: list[dict[str, Any]] = list(fixtures)
        self.calls: list[dict[str, Any]] = []
        schemas = dict(stub_schemas or {})

        async def _taped_handler(_ctx: ToolContext, **_kwargs: Any) -> Any:
            msg = "taped tools serve fixtures through TapedToolRegistry.call"
            raise TapedReplayError(msg)

        capabilities: dict[str, None] = {}
        for fixture in self._fixtures:
            capability = fixture.get("capability")
            if isinstance(capability, str):
                capabilities.setdefault(capability)
        # Every declared capability gets a stub (fixtures may be absent for
        # denied/unreached calls, but the runtime still requires them).
        for declared in schemas:
            capabilities.setdefault(declared)
        stubs: list[Tool] = []
        for capability in capabilities:
            schema = schemas.get(capability, {})
            input_schema = {
                name: _WRITE_TYPE_MAP.get(str(kind), str)
                for name, kind in schema.get("inputs", {}).items()
            }
            output_schema: dict[str, type] | None = {
                name: _WRITE_TYPE_MAP.get(str(kind), str)
                for name, kind in schema.get("outputs", {}).items()
            } or None
            stubs.append(
                Tool(
                    name=f"taped-{capability}",
                    capability=capability,
                    description="Taped replay stub (no live effects).",
                    input_schema=input_schema,
                    handler=_taped_handler,
                    output_schema=output_schema,
                )
            )
        super().__init__(stubs)
        # Recorded tool names resolve to their capability stub: the runtime
        # looks tools up by the name the model requested, but taped stubs
        # are per-capability ("taped-<capability>"). Without this, every
        # model-requested call fails as "unknown tool" before dispatch.
        # First-seen mapping wins; unrecorded names fall through to None
        # (honest unknown-tool divergence, same shape as live).
        by_capability = {stub.capability: stub for stub in stubs}
        self._tools_by_recorded_name: dict[str, Tool] = {}
        for fixture in self._fixtures:
            tool_name = fixture.get("tool_name")
            capability = fixture.get("capability")
            if not isinstance(tool_name, str):
                continue
            if tool_name in self._tools_by_recorded_name:
                continue
            stub = by_capability.get(capability) if isinstance(capability, str) else None
            if stub is not None:
                self._tools_by_recorded_name[tool_name] = stub

    def get_by_name(self, name: str) -> Tool | None:
        stub = self._tools_by_recorded_name.get(name)
        if stub is not None:
            return stub
        return super().get_by_name(name)

    @staticmethod
    def _validate_tools(tools: list[Tool]) -> None:  # noqa: ARG004 - override must keep parent signature
        # Taped stubs are per-recorded-capability fixtures, not catalog tools.
        return None

    async def call(
        self,
        capability: str,
        args: Mapping[str, Any],
        ctx: ToolContext,
        *,
        tool_name: str | None = None,
    ) -> Any:
        if not self._fixtures:
            msg = (
                f"taped replay has no tool fixture for capability "
                f"'{capability}' at stage '{ctx.stage_id}'"
            )
            raise TapedReplayError(msg)
        fixture = self._fixtures.pop(0)
        self.calls.append(
            {
                "capability": capability,
                "tool_name": tool_name,
                "args": dict(args),
                "fixture_capability": fixture.get("capability"),
                "fixture_tool_name": fixture.get("tool_name"),
            }
        )
        payload = fixture.get("payload", fixture)
        if not isinstance(payload, dict):
            msg = "taped tool fixture payload must be an object"
            raise TapedReplayError(msg)
        payload_map = cast("dict[Any, Any]", payload)
        if "error" in payload_map:
            detail = payload_map["error"]
            if not isinstance(detail, dict):
                code = "tool_failed"
            else:
                code = cast("dict[Any, Any]", detail).get("code", "tool_failed")
            msg = f"taped replay reproduces recorded tool failure ({code})"
            raise ToolInvocationError(msg)
        result = payload_map.get("result")
        if _is_redaction_marker(result):
            msg = (
                f"taped replay cannot serve a redacted fixture for "
                f"capability '{capability}'"
            )
            raise TapedReplayError(msg)
        # Nested markers become typed placeholders: the live run already
        # proved validity at those positions, so validation must not fail
        # on the stand-in. Comparison still skips markered positions.
        return _unmark_fixture(result)


# ---------------------------------------------------------------------------
# Ledger comparison
# ---------------------------------------------------------------------------

_COMPARE_KINDS = frozenset(
    {
        "run_started",
        "stage_started",
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
    }
)


def _live_key(event: WorkflowEvent) -> tuple[Any, ...]:
    kind = event.kind
    if kind == "run_started":
        return (kind,)
    if kind in ("run_completed", "run_failed"):
        return (kind,)
    base: tuple[Any, ...] = (kind, event.stage_id)
    if kind == "transition_selected":
        return (*base, event.transition_to)
    if kind in ("tool_call_started", "tool_call_completed", "tool_call_failed"):
        return (*base, event.capability)
    if kind == "policy_checked":
        metadata_map = dict(event.metadata)
        return (
            *base,
            event.capability,
            metadata_map.get("policy_kind"),
            metadata_map.get("denied"),
        )
    if kind == "policy_denied":
        return (*base, event.capability)
    return base


def _ledger_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    kind = record.get("kind")
    if kind == "run_started":
        return (kind,)
    if kind in ("run_completed", "run_failed"):
        return (kind,)
    base: tuple[Any, ...] = (kind, record.get("stage_id"))
    if kind == "transition_selected":
        return (*base, record.get("transition_to"))
    if kind in ("tool_call_started", "tool_call_completed", "tool_call_failed"):
        return (*base, record.get("capability"))
    if kind == "policy_checked":
        raw_metadata = record.get("metadata")
        if not isinstance(raw_metadata, dict):
            return (*base, record.get("capability"), None, None)
        metadata_map = cast("dict[Any, Any]", raw_metadata)
        return (
            *base,
            record.get("capability"),
            metadata_map.get("policy_kind"),
            metadata_map.get("denied"),
        )
    if kind == "policy_denied":
        return (*base, record.get("capability"))
    return base


# ---------------------------------------------------------------------------
# Replay driver
# ---------------------------------------------------------------------------


@dataclass
class _ReplayCollector:
    events: list[WorkflowEvent] = field(
        default_factory=lambda: cast("list[WorkflowEvent]", [])
    )

    async def __call__(self, event: WorkflowEvent) -> None:
        self.events.append(event)


async def replay_trace(
    archive: str | Path,
    passphrase: str | bytes,
    *,
    options: RunOptions | None = None,
    event_sink: Any = None,
) -> ReplayReport:
    """Taped-replay an unlocked trace; compare the path against the ledger.

    Returns a :class:`ReplayReport`. ``matched`` is True only when the
    re-executed control path (stage order, transitions, tool/policy/model
    milestones, stage outputs, terminal status) agrees with the recorded
    archive. Any fixture miss, live-effect attempt, or path divergence
    yields ``matched=False`` with explicit divergences — never a silent
    fallback to live providers or tools.
    """
    records, unlock_report = unlock_archive(archive, passphrase)
    if not unlock_report.ok or unlock_report.replayability != "taped-replay":
        return ReplayReport(
            matched=False,
            divergences=tuple(unlock_report.errors) or ("archive verification failed",),
            verification=unlock_report,
        )
    try:
        manifest_record = next(
            r for r in records if r.get("record_type") == "full_workflow_ir"
        )
    except StopIteration:
        return ReplayReport(
            matched=False,
            divergences=("vault has no full_workflow_ir manifest snapshot",),
            verification=unlock_report,
        )
    try:
        manifest = _manifest_from_dict(manifest_record.get("payload"))
    except (TapedReplayError, ValueError, TypeError, AttributeError) as exc:
        return ReplayReport(
            matched=False,
            divergences=(f"vault manifest snapshot unusable: {exc}",),
            verification=unlock_report,
        )
    try:
        entries = read_archive_entries(archive)
        recorded_manifest = parse_json_strict(entries["manifest.json"].decode("utf-8"))
        if isinstance(recorded_manifest, dict):
            manifest_map = cast("dict[Any, Any]", recorded_manifest)
            manifest_status = str(manifest_map.get("status", "complete"))
        else:
            manifest_status = "complete"
    except Exception as exc:
        return ReplayReport(
            matched=False,
            divergences=(f"cannot read recorded terminal status: {exc}",),
            verification=unlock_report,
        )
    if manifest_status == "interrupted":
        return ReplayReport(
            matched=False,
            divergences=("interrupted traces are evidence, not replayable runs",),
            verification=unlock_report,
        )
    try:
        run_inputs_record = next(
            r for r in records if r.get("record_type") == "run_inputs"
        )
    except StopIteration:
        return ReplayReport(
            matched=False,
            divergences=("vault has no run_inputs fixture",),
            verification=unlock_report,
        )
    replay_inputs = run_inputs_record.get("payload", {})
    if not isinstance(replay_inputs, dict):
        return ReplayReport(
            matched=False,
            divergences=("vault run_inputs payload must be an object",),
            verification=unlock_report,
        )
    # Partition fixtures. Visit ids map to stages via snapshot payloads;
    # unknown visits route to "" and fail closed on first use.
    visit_stage: dict[Any, str] = {}
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_map = cast("dict[Any, Any]", payload)
        if not isinstance(payload_map.get("stage_id"), str):
            continue
        visit_stage.setdefault(record.get("stage_visit_id"), str(payload_map["stage_id"]))
    model_by_stage: dict[str, list[dict[str, Any]]] = {}
    tool_fixtures: list[dict[str, Any]] = []
    for record in records:
        rtype = record.get("record_type")
        record_payload = record.get("payload")
        if not isinstance(record_payload, dict):
            continue
        payload_dict = cast("dict[Any, Any]", record_payload)
        if rtype == "model_response":
            stage_id = visit_stage.get(record.get("stage_visit_id"), "")
            model_by_stage.setdefault(stage_id, []).append(payload_dict)
        elif rtype == "tool_result":
            tool_fixtures.append(
                {
                    "capability": payload_dict.get("capability"),
                    "tool_name": payload_dict.get("tool_name"),
                    "payload": payload_dict,
                    "tool_call_id": record.get("tool_call_id"),
                }
            )
    # Stub schemas from the manifest: one stub per recorded capability.
    stub_schemas: dict[str, dict[str, Any]] = {}
    for capability in manifest.capabilities:
        stub_schemas.setdefault(capability, {"inputs": {}, "outputs": {}})
    for stage in manifest.stages:
        for capability in stage.requires:
            schema = stub_schemas.setdefault(capability, {"inputs": {}, "outputs": {}})
            if stage.execution.kind == "tool" and stage.execution.capability == capability:
                for arg_name in (stage.execution.args or {}):
                    schema["inputs"].setdefault(arg_name, "json")
            for write in stage.writes:
                schema["outputs"].setdefault(write.name, write.type)
    # Evidence-based input names: model-requested tool calls normalize
    # against these stub schemas, so every recorded argument name must be
    # known or replay rejects real calls as "unknown argument".
    for arg_name, arg_kind in _recorded_arg_shapes(records).items():
        for schema in stub_schemas.values():
            schema["inputs"].setdefault(arg_name, arg_kind)
    taped_tools = TapedToolRegistry(tool_fixtures, stub_schemas=stub_schemas)
    taped_model = TapedModelAdapter(model_by_stage)
    # Taped deny-policy outcomes: the runtime reproduces recorded allow/deny
    # instead of re-evaluating expressions against scrubbed fixture args.
    # A dedicated recorder carries the tape; it never finalizes an archive.
    tape_recorder = NoOpTraceRecorder()
    tape_recorder.policy_tape = _policy_tape_for_replay(records, manifest)
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=taped_tools,
        stage_executor=ModelStageExecutor(model=taped_model, tools=taped_tools),
    )
    collector = _ReplayCollector()
    sinks: list[Any] = [collector]
    if event_sink is not None:
        sinks.append(event_sink)

    async def _fanout(event: WorkflowEvent) -> None:
        for sink in sinks:
            await sink(event)

    replayed_status = "complete"
    replay_error: str | None = None
    try:
        replay_args = dict(cast("dict[Any, Any]", replay_inputs))
        await runtime.run(
            replay_args, options=options, event_sink=_fanout, trace_recorder=tape_recorder
        )
    except TapedReplayError as exc:
        replayed_status = "failed"
        replay_error = f"taped fixture error: {exc}"
    except Exception as exc:
        replayed_status = "failed"
        replay_error = f"replayed run raised {type(exc).__name__}"
    divergences = _compare_paths(archive, collector.events, records)
    if manifest_status == "failed" and replayed_status != "failed":
        divergences.append("recorded run failed but replay completed")
    elif manifest_status == "complete" and replayed_status != "complete":
        divergences.append(
            replay_error or "recorded run completed but replay did not finish"
        )
    steps = sum(1 for e in collector.events if e.kind == "stage_completed")
    return ReplayReport(
        matched=not divergences,
        divergences=tuple(divergences),
        verification=unlock_report,
        steps=steps,
        replayed_status=replayed_status,
    )


def _parse_json_line(line: bytes) -> Any:
    """Parse one NDJSON line, returning None when it is not valid JSON."""
    try:
        return parse_json_strict(line.decode("utf-8"))
    except Exception:
        return None


def _compare_paths(
    archive: str | Path,
    replayed: list[WorkflowEvent],
    records: list[dict[str, Any]],
) -> list[str]:
    """Compare the replayed path against the recorded public ledger."""
    divergences: list[str] = []
    try:
        entries = read_archive_entries(archive)
    except Exception as exc:
        return [f"cannot re-read archive for comparison: {exc}"]
    ledger: list[dict[str, Any]] = []
    for line in entries["public/events.ndjson"].split(b"\n"):
        if not line.strip():
            continue
        event = _parse_json_line(line)
        if not isinstance(event, dict):
            continue
        event_map = cast("dict[Any, Any]", event)
        if event_map.get("kind") in _COMPARE_KINDS:
            ledger.append(event_map)
    replayed_keys = [_live_key(e) for e in replayed if e.kind in _COMPARE_KINDS]
    ledger_keys = [_ledger_key(e) for e in ledger]
    if len(replayed_keys) != len(ledger_keys):
        divergences.append(
            f"path length differs: replayed {len(replayed_keys)} milestone events, "
            f"ledger has {len(ledger_keys)}"
        )
    for index, (got, want) in enumerate(zip(replayed_keys, ledger_keys, strict=False)):
        if got != want:
            divergences.append(
                f"divergence at milestone {index}: replayed {got!r} != recorded {want!r}"
            )
            break
    # Stage-output values against vault snapshots (marker-aware).
    snapshots: dict[Any, Any] = {}
    for record in records:
        if record.get("record_type") == "stage_snapshot":
            snapshots[record.get("stage_visit_id")] = record.get("payload", {})
    visit_to_output: dict[Any, Any] = {}
    tracker = _VisitTracker()
    current_visit = ""
    for event in replayed:
        if event.kind == "stage_started":
            current_visit = tracker.next()
        if event.kind == "stage_completed":
            visit_to_output[current_visit] = event.output or {}
    for visit, snapshot in snapshots.items():
        if not isinstance(snapshot, dict):
            continue
        snapshot_map = cast("dict[Any, Any]", snapshot)
        expected = snapshot_map.get("output", {})
        actual = visit_to_output.get(visit)
        if actual is None:
            divergences.append(f"replay never reached recorded visit {visit}")
        elif not _marker_aware_equal(actual, expected):
            divergences.append(f"stage output differs at visit {visit}")
    return divergences


class _VisitTracker:
    """Maps replayed ``stage_started`` order to run-local visit ids.

    The replayed run regenerates visit ids (``s-N``) in the same order as
    the recorded run when the path matches, so the Nth ``stage_started``
    maps to the Nth visit. Any path divergence is reported separately by
    milestone comparison.
    """

    def __init__(self) -> None:
        self.count = 0

    def next(self) -> str:
        self.count += 1
        return f"s-{self.count}"
