from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from nemoir_runtime.capabilities import CAPABILITY_CATALOG
from nemoir_runtime.errors import (
    DataUnavailableError,
    MaxStepsExceededError,
    MissingCapabilityError,
    NoTransitionMatchedError,
    PolicyDeniedError,
    PolicyEvaluationError,
    StageOutputValidationError,
    WorkflowValidationError,
)
from nemoir_runtime.events import WorkflowEvent, WorkflowEventEmitter, WorkflowEventSink
from nemoir_runtime.tools import ToolContext, ToolRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

# ---------------------------------------------------------------------------
# Manifest dataclasses (mirror Rust IR shape)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputSpec:
    name: str
    type: str


@dataclass(frozen=True)
class RefSpec:
    kind: Literal["input", "node_output", "bound"]
    name: str | None = None
    node: str | None = None
    field: str | None = None


@dataclass(frozen=True)
class ExprSpec:
    kind: Literal["not", "method_call", "ref", "literal"]
    expr: ExprSpec | None = None
    receiver: ExprSpec | None = None
    method: str | None = None
    args: tuple[ExprSpec, ...] = ()
    ref: RefSpec | None = None
    type: str | None = None
    value: Any = None


@dataclass(frozen=True)
class GuardSpec:
    kind: Literal["always", "has_value", "missing", "eq"]
    ref: RefSpec | None = None
    left: ExprSpec | None = None
    right: ExprSpec | None = None


@dataclass(frozen=True)
class ReadSpec:
    ref: RefSpec
    optional: bool


@dataclass(frozen=True)
class WriteSpec:
    name: str
    type: str
    optional: bool


@dataclass(frozen=True)
class TransitionSpec:
    to: str
    priority: int
    reason: str
    guard: GuardSpec


@dataclass(frozen=True)
class StageSpec:
    id: str
    prompt: str
    reads: tuple[ReadSpec, ...]
    writes: tuple[WriteSpec, ...]
    requires: frozenset[str]
    transitions: tuple[TransitionSpec, ...]


@dataclass(frozen=True)
class TriggerSpec:
    capability: str
    bind: Mapping[str, str]


@dataclass(frozen=True)
class RequiredCapabilitySpec:
    capability: str
    args: Mapping[str, RefSpec]


@dataclass(frozen=True)
class PolicySpec:
    id: str
    kind: Literal["before", "deny"]
    trigger: TriggerSpec
    requires: tuple[RequiredCapabilitySpec, ...] = ()
    condition: ExprSpec | None = None


@dataclass(frozen=True)
class WorkflowManifest:
    workflow_id: str
    entry_stage_id: str
    exit_stage_ids: frozenset[str]
    inputs: tuple[InputSpec, ...]
    capabilities: frozenset[str]
    policies: tuple[PolicySpec, ...]
    stages: tuple[StageSpec, ...]


# ---------------------------------------------------------------------------
# Run options and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    max_steps: int = 64
    max_model_retries: int = 3
    # Accepted but not enforced in Phase 2. Only max_steps is enforced.
    timeout_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]
    reasoning: Literal["none", "raw"] = "none"


@dataclass(frozen=True)
class WorkflowState:
    current_stage_id: str
    stage_outputs: Mapping[str, Mapping[str, Any]]
    steps: int


@dataclass(frozen=True)
class WorkflowResult:
    output: Mapping[str, Any]
    state: WorkflowState


# ---------------------------------------------------------------------------
# Stage execution boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageContext:
    workflow_id: str
    stage: StageSpec
    inputs: Mapping[str, Any]
    readable_context: Mapping[str, Any]
    allowed_capabilities: frozenset[str]
    options: RunOptions
    call_tool: Callable[..., Awaitable[Any]]
    event_emitter: WorkflowEventEmitter | None = None


class StageExecutor(Protocol):
    async def execute(self, ctx: StageContext) -> Mapping[str, Any]: ...


# ---------------------------------------------------------------------------
# WorkflowRuntime
# ---------------------------------------------------------------------------


class WorkflowRuntime:
    def __init__(
        self,
        *,
        manifest: WorkflowManifest,
        tools: ToolRegistry,
        stage_executor: StageExecutor,
    ) -> None:
        self._manifest = manifest
        self._tools = tools
        self._stage_executor = stage_executor
        tools.require_capabilities(manifest.capabilities)

        self._stage_map: dict[str, StageSpec] = {}
        for s in manifest.stages:
            if s.id in self._stage_map:
                msg = f"Workflow '{manifest.workflow_id}' has duplicate stage id '{s.id}'"
                raise WorkflowValidationError(msg)
            self._stage_map[s.id] = s

        if manifest.entry_stage_id not in self._stage_map:
            msg = (
                f"Workflow '{manifest.workflow_id}': "
                f"entry stage '{manifest.entry_stage_id}' not found"
            )
            raise WorkflowValidationError(msg)

        for exit_id in manifest.exit_stage_ids:
            if exit_id not in self._stage_map:
                msg = f"Workflow '{manifest.workflow_id}': exit stage '{exit_id}' not found"
                raise WorkflowValidationError(msg)

        for stage in manifest.stages:
            for t in stage.transitions:
                if t.to not in self._stage_map:
                    msg = (
                        f"Workflow '{manifest.workflow_id}': stage '{stage.id}' "
                        f"has transition to unknown stage '{t.to}'"
                    )
                    raise WorkflowValidationError(msg)

        self._exit_ids: frozenset[str] = manifest.exit_stage_ids
        self._policies_by_trigger: dict[str, list[PolicySpec]] = {}
        for p in manifest.policies:
            self._policies_by_trigger.setdefault(p.trigger.capability, []).append(p)

    # ------------------------------------------------------------------
    # Public execution
    # ------------------------------------------------------------------

    async def run(
        self,
        inputs: Mapping[str, Any],
        *,
        options: RunOptions | None = None,
        event_sink: WorkflowEventSink | None = None,
    ) -> WorkflowResult:
        opts = options if options is not None else RunOptions()
        stage_outputs: dict[str, dict[str, Any]] = {}
        current_id = self._manifest.entry_stage_id
        steps = 0
        run_id = uuid.uuid4().hex
        emitter = WorkflowEventEmitter(run_id=run_id, sink=event_sink)

        await emitter.emit(
            "run_started",
            metadata={"workflow_id": self._manifest.workflow_id, "entry": current_id},
        )

        try:
            while True:
                if steps >= opts.max_steps:
                    msg = (
                        f"Workflow '{self._manifest.workflow_id}' "
                        f"exceeded max_steps={opts.max_steps}"
                    )
                    raise MaxStepsExceededError(msg)  # noqa: TRY301

                stage = self._require_stage(current_id)
                readable = self._resolve_reads(stage, inputs, stage_outputs)
                ctx = self._make_stage_context(stage, inputs, readable, opts, emitter)
                await emitter.emit("stage_started", stage_id=stage.id)
                raw_output = await self._stage_executor.execute(ctx)
                self._validate_output(stage, raw_output)
                normalized = self._normalize_optional_empty_arrays(stage, raw_output)
                stage_outputs[stage.id] = normalized
                steps += 1
                await emitter.emit(
                    "stage_completed",
                    stage_id=stage.id,
                    output=dict(normalized),
                )

                if stage.id in self._exit_ids:
                    result = WorkflowResult(
                        output=raw_output,
                        state=WorkflowState(
                            current_stage_id=stage.id,
                            stage_outputs=dict(stage_outputs),
                            steps=steps,
                        ),
                    )
                    await emitter.emit("run_completed", result=result)
                    return result

                selected = self._select_transition(stage, inputs, stage_outputs)
                await emitter.emit(
                    "transition_selected",
                    stage_id=stage.id,
                    transition_to=selected.to,
                    metadata={"reason": selected.reason, "priority": selected.priority},
                )
                current_id = selected.to
        except Exception as exc:
            await emitter.emit(
                "run_failed",
                error=str(exc),
                metadata={"reason": type(exc).__name__},
            )
            raise

    # ------------------------------------------------------------------
    # Read resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_reads(
        stage: StageSpec,
        inputs: Mapping[str, Any],
        stage_outputs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for read in stage.reads:
            key = _read_display_key(read.ref)
            value = _resolve_read_ref(read.ref, inputs, stage_outputs)
            if read.optional:
                result[key] = value
            else:
                if value is None:
                    msg = f"Stage '{stage.id}': required read '{key}' is not available"
                    raise DataUnavailableError(msg)
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_output(stage: StageSpec, output: Mapping[str, Any]) -> None:
        allowed_names = {w.name for w in stage.writes}
        for key in output:
            if key not in allowed_names:
                msg = f"Stage '{stage.id}' returned unknown output field '{key}'"
                raise StageOutputValidationError(msg)
        for write in stage.writes:
            if not write.optional and (write.name not in output or output[write.name] is None):
                msg = f"Stage '{stage.id}' is missing required output field '{write.name}'"
                raise StageOutputValidationError(msg)
            val = output.get(write.name)
            if val is not None:
                _validate_write_type(val, write.type, write.name, stage.id)

    @staticmethod
    def _normalize_optional_empty_arrays(
        stage: StageSpec,
        output: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Treat empty lists for optional array outputs as None.

        LLMs often emit ``[]`` for optional array outputs when they mean
        "no items".  The runtime treats ``[]`` as a truthy value, which
        causes ``has_value`` guards to fire when they shouldn't.  This
        normalizes empty optional arrays to ``None`` so that the
        ``has_value`` guard evaluates to ``False`` and the workflow
        proceeds correctly.
        """
        result = dict(output)
        for write in stage.writes:
            if not write.optional:
                continue
            if not write.type.endswith("[]"):
                continue
            val = result.get(write.name)
            if isinstance(val, list) and len(val) == 0:  # type: ignore[reportUnknownArgumentType]
                result[write.name] = None
        return result

    # ------------------------------------------------------------------
    # Transition evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _select_transition(
        stage: StageSpec,
        inputs: Mapping[str, Any],
        stage_outputs: Mapping[str, Mapping[str, Any]],
    ) -> TransitionSpec:
        sorted_transitions = sorted(stage.transitions, key=lambda t: t.priority)
        for trans in sorted_transitions:
            if WorkflowRuntime._evaluate_guard(trans.guard, inputs, stage_outputs):
                return trans
        msg = f"Stage '{stage.id}': no transition matched"
        raise NoTransitionMatchedError(msg)

    # ------------------------------------------------------------------
    # Guard evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_guard(
        guard: GuardSpec,
        inputs: Mapping[str, Any],
        stage_outputs: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        if guard.kind == "always":
            return True
        if guard.kind == "has_value":
            return (
                guard.ref is not None
                and _resolve_guard_ref(guard.ref, inputs, stage_outputs) is not None
            )
        if guard.kind == "missing":
            return guard.ref is None or _resolve_guard_ref(guard.ref, inputs, stage_outputs) is None
        if guard.kind == "eq":
            if guard.left is None or guard.right is None:
                return False
            left_val = _eval_guard_expr(guard.left, inputs, stage_outputs)
            right_val = _eval_guard_expr(guard.right, inputs, stage_outputs)
            return left_val == right_val
        return False

    # ------------------------------------------------------------------
    # Stage context factory
    # ------------------------------------------------------------------

    def _make_stage_context(
        self,
        stage: StageSpec,
        inputs: Mapping[str, Any],
        readable: Mapping[str, Any],
        options: RunOptions,
        emitter: WorkflowEventEmitter,
    ) -> StageContext:
        return StageContext(
            workflow_id=self._manifest.workflow_id,
            stage=stage,
            inputs=inputs,
            readable_context=readable,
            allowed_capabilities=stage.requires,
            options=options,
            call_tool=self._make_tool_caller(stage, inputs, options, emitter),
            event_emitter=emitter,
        )

    def _make_tool_caller(
        self,
        stage: StageSpec,
        inputs: Mapping[str, Any],
        run_opts: RunOptions,
        emitter: WorkflowEventEmitter,
    ) -> Callable[..., Awaitable[Any]]:
        async def call_tool(
            capability: str,
            args: Mapping[str, Any],
            *,
            tool_name: str | None = None,
        ) -> Any:
            return await self._enforce_and_call(
                stage, capability, args, inputs, run_opts, emitter, tool_name=tool_name
            )

        return call_tool

    # ------------------------------------------------------------------
    # Policy enforcement + tool call
    # ------------------------------------------------------------------

    async def _enforce_and_call(
        self,
        stage: StageSpec,
        capability: str,
        args: Mapping[str, Any],
        inputs: Mapping[str, Any],
        run_opts: RunOptions,
        emitter: WorkflowEventEmitter,
        *,
        tool_name: str | None = None,
    ) -> Any:
        if capability not in stage.requires:
            msg = f"capability '{capability}' is not available in stage '{stage.id}'"
            raise MissingCapabilityError(msg)
        return await self._enforce_and_call_with_policies(
            capability,
            args,
            inputs,
            stage,
            allow_before=True,
            run_opts=run_opts,
            emitter=emitter,
            tool_name=tool_name,
        )

    async def _enforce_policy_call(
        self,
        capability: str,
        args: Mapping[str, Any],
        inputs: Mapping[str, Any],
        stage: StageSpec,
        run_opts: RunOptions,
        emitter: WorkflowEventEmitter,
    ) -> Any:
        return await self._enforce_and_call_with_policies(
            capability,
            args,
            inputs,
            stage,
            allow_before=False,
            run_opts=run_opts,
            emitter=emitter,
        )

    async def _enforce_and_call_with_policies(  # noqa: C901, PLR0912
        self,
        capability: str,
        args: Mapping[str, Any],
        inputs: Mapping[str, Any],
        stage: StageSpec,
        *,
        allow_before: bool,
        run_opts: RunOptions,
        emitter: WorkflowEventEmitter,
        tool_name: str | None = None,
    ) -> Any:
        policies = self._policies_by_trigger.get(capability, [])

        for policy in policies:
            if policy.kind == "deny" and policy.condition is not None:
                bound_args = self._bind_trigger_args(
                    policy.trigger, args, policy_id=policy.id, capability=capability
                )
                try:
                    denied = _eval_policy_expr(
                        policy.condition,
                        inputs,
                        bound_args,
                        policy_id=policy.id,
                        capability=capability,
                    )
                except DataUnavailableError as e:
                    await emitter.emit(
                        "policy_checked",
                        stage_id=stage.id,
                        capability=capability,
                        metadata={
                            "policy_id": policy.id,
                            "policy_kind": "deny",
                            "denied": True,
                            "error": str(e),
                        },
                    )
                    msg = (
                        f"Policy '{policy.id}': condition evaluation failed "
                        f"for capability '{capability}': {e}"
                    )
                    raise PolicyEvaluationError(msg) from e
                await emitter.emit(
                    "policy_checked",
                    stage_id=stage.id,
                    capability=capability,
                    metadata={
                        "policy_id": policy.id,
                        "policy_kind": "deny",
                        "denied": denied,
                    },
                )
                if denied:
                    await emitter.emit(
                        "policy_denied",
                        stage_id=stage.id,
                        capability=capability,
                        error=f"Policy '{policy.id}' denied capability '{capability}'",
                        metadata={"policy_id": policy.id},
                    )
                    msg = f"Policy '{policy.id}' denied capability '{capability}'"
                    raise PolicyDeniedError(msg)

        for policy in policies:
            if policy.kind == "before" and allow_before:
                bound_args = self._bind_trigger_args(
                    policy.trigger, args, policy_id=policy.id, capability=capability
                )
                if emitter.has_sink:
                    required_caps = [req.capability for req in policy.requires]
                    await emitter.emit(
                        "policy_checked",
                        stage_id=stage.id,
                        capability=capability,
                        metadata={
                            "policy_id": policy.id,
                            "policy_kind": "before",
                            "required_capabilities": required_caps,
                        },
                    )
                for req in policy.requires:
                    req_args = self._resolve_required_args(
                        req, inputs, bound_args, policy_id=policy.id, capability=capability
                    )
                    if not req_args and req.capability == "user.confirm":
                        req_args = {"message": f"Allow policy-required call before {capability}?"}
                    self._check_required_args_present(
                        req.capability, req_args, policy_id=policy.id, capability=capability
                    )
                    result = await self._enforce_policy_call(
                        req.capability, req_args, inputs, stage, run_opts, emitter
                    )
                    if req.capability == "user.confirm" and result is False:
                        await emitter.emit(
                            "policy_denied",
                            stage_id=stage.id,
                            capability=capability,
                            error=f"user.confirm returned False for policy '{policy.id}'",
                            metadata={"policy_id": policy.id},
                        )
                        msg = (
                            f"Policy '{policy.id}': user.confirm returned False, "
                            f"blocking capability '{capability}'"
                        )
                        raise PolicyDeniedError(msg)

        # Emit tool_call_started before the handler runs.
        # Prefer the caller-provided tool_name; fall back to the first
        # registered tool for the capability.
        if tool_name is not None:
            resolved_name = tool_name
        else:
            tool_obj = self._tools.get(capability)
            resolved_name = tool_obj.name if tool_obj else capability
        await emitter.emit(
            "tool_call_started",
            stage_id=stage.id,
            capability=capability,
            tool_name=resolved_name,
            args=dict(args),
        )

        ctx = ToolContext(
            workflow_id=self._manifest.workflow_id,
            stage_id=stage.id,
            inputs=inputs,
            metadata=run_opts.metadata,
        )
        try:
            result = await self._tools.call(capability, args, ctx, tool_name=tool_name)
        except Exception:
            await emitter.emit(
                "tool_call_failed",
                stage_id=stage.id,
                capability=capability,
                tool_name=resolved_name,
                error=str(_active_exception()),
            )
            raise
        await emitter.emit(
            "tool_call_completed",
            stage_id=stage.id,
            capability=capability,
            tool_name=resolved_name,
            metadata={"result_preview": _safe_result_preview(result)},
        )
        return result

    @staticmethod
    def _bind_trigger_args(
        trigger: TriggerSpec,
        args: Mapping[str, Any],
        *,
        policy_id: str,
        capability: str,
    ) -> dict[str, Any]:
        bound: dict[str, Any] = {}
        for bound_var, arg_name in trigger.bind.items():
            if arg_name not in args:
                msg = (
                    f"Policy '{policy_id}': trigger-bound argument '{arg_name}' "
                    f"is missing from capability '{capability}' call args"
                )
                raise PolicyEvaluationError(msg)
            bound[bound_var] = args[arg_name]
        return bound

    @staticmethod
    def _resolve_required_args(
        req: RequiredCapabilitySpec,
        inputs: Mapping[str, Any],
        bound_args: Mapping[str, Any],
        *,
        policy_id: str,
        capability: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for arg_name, ref in req.args.items():
            result[arg_name] = _resolve_policy_ref(
                ref, inputs, bound_args, policy_id=policy_id, capability=capability
            )
        return result

    @staticmethod
    def _check_required_args_present(
        req_capability: str,
        req_args: Mapping[str, Any],
        *,
        policy_id: str,
        capability: str,
    ) -> None:
        spec = CAPABILITY_CATALOG.get(req_capability)
        if spec is None:
            msg = (
                f"Policy '{policy_id}': required capability '{req_capability}' "
                f"is not in the catalog"
            )
            raise PolicyEvaluationError(msg)
        for param in spec.required_params:
            if param.name not in req_args or req_args[param.name] is None:
                msg = (
                    f"Policy '{policy_id}': required capability '{req_capability}' "
                    f"is missing catalog-required argument '{param.name}' "
                    f"for original capability '{capability}'"
                )
                raise PolicyEvaluationError(msg)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        inputs: Mapping[str, Any],
        *,
        options: RunOptions | None = None,
    ) -> AsyncIterator[WorkflowEvent]:
        queue: asyncio.Queue[WorkflowEvent | _RunDone] = asyncio.Queue()

        async def sink(event: WorkflowEvent) -> None:
            await queue.put(event)

        async def run_task() -> None:
            try:
                await self.run(inputs, options=options, event_sink=sink)
            except BaseException as exc:
                await queue.put(_RunDone(error=exc))
            else:
                await queue.put(_RunDone(error=None))

        task = asyncio.create_task(run_task())
        try:
            while True:
                item = await queue.get()
                if isinstance(item, _RunDone):
                    if item.error is not None:
                        raise item.error  # noqa: TRY301
                    return
                yield item
        except (asyncio.CancelledError, GeneratorExit):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise
        except BaseException:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_stage(self, stage_id: str) -> StageSpec:
        stage = self._stage_map.get(stage_id)
        if stage is None:
            msg = f"Stage '{stage_id}' not found in manifest"
            raise WorkflowValidationError(msg)
        return stage


# ---------------------------------------------------------------------------
# Read resolution helpers
# ---------------------------------------------------------------------------


def _resolve_read_ref(
    ref: RefSpec,
    inputs: Mapping[str, Any],
    stage_outputs: Mapping[str, Mapping[str, Any]],
) -> Any:
    if ref.kind == "input":
        return inputs.get(ref.name) if ref.name is not None else None
    if ref.kind == "node_output":
        if ref.node is None or ref.field is None:
            return None
        node = stage_outputs.get(ref.node)
        if node is None:
            return None
        return node.get(ref.field)
    if ref.kind == "bound":
        msg = f"Read refs bound variable '{ref.name}' which is policy-local only"
        raise DataUnavailableError(msg)
    return None


def _resolve_guard_ref(
    ref: RefSpec,
    inputs: Mapping[str, Any],
    stage_outputs: Mapping[str, Mapping[str, Any]],
) -> Any:
    if ref.kind == "bound":
        msg = f"Guard refs bound variable '{ref.name}' which is policy-local only"
        raise DataUnavailableError(msg)
    return _resolve_read_ref(ref, inputs, stage_outputs)


def _resolve_policy_ref(
    ref: RefSpec,
    inputs: Mapping[str, Any],
    bound_args: Mapping[str, Any],
    *,
    policy_id: str,
    capability: str,
) -> Any:
    if ref.kind == "input":
        if ref.name is None or ref.name not in inputs:
            msg = (
                f"Policy '{policy_id}' for capability '{capability}': "
                f"input ref '{ref.name}' is not provided"
            )
            raise PolicyEvaluationError(msg)
        return inputs[ref.name]
    if ref.kind == "bound":
        if ref.name is None or ref.name not in bound_args:
            msg = (
                f"Policy '{policy_id}' for capability '{capability}': "
                f"bound ref '{ref.name}' could not be resolved"
            )
            raise PolicyEvaluationError(msg)
        return bound_args[ref.name]
    if ref.kind == "node_output":
        msg = (
            f"Policy '{policy_id}' for capability '{capability}': "
            f"refs node_output '{ref.node}.{ref.field}' which is not allowed"
        )
        raise PolicyEvaluationError(msg)
    return None


def _read_display_key(ref: RefSpec) -> str:
    if ref.kind == "input":
        return f"input.{ref.name}"
    if ref.kind == "node_output":
        return f"{ref.node}.{ref.field}"
    return f"unknown.{ref.name}"


# ---------------------------------------------------------------------------
# Write type validation
# ---------------------------------------------------------------------------


def _validate_write_type(value: Any, write_type: str, field_name: str, stage_id: str) -> None:
    if write_type == "string":
        if not isinstance(value, str):
            msg = (
                f"Stage '{stage_id}' output field '{field_name}': expected str, "
                f"got {type(value).__name__}"
            )
            raise StageOutputValidationError(msg)
    elif write_type == "bool":
        if not isinstance(value, bool):
            msg = (
                f"Stage '{stage_id}' output field '{field_name}': expected bool, "
                f"got {type(value).__name__}"
            )
            raise StageOutputValidationError(msg)
    elif write_type == "path":
        if not isinstance(value, Path):
            msg = (
                f"Stage '{stage_id}' output field '{field_name}': expected Path, "
                f"got {type(value).__name__}"
            )
            raise StageOutputValidationError(msg)
    elif write_type == "string[]":
        if not isinstance(value, list):
            msg = (
                f"Stage '{stage_id}' output field '{field_name}': expected list[str], "
                f"got {type(value).__name__}"
            )
            raise StageOutputValidationError(msg)
        if not all(isinstance(v, str) for v in value):  # type: ignore[reportUnknownVariableType]
            msg = (
                f"Stage '{stage_id}' output field '{field_name}': expected list[str] "
                f"but contains non-string elements"
            )
            raise StageOutputValidationError(msg)
    else:
        msg = f"Stage '{stage_id}' output field '{field_name}': unsupported type '{write_type}'"
        raise StageOutputValidationError(msg)


# ---------------------------------------------------------------------------
# Expression evaluation helpers
# ---------------------------------------------------------------------------


def _eval_guard_expr(
    expr: ExprSpec,
    inputs: Mapping[str, Any],
    stage_outputs: Mapping[str, Mapping[str, Any]],
) -> Any:
    return _eval_expr_impl(
        expr,
        lambda r: _resolve_guard_ref(r, inputs, stage_outputs),
        lambda e: _eval_guard_expr(e, inputs, stage_outputs),
    )


def _eval_policy_expr(
    expr: ExprSpec,
    inputs: Mapping[str, Any],
    bound_args: Mapping[str, Any],
    *,
    policy_id: str,
    capability: str,
) -> Any:
    return _eval_expr_impl(
        expr,
        lambda r: _resolve_policy_ref(
            r, inputs, bound_args, policy_id=policy_id, capability=capability
        ),
        lambda e: _eval_policy_expr(
            e, inputs, bound_args, policy_id=policy_id, capability=capability
        ),
    )


def _eval_expr_impl(
    expr: ExprSpec,
    resolve: Callable[[RefSpec], Any],
    recurse: Callable[[ExprSpec], Any],
) -> Any:
    if expr.kind == "ref":
        if expr.ref is None:
            msg = "Ref expression has no ref"
            raise DataUnavailableError(msg)
        return resolve(expr.ref)
    if expr.kind == "literal":
        return expr.value
    if expr.kind == "not":
        if expr.expr is None:
            msg = "Not expression has no inner expr"
            raise DataUnavailableError(msg)
        inner = recurse(expr.expr)
        return not inner
    if expr.kind == "method_call":
        if expr.receiver is None:
            msg = "Method call has no receiver"
            raise DataUnavailableError(msg)
        receiver_val = recurse(expr.receiver)
        arg_vals = [recurse(a) for a in expr.args]
        return _eval_method_call(expr.method, receiver_val, arg_vals)
    msg = f"Unknown expression kind '{expr.kind}'"
    raise DataUnavailableError(msg)


def _eval_method_call(
    method: str | None,
    receiver_val: Any,
    arg_vals: list[Any],
) -> Any:
    if method == "contains":
        if isinstance(receiver_val, str):
            receiver_val = Path(receiver_val)
        if not isinstance(receiver_val, Path):
            msg = f"contains() receiver must be Path, got {type(receiver_val).__name__}"
            raise DataUnavailableError(msg)
        if len(arg_vals) < 1:
            msg = "contains() requires at least 1 argument"
            raise DataUnavailableError(msg)
        target = arg_vals[0]
        if isinstance(target, str):
            target = Path(target)
        if isinstance(target, Path):
            try:
                cwd_resolved = receiver_val.resolve(strict=False)
                target_resolved = (
                    receiver_val / target if not target.is_absolute() else target
                ).resolve(strict=False)
            except (OSError, ValueError):
                return False
            else:
                return target_resolved == cwd_resolved or cwd_resolved in target_resolved.parents
        return False
    msg = f"Unknown method '{method}'"
    raise DataUnavailableError(msg)


# ---------------------------------------------------------------------------
# Streaming helpers (internal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RunDone:
    """Sentinel pushed to the stream queue when a run finishes."""

    error: BaseException | None = None


# ---------------------------------------------------------------------------
# Event helpers (internal)
# ---------------------------------------------------------------------------


def _active_exception() -> str:
    """Return the current active exception as a string."""
    import sys  # noqa: PLC0415

    exc = sys.exc_info()[1]
    if exc is not None:
        return f"{type(exc).__name__}: {exc}"
    return "unknown"


_RESULT_PREVIEW_MAX_LEN = 200


def _safe_result_preview(value: Any) -> str | None:  # noqa: PLR0911
    """Return a short, safe preview of a tool result for event metadata.

    Large results are truncated to avoid bloating events.
    """
    if value is None:
        return "null"
    if isinstance(value, str):
        if len(value) > _RESULT_PREVIEW_MAX_LEN:
            return value[:_RESULT_PREVIEW_MAX_LEN] + "..."
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, dict)):
        import json  # noqa: PLC0415

        s = json.dumps(value, default=str)
        if len(s) > _RESULT_PREVIEW_MAX_LEN:
            return s[:_RESULT_PREVIEW_MAX_LEN] + "..."
        return s
    return str(type(value).__name__)
