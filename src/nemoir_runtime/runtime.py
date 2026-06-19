from __future__ import annotations

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
from nemoir_runtime.tools import ToolContext, ToolRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

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
    # Accepted but not enforced in Phase 2. Only max_steps is enforced.
    timeout_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


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
    call_tool: Callable[[str, Mapping[str, Any]], Awaitable[Any]]


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
    ) -> WorkflowResult:
        opts = options if options is not None else RunOptions()
        stage_outputs: dict[str, dict[str, Any]] = {}
        current_id = self._manifest.entry_stage_id
        steps = 0

        while True:
            if steps >= opts.max_steps:
                msg = f"Workflow '{self._manifest.workflow_id}' exceeded max_steps={opts.max_steps}"
                raise MaxStepsExceededError(msg)

            stage = self._require_stage(current_id)
            readable = self._resolve_reads(stage, inputs, stage_outputs)
            ctx = self._make_stage_context(stage, inputs, readable, opts)
            raw_output = await self._stage_executor.execute(ctx)
            self._validate_output(stage, raw_output)
            stage_outputs[stage.id] = dict(raw_output)
            steps += 1

            if stage.id in self._exit_ids:
                return WorkflowResult(
                    output=raw_output,
                    state=WorkflowState(
                        current_stage_id=stage.id,
                        stage_outputs=dict(stage_outputs),
                        steps=steps,
                    ),
                )

            next_id = self._select_transition(stage, inputs, stage_outputs)
            current_id = next_id

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

    # ------------------------------------------------------------------
    # Transition evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _select_transition(
        stage: StageSpec,
        inputs: Mapping[str, Any],
        stage_outputs: Mapping[str, Mapping[str, Any]],
    ) -> str:
        sorted_transitions = sorted(stage.transitions, key=lambda t: t.priority)
        for trans in sorted_transitions:
            if WorkflowRuntime._evaluate_guard(trans.guard, inputs, stage_outputs):
                return trans.to
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
    ) -> StageContext:
        return StageContext(
            workflow_id=self._manifest.workflow_id,
            stage=stage,
            inputs=inputs,
            readable_context=readable,
            allowed_capabilities=stage.requires,
            options=options,
            call_tool=self._make_tool_caller(stage, inputs, options),
        )

    def _make_tool_caller(
        self,
        stage: StageSpec,
        inputs: Mapping[str, Any],
        run_opts: RunOptions,
    ) -> Callable[[str, Mapping[str, Any]], Awaitable[Any]]:
        async def call_tool(capability: str, args: Mapping[str, Any]) -> Any:
            return await self._enforce_and_call(stage, capability, args, inputs, run_opts)

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
    ) -> Any:
        if capability not in stage.requires:
            msg = f"capability '{capability}' is not available in stage '{stage.id}'"
            raise MissingCapabilityError(msg)
        return await self._enforce_and_call_with_policies(
            capability, args, inputs, stage, allow_before=True, run_opts=run_opts
        )

    async def _enforce_policy_call(
        self,
        capability: str,
        args: Mapping[str, Any],
        inputs: Mapping[str, Any],
        stage: StageSpec,
        run_opts: RunOptions,
    ) -> Any:
        return await self._enforce_and_call_with_policies(
            capability, args, inputs, stage, allow_before=False, run_opts=run_opts
        )

    async def _enforce_and_call_with_policies(
        self,
        capability: str,
        args: Mapping[str, Any],
        inputs: Mapping[str, Any],
        stage: StageSpec,
        *,
        allow_before: bool,
        run_opts: RunOptions,
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
                    msg = (
                        f"Policy '{policy.id}': condition evaluation failed "
                        f"for capability '{capability}': {e}"
                    )
                    raise PolicyEvaluationError(msg) from e
                if denied:
                    msg = f"Policy '{policy.id}' denied capability '{capability}'"
                    raise PolicyDeniedError(msg)

        for policy in policies:
            if policy.kind == "before" and allow_before:
                bound_args = self._bind_trigger_args(
                    policy.trigger, args, policy_id=policy.id, capability=capability
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
                        req.capability, req_args, inputs, stage, run_opts
                    )
                    if req.capability == "user.confirm" and result is False:
                        msg = (
                            f"Policy '{policy.id}': user.confirm returned False, "
                            f"blocking capability '{capability}'"
                        )
                        raise PolicyDeniedError(msg)

        ctx = ToolContext(
            workflow_id=self._manifest.workflow_id,
            stage_id=stage.id,
            inputs=inputs,
            metadata=run_opts.metadata,
        )
        return await self._tools.call(capability, args, ctx)

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
