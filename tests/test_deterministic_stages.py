"""Tests for deterministic (non-LLM) stage execution via exec:."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.errors import (
    PolicyDeniedError,
    StageOutputValidationError,
    WorkflowValidationError,
)
from nemoir_runtime.models import ModelResponse, ModelStageExecutor
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
    TransitionSpec,
    TriggerSpec,
    WorkflowManifest,
    WorkflowRuntime,
    WriteSpec,
)
from nemoir_runtime.tools import ToolContext, ToolRegistry, tool

if TYPE_CHECKING:
    from nemoir_runtime.events import WorkflowEvent


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_single_stage_tool_manifest(
    *,
    stage_id: str = "R",
    capability: str = "fs.read",
    args: dict[str, RefSpec] | None = None,
    writes: tuple[WriteSpec, ...] = (WriteSpec(name="content", type="string", optional=False),),
    inputs: tuple[InputSpec, ...] = (InputSpec(name="p", type="string"),),
    policies: tuple[PolicySpec, ...] = (),
) -> WorkflowManifest:
    """Build a minimal manifest with a single tool-kind exit stage."""
    if args is None:
        args = {"path": RefSpec(kind="input", name="p")}
    exec_args = {k: ExprSpec(kind="ref", ref=v) for k, v in args.items()}
    return WorkflowManifest(
        workflow_id="Test",
        entry_stage_id=stage_id,
        exit_stage_ids=frozenset({stage_id}),
        inputs=inputs,
        capabilities=frozenset({capability}),
        policies=policies,
        stages=(
            StageSpec(
                id=stage_id,
                prompt="",
                reads=tuple(
                    ReadSpec(ref=RefSpec(kind="input", name=v.name), optional=False)
                    for v in args.values()
                    if v.kind == "input" and v.name is not None
                ),
                writes=writes,
                requires=frozenset({capability}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability=capability,
                    args=exec_args,
                ),
            ),
        ),
    )


async def _collect_events(
    runtime: WorkflowRuntime,
    inputs: dict[str, Any],
    options: RunOptions | None = None,
) -> list[WorkflowEvent]:
    events: list[WorkflowEvent] = []

    async def sink(event: WorkflowEvent) -> None:
        events.append(event)

    await runtime.run(inputs, options=options, event_sink=sink)
    return events


# ------------------------------------------------------------------
# Construction / tool selection
# ------------------------------------------------------------------


async def test_deterministic_stage_runs_and_calls_fixed_tool() -> None:
    """A deterministic fs.read stage runs without any model call."""
    called: list[bool] = []

    @tool(capability="fs.read", description="r", returns={"content": str})
    async def rd(*, path: Path, ctx: ToolContext) -> dict[str, str]:
        called.append(True)
        return {"content": "hello", "extra": "ignored"}

    manifest = _make_single_stage_tool_manifest(capability="fs.read")
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type] — not used for tool stages
    )
    result = await runtime.run({"p": "/tmp/x"})
    assert result.output == {"content": "hello"}
    assert called == [True]


async def test_tool_selection_no_match_raises() -> None:
    """If no registered tool satisfies the stage, fail at init time."""

    @tool(capability="fs.read", description="r")
    async def rd(*, path: Path, ctx: ToolContext) -> str:
        return "ok"

    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        writes=(WriteSpec(name="content", type="string", optional=False),),
    )
    with pytest.raises(WorkflowValidationError, match=r"no registered tool"):
        WorkflowRuntime(
            manifest=manifest,
            tools=ToolRegistry([rd]),
            stage_executor=None,  # type: ignore[arg-type]
        )


async def test_tool_selection_auto_selects_capability_match() -> None:
    """When one tool matches input+output among multiple, auto-select it."""

    @tool(capability="fs.read", description="a")
    async def rd_a(*, path: Path, ctx: ToolContext) -> str:
        return "a"

    @tool(capability="fs.read", description="b", returns={"content": str})
    async def rd_b(*, path: Path, ctx: ToolContext) -> dict[str, str]:
        return {"content": "b"}

    manifest = _make_single_stage_tool_manifest(capability="fs.read")
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd_a, rd_b]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    result = await runtime.run({"p": "/tmp/x"})
    assert result.output == {"content": "b"}


async def test_tool_with_no_known_output_shape_satisfies_output_empty() -> None:
    """Tool with None output_schema satisfies output: {} stages."""

    @tool(capability="fs.read", description="r")
    async def rd(*, path: Path, ctx: ToolContext) -> str:
        return "ignored"

    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        writes=(),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    result = await runtime.run({"p": "/tmp/x"})
    assert result.output == {}


# ------------------------------------------------------------------
# Dataclass result normalization
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _Result:
    content: str
    size: int


async def test_dataclass_result_normalization_with_projection() -> None:
    """Dataclass result → asdict() → projected to declared writes, extras dropped."""

    @tool(capability="fs.read", description="r", returns={"content": str})
    async def rd(*, path: Path, ctx: ToolContext) -> _Result:
        return _Result(content="data", size=42)

    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        writes=(WriteSpec(name="content", type="string", optional=False),),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    result = await runtime.run({"p": "/tmp/x"})
    assert result.output == {"content": "data"}


@dataclass(frozen=True)
class _BadResult:
    lines: int


async def test_dataclass_result_missing_required_field_raises() -> None:
    """Dataclass result missing a required write → StageOutputValidationError."""

    @tool(capability="fs.read", description="r", returns={"content": str})
    async def rd(*, path: Path, ctx: ToolContext) -> _BadResult:
        return _BadResult(lines=0)

    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        writes=(WriteSpec(name="content", type="string", optional=False),),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    with pytest.raises(StageOutputValidationError, match="missing required output field"):
        await runtime.run({"p": "/tmp/x"})


async def test_scalar_result_with_outputs_raises() -> None:
    """Bare scalar result when stage declares outputs → error (no wrapping)."""

    @tool(capability="fs.read", description="r", returns={"content": str})
    async def rd(*, path: Path, ctx: ToolContext) -> str:  # returns scalar despite output_schema
        return "naked"

    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        writes=(WriteSpec(name="content", type="string", optional=False),),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    with pytest.raises(StageOutputValidationError, match=r"deterministic stage"):
        await runtime.run({"p": "/tmp/x"})


# ------------------------------------------------------------------
# output: {} ignores result
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _EmptyResult:
    x: str


async def test_output_empty_ignores_dataclass_result() -> None:
    """output: {} → dataclass result ignored entirely."""

    @tool(capability="fs.read", description="r")
    async def rd(*, path: Path, ctx: ToolContext) -> _EmptyResult:
        return _EmptyResult(x="hi")

    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        writes=(),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    result = await runtime.run({"p": "/tmp/x"})
    assert result.output == {}


async def test_output_empty_ignores_none_result() -> None:
    """output: {} → None result accepted."""

    @tool(capability="fs.read", description="r")
    async def rd(*, path: Path, ctx: ToolContext) -> None:
        return None

    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        writes=(),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    result = await runtime.run({"p": "/tmp/x"})
    assert result.output == {}


# ------------------------------------------------------------------
# Deny policy on deterministic call
# ------------------------------------------------------------------


async def test_deny_policy_blocks_deterministic_stage() -> None:
    """A deny policy on the exec capability blocks the call."""

    @tool(capability="fs.read", description="r", returns={"content": str})
    async def rd(*, path: Path, ctx: ToolContext) -> dict[str, str]:
        return {"content": "blocked"}

    deny_policy = PolicySpec(
        id="deny_fs_read",
        kind="deny",
        trigger=TriggerSpec(capability="fs.read", bind={"path": "path"}),
        condition=ExprSpec(kind="literal", type="bool", value=True),
    )
    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        policies=(deny_policy,),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    with pytest.raises(PolicyDeniedError):
        await runtime.run({"p": "/tmp/x"})


# ------------------------------------------------------------------
# Events
# ------------------------------------------------------------------


async def test_deterministic_stage_emits_tool_events_not_model_events() -> None:
    """Tool events are emitted; model events are not for deterministic stages."""

    @tool(capability="fs.read", description="r", returns={"content": str})
    async def rd(*, path: Path, ctx: ToolContext) -> dict[str, str]:
        return {"content": "ok"}

    manifest = _make_single_stage_tool_manifest(capability="fs.read")
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    events = await _collect_events(runtime, {"p": "/tmp/x"})
    kinds = {e.kind for e in events}
    assert "tool_call_started" in kinds
    assert "tool_call_completed" in kinds
    assert "stage_started" in kinds
    assert "stage_completed" in kinds
    assert "run_started" in kinds
    assert "run_completed" in kinds
    assert "model_delta" not in kinds
    assert "model_completed" not in kinds
    assert "model_retry" not in kinds


# ------------------------------------------------------------------
# string[] write against list[str] dataclass tool
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _LinesResult:
    lines: list[str]


async def test_string_array_write_satisfied_by_list_str_dataclass() -> None:
    """string[] write matches list[str] dataclass field (regression: Medium #7)."""

    @tool(capability="fs.read", description="r")
    async def rd(*, path: Path, ctx: ToolContext) -> _LinesResult:
        return _LinesResult(lines=["a", "b"])

    manifest = _make_single_stage_tool_manifest(
        capability="fs.read",
        writes=(WriteSpec(name="lines", type="string[]", optional=False),),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    result = await runtime.run({"p": "/tmp/x"})
    assert result.output == {"lines": ["a", "b"]}


# ------------------------------------------------------------------
# path coercion
# ------------------------------------------------------------------


@dataclass(frozen=True)
class _PathResult:
    path: str  # official tools annotate path: str


async def test_path_typed_output_str_coerced_to_path() -> None:
    """path-typed write returned as str → coerced to Path (regression: §1.9)."""

    @tool(capability="fs.read", description="r", returns={"path": str})
    async def rd(*, path: Path, ctx: ToolContext) -> _PathResult:
        return _PathResult(path="/tmp/out.txt")

    args = {"path": RefSpec(kind="input", name="p")}
    exec_args = {k: ExprSpec(kind="ref", ref=v) for k, v in args.items()}
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="R",
        exit_stage_ids=frozenset({"R"}),
        inputs=(InputSpec(name="p", type="path"),),
        capabilities=frozenset({"fs.read"}),
        policies=(),
        stages=(
            StageSpec(
                id="R",
                prompt="",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="p"), optional=False),),
                writes=(WriteSpec(name="path", type="path", optional=False),),
                requires=frozenset({"fs.read"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.read",
                    args=exec_args,
                ),
            ),
        ),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    result = await runtime.run({"p": "/tmp/in.txt"})
    assert isinstance(result.output["path"], Path)
    assert str(result.output["path"]) == "/tmp/out.txt"


# ------------------------------------------------------------------
# Bool-branch transitions on deterministic output
# ------------------------------------------------------------------


async def test_bool_branch_transition_on_deterministic_output() -> None:
    """Bool-branch transition fires on deterministic stage output."""

    @tool(capability="os.shell", description="s", returns={"ok": bool, "log": str})
    async def sh(*, command: str, ctx: ToolContext) -> dict[str, Any]:
        return {"ok": True, "log": "done"}

    args = {"command": RefSpec(kind="input", name="cmd")}
    exec_args = {k: ExprSpec(kind="ref", ref=v) for k, v in args.items()}
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="Check",
        exit_stage_ids=frozenset({"Done"}),
        inputs=(InputSpec(name="cmd", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(),
        stages=(
            StageSpec(
                id="Check",
                prompt="",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="cmd"), optional=False),),
                writes=(
                    WriteSpec(name="ok", type="bool", optional=False),
                    WriteSpec(name="log", type="string", optional=False),
                ),
                requires=frozenset({"os.shell"}),
                transitions=(
                    TransitionSpec(
                        to="Done",
                        priority=0,
                        reason="output_branch_true",
                        guard=GuardSpec(
                            kind="eq",
                            left=ExprSpec(
                                kind="ref",
                                ref=RefSpec(kind="node_output", node="Check", field="ok"),
                            ),
                            right=ExprSpec(kind="literal", type="bool", value=True),
                        ),
                    ),
                ),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="os.shell",
                    args=exec_args,
                ),
            ),
            StageSpec(
                id="Done",
                prompt="",
                reads=(),
                writes=(WriteSpec(name="summary", type="string", optional=False),),
                requires=frozenset(),
                transitions=(),
            ),
        ),
    )

    class FakeM:
        async def complete(self, _request: Any) -> Any:
            return ModelResponse(content='{"summary": "done"}')

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([sh]),
        stage_executor=ModelStageExecutor(model=FakeM(), tools=ToolRegistry([sh])),
    )
    result = await runtime.run({"cmd": "echo hi"})
    assert result.output == {"summary": "done"}


# ------------------------------------------------------------------
# Mixed deterministic + model workflow
# ------------------------------------------------------------------


async def test_mixed_deterministic_and_model_stages() -> None:
    """A deterministic stage followed by a model stage runs end-to-end."""

    @tool(capability="fs.read", description="r", returns={"content": str})
    async def rd(*, path: Path, ctx: ToolContext) -> dict[str, str]:
        return {"content": "the content"}

    class FakeModel:
        async def complete(self, _request: Any) -> Any:
            return ModelResponse(content='{"summary": "processed"}')

    args = {"path": RefSpec(kind="input", name="f")}
    exec_args = {k: ExprSpec(kind="ref", ref=v) for k, v in args.items()}
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="Read",
        exit_stage_ids=frozenset({"Process"}),
        inputs=(InputSpec(name="f", type="string"),),
        capabilities=frozenset({"fs.read"}),
        policies=(),
        stages=(
            StageSpec(
                id="Read",
                prompt="",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="f"), optional=False),),
                writes=(WriteSpec(name="content", type="string", optional=False),),
                requires=frozenset({"fs.read"}),
                transitions=(
                    TransitionSpec(
                        to="Process",
                        priority=0,
                        reason="fallthrough",
                        guard=GuardSpec(kind="always"),
                    ),
                ),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.read",
                    args=exec_args,
                ),
            ),
            StageSpec(
                id="Process",
                prompt="process: @Read.content",
                reads=(
                    ReadSpec(
                        ref=RefSpec(kind="node_output", node="Read", field="content"),
                        optional=False,
                    ),
                ),
                writes=(WriteSpec(name="summary", type="string", optional=False),),
                requires=frozenset(),
                transitions=(),
            ),
        ),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd]),
        stage_executor=ModelStageExecutor(model=FakeModel(), tools=ToolRegistry([rd])),
    )
    result = await runtime.run({"f": "/tmp/x"})
    assert result.output == {"summary": "processed"}


# ------------------------------------------------------------------
# Ambiguous tool match (plan §1.8, §16.1)
# ------------------------------------------------------------------


async def test_tool_selection_ambiguous_match_raises() -> None:
    """When multiple tools equally satisfy a deterministic stage,
    construction raises WorkflowValidationError naming the candidates."""

    @tool(capability="fs.read", description="a", returns={"content": str})
    async def rd_a(*, path: Path, ctx: ToolContext) -> dict[str, str]:
        return {"content": "a"}

    @tool(capability="fs.read", description="b", returns={"content": str})
    async def rd_b(*, path: Path, ctx: ToolContext) -> dict[str, str]:
        return {"content": "b"}

    manifest = _make_single_stage_tool_manifest(capability="fs.read")
    with pytest.raises(WorkflowValidationError, match=r"multiple tools equally satisfy"):
        WorkflowRuntime(
            manifest=manifest,
            tools=ToolRegistry([rd_a, rd_b]),
            stage_executor=None,  # type: ignore[arg-type]
        )
    # Error must name at least one candidate tool.
    with pytest.raises(WorkflowValidationError, match=r"rd_a|rd_b"):
        WorkflowRuntime(
            manifest=manifest,
            tools=ToolRegistry([rd_a, rd_b]),
            stage_executor=None,  # type: ignore[arg-type]
        )


# ------------------------------------------------------------------
# Before policy on deterministic stages (plan §14)
# ------------------------------------------------------------------


async def test_before_policy_runs_before_deterministic_stage() -> None:
    """A before fs.write requires user.confirm policy is enforced
    on a deterministic exec: fs.write stage."""
    confirm_called: list[bool] = []

    @tool(capability="user.confirm", description="c")
    async def cnf(*, message: str, ctx: ToolContext) -> bool:
        confirm_called.append(True)
        return True

    write_called: list[dict[str, Any]] = []

    @tool(capability="fs.write", description="w", returns={"result": str})
    async def wr(*, path: Path, content: str, ctx: ToolContext) -> dict[str, str]:
        write_called.append({"path": str(path), "content": content})
        return {"result": "written"}

    exec_args = {
        "path": ExprSpec(kind="literal", type="string", value="/tmp/out.txt"),
        "content": ExprSpec(kind="literal", type="string", value="data"),
    }
    policies = (
        PolicySpec(
            id="before-fs-write",
            kind="before",
            trigger=TriggerSpec(capability="fs.write", bind={"path": "path", "content": "content"}),
            requires=(
                RequiredCapabilitySpec(
                    capability="user.confirm",
                    args={"message": RefSpec(kind="bound", name="path")},
                ),
            ),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="W",
        exit_stage_ids=frozenset({"W"}),
        inputs=(),
        capabilities=frozenset({"fs.write", "user.confirm"}),
        policies=policies,
        stages=(
            StageSpec(
                id="W",
                prompt="",
                reads=(),
                writes=(WriteSpec(name="result", type="string", optional=False),),
                requires=frozenset({"fs.write"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.write",
                    args=exec_args,
                ),
            ),
        ),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([cnf, wr]),
        stage_executor=None,  # type: ignore[arg-type]
    )
    await runtime.run({})
    assert confirm_called == [True]
    assert len(write_called) == 1


# ------------------------------------------------------------------
# All-deterministic workflow (plan §14, §16.5)
# ------------------------------------------------------------------


async def test_all_deterministic_workflow_runs_with_placeholder_model() -> None:
    """An all-deterministic workflow runs end-to-end; the model executor
    is never invoked (§16.5 — model= is still required but unused)."""

    @tool(capability="fs.read", description="r", returns={"content": str})
    async def rd(*, path: Path, ctx: ToolContext) -> dict[str, str]:
        return {"content": "step1"}

    @tool(capability="fs.write", description="w", returns={"result": str})
    async def wr(*, path: Path, content: str, ctx: ToolContext) -> dict[str, str]:
        return {"result": "written"}

    class SentinelModel:
        """Model that records every call — should never be invoked."""

        calls: list[Any] = []  # noqa: RUF012

        @classmethod
        def reset(cls) -> None:
            cls.calls = []

        async def complete(self, _request: Any) -> Any:
            self.__class__.calls.append(_request)
            return ModelResponse(content='{"summary": "should not be called"}')

    SentinelModel.reset()

    read_args = {
        "path": ExprSpec(kind="literal", type="string", value="/tmp/x"),
    }
    write_args = {
        "path": ExprSpec(kind="literal", type="string", value="/tmp/out.txt"),
        "content": ExprSpec(
            kind="ref",
            ref=RefSpec(kind="node_output", node="Read", field="content"),
        ),
    }
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="Read",
        exit_stage_ids=frozenset({"Write"}),
        inputs=(),
        capabilities=frozenset({"fs.read", "fs.write"}),
        policies=(),
        stages=(
            StageSpec(
                id="Read",
                prompt="",
                reads=(),
                writes=(WriteSpec(name="content", type="string", optional=False),),
                requires=frozenset({"fs.read"}),
                transitions=(
                    TransitionSpec(
                        to="Write",
                        priority=0,
                        reason="fallthrough",
                        guard=GuardSpec(kind="always"),
                    ),
                ),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.read",
                    args=read_args,
                ),
            ),
            StageSpec(
                id="Write",
                prompt="",
                reads=(
                    ReadSpec(
                        ref=RefSpec(
                            kind="node_output",
                            node="Read",
                            field="content",
                        ),
                        optional=False,
                    ),
                ),
                writes=(WriteSpec(name="result", type="string", optional=False),),
                requires=frozenset({"fs.write"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.write",
                    args=write_args,
                ),
            ),
        ),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([rd, wr]),
        stage_executor=ModelStageExecutor(model=SentinelModel(), tools=ToolRegistry([rd, wr])),
    )
    result = await runtime.run({})
    assert result.output == {"result": "written"}
    assert SentinelModel.calls == []
