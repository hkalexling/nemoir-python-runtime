from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.errors import (
    DataUnavailableError,
    MaxStepsExceededError,
    NoTransitionMatchedError,
    StageOutputValidationError,
    WorkflowValidationError,
)
from nemoir_runtime.runtime import (
    ExprSpec,
    GuardSpec,
    InputSpec,
    PolicySpec,
    ReadSpec,
    RefSpec,
    RunOptions,
    StageContext,
    StageSpec,
    TransitionSpec,
    WorkflowManifest,
    WorkflowResult,
    WorkflowRuntime,
    WriteSpec,
)
from nemoir_runtime.tools import Tool, ToolContext, ToolRegistry, tool

if TYPE_CHECKING:
    from collections.abc import Mapping


def _scripted_executor(outputs_by_stage: Mapping[str, list[Mapping[str, object]]]) -> Any:
    class ScriptedExecutor:
        def __init__(self) -> None:
            self.outputs: dict[str, list[Mapping[str, object]]] = {
                k: list(v) for k, v in outputs_by_stage.items()
            }

        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            bucket = self.outputs.get(ctx.stage.id, [])
            if not bucket:
                msg = f"No scripted output for stage '{ctx.stage.id}'"
                raise RuntimeError(msg)
            return bucket.pop(0)

    return ScriptedExecutor()


def _make_manifest(
    *,
    stages: tuple[StageSpec, ...],
    inputs: tuple[InputSpec, ...] = (InputSpec(name="task", type="string"),),
    entry_id: str = "A",
    exit_ids: frozenset[str] = frozenset({"B"}),
    capabilities: frozenset[str] = frozenset({"fs.read"}),
    policies: tuple[PolicySpec, ...] = (),
) -> WorkflowManifest:
    return WorkflowManifest(
        workflow_id="TestWorkflow",
        entry_stage_id=entry_id,
        exit_stage_ids=exit_ids,
        inputs=inputs,
        capabilities=capabilities,
        policies=policies,
        stages=stages,
    )


def _make_tool_registry() -> ToolRegistry:
    @tool(capability="fs.read", description="r")
    async def read_file(*, path: Path, ctx: ToolContext) -> str:
        return f"read:{path}"

    @tool(capability="fs.write", description="w")
    async def write_file(*, path: Path, content: str, ctx: ToolContext) -> None:
        pass

    @tool(capability="user.confirm", description="confirm")
    async def confirm(*, message: str, ctx: ToolContext) -> bool:
        return True

    return ToolRegistry([read_file, write_file, confirm])


# ------------------------------------------------------------------
# Entry / exit tests
# ------------------------------------------------------------------


async def test_entry_stage_starts_correctly() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"fs.read"}),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="fallthrough",
                    guard=GuardSpec(kind="always"),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out_b", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages)
    executor = _scripted_executor({"A": [{"out_a": "hello"}], "B": [{"out_b": "done"}]})
    runtime = WorkflowRuntime(
        manifest=manifest, tools=_make_tool_registry(), stage_executor=executor
    )

    result = await runtime.run({"task": "test"})
    assert isinstance(result, WorkflowResult)
    assert result.output["out_b"] == "done"
    assert result.state.current_stage_id == "B"
    assert result.state.steps == 2


# ------------------------------------------------------------------
# Transition tests
# ------------------------------------------------------------------


async def test_transitions_evaluated_by_priority() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="flag", type="bool", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="C",
                    priority=0,
                    reason="low_priority_branch",
                    guard=GuardSpec(kind="always"),
                ),
                TransitionSpec(
                    to="B",
                    priority=1,
                    reason="high_priority_branch",
                    guard=GuardSpec(kind="always"),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out_b", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
        StageSpec(
            id="C",
            prompt="C",
            reads=(),
            writes=(WriteSpec(name="out_c", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(
        stages=stages,
        exit_ids=frozenset({"B", "C"}),
    )
    executor = _scripted_executor({"A": [{"flag": True}], "C": [{"out_c": "c"}]})
    runtime = WorkflowRuntime(
        manifest=manifest, tools=_make_tool_registry(), stage_executor=executor
    )

    result = await runtime.run({"task": "test"})
    # Priority 0 wins, so A -> C
    assert result.output["out_c"] == "c"


async def test_has_value_and_missing_guards() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="opt", type="string", optional=True),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="has_value_branch",
                    guard=GuardSpec(
                        kind="has_value",
                        ref=RefSpec(kind="node_output", node="A", field="opt"),
                    ),
                ),
                TransitionSpec(
                    to="C",
                    priority=1,
                    reason="missing_branch",
                    guard=GuardSpec(
                        kind="missing",
                        ref=RefSpec(kind="node_output", node="A", field="opt"),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out_b", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
        StageSpec(
            id="C",
            prompt="C",
            reads=(),
            writes=(WriteSpec(name="out_c", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages, exit_ids=frozenset({"B", "C"}))

    # Test has_value: opt present
    executor = _scripted_executor({"A": [{"opt": "yes"}], "B": [{"out_b": "b"}]})
    runtime = WorkflowRuntime(
        manifest=manifest, tools=_make_tool_registry(), stage_executor=executor
    )
    result = await runtime.run({"task": "test"})
    assert result.output["out_b"] == "b"

    # Test missing: opt absent
    executor2 = _scripted_executor({"A": [{}], "C": [{"out_c": "c"}]})
    runtime2 = WorkflowRuntime(
        manifest=manifest, tools=_make_tool_registry(), stage_executor=executor2
    )
    result2 = await runtime2.run({"task": "test"})
    assert result2.output["out_c"] == "c"


async def test_eq_guard() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="ok", type="bool", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="true_branch",
                    guard=GuardSpec(
                        kind="eq",
                        left=ExprSpec(
                            kind="ref",
                            ref=RefSpec(kind="node_output", node="A", field="ok"),
                        ),
                        right=ExprSpec(kind="literal", type="bool", value=True),
                    ),
                ),
                TransitionSpec(
                    to="C",
                    priority=1,
                    reason="false_branch",
                    guard=GuardSpec(
                        kind="eq",
                        left=ExprSpec(
                            kind="ref",
                            ref=RefSpec(kind="node_output", node="A", field="ok"),
                        ),
                        right=ExprSpec(kind="literal", type="bool", value=False),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(WriteSpec(name="out_b", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
        StageSpec(
            id="C",
            prompt="C",
            reads=(),
            writes=(WriteSpec(name="out_c", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages, exit_ids=frozenset({"B", "C"}))

    executor = _scripted_executor({"A": [{"ok": True}], "B": [{"out_b": "b"}]})
    runtime = WorkflowRuntime(
        manifest=manifest, tools=_make_tool_registry(), stage_executor=executor
    )
    result = await runtime.run({"task": "test"})
    assert result.output["out_b"] == "b"

    executor2 = _scripted_executor({"A": [{"ok": False}], "C": [{"out_c": "c"}]})
    runtime2 = WorkflowRuntime(
        manifest=manifest, tools=_make_tool_registry(), stage_executor=executor2
    )
    result2 = await runtime2.run({"task": "test"})
    assert result2.output["out_c"] == "c"


# ------------------------------------------------------------------
# Error tests
# ------------------------------------------------------------------


async def test_missing_required_read_raises() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(
                ReadSpec(
                    ref=RefSpec(kind="input", name="nonexistent"),
                    optional=False,
                ),
            ),
            writes=(),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages, exit_ids=frozenset({"A"}))
    runtime = WorkflowRuntime(
        manifest=manifest, tools=_make_tool_registry(), stage_executor=_scripted_executor({})
    )

    with pytest.raises(DataUnavailableError, match="required read"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})


async def test_missing_required_output_raises() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages, exit_ids=frozenset({"A"}))
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{}]}),
    )

    with pytest.raises(StageOutputValidationError, match="missing required output"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})


async def test_wrong_output_type_raises() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="val", type="bool", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages, exit_ids=frozenset({"A"}))
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"val": "not-a-bool"}]}),
    )

    with pytest.raises(StageOutputValidationError, match="expected bool"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})


async def test_unknown_output_field_raises() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages, exit_ids=frozenset({"A"}))
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"surprise": 1}]}),
    )

    with pytest.raises(StageOutputValidationError, match="unknown output field"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})


async def test_max_steps_exceeded() -> None:
    # Looping A -> A (always)
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="x", type="string", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="A",
                    priority=0,
                    reason="loop",
                    guard=GuardSpec(kind="always"),
                ),
            ),
        ),
    )
    manifest = _make_manifest(stages=stages, exit_ids=frozenset())
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"x": "a"}] * 10}),
    )

    with pytest.raises(MaxStepsExceededError, match="max_steps"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"}, options=RunOptions(max_steps=5))


async def test_no_transition_matched_raises() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="x", type="string", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="never",
                    guard=GuardSpec(
                        kind="eq",
                        left=ExprSpec(kind="literal", type="bool", value=False),
                        right=ExprSpec(kind="literal", type="bool", value=True),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(),
            writes=(),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = _make_manifest(stages=stages, exit_ids=frozenset({"B"}))
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=_make_tool_registry(),
        stage_executor=_scripted_executor({"A": [{"x": "a"}]}),
    )

    with pytest.raises(NoTransitionMatchedError, match="no transition matched"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})


# ------------------------------------------------------------------
# Re-execution clears stale optional outputs
# ------------------------------------------------------------------


async def test_re_execution_clears_stale_optional_outputs() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="opt", type="string", optional=True),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="B",
                    priority=0,
                    reason="go_to_B",
                    guard=GuardSpec(kind="always"),
                ),
            ),
        ),
        StageSpec(
            id="B",
            prompt="B",
            reads=(
                ReadSpec(
                    ref=RefSpec(kind="node_output", node="A", field="opt"),
                    optional=True,
                ),
            ),
            writes=(WriteSpec(name="out_b", type="string", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="A",
                    priority=0,
                    reason="loop_on_has_value",
                    guard=GuardSpec(
                        kind="has_value",
                        ref=RefSpec(kind="node_output", node="A", field="opt"),
                    ),
                ),
                TransitionSpec(
                    to="C",
                    priority=1,
                    reason="exit_on_missing",
                    guard=GuardSpec(
                        kind="missing",
                        ref=RefSpec(kind="node_output", node="A", field="opt"),
                    ),
                ),
            ),
        ),
        StageSpec(
            id="C",
            prompt="C",
            reads=(),
            writes=(WriteSpec(name="done", type="string", optional=False),),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"C"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )

    outputs: Mapping[str, list[Mapping[str, object]]] = {
        "A": [
            {"opt": "first"},  # A produces opt
            {},  # A re-executes without opt — stale cleared
        ],
        "B": [
            {"out_b": "first-b"},  # loop: sees A.opt="first" → has_value → back to A
            {"out_b": "second-b"},  # exit: sees A.opt=None → missing → C
        ],
        "C": [
            {"done": "Done"},
        ],
    }
    executor = _scripted_executor(outputs)
    runtime = WorkflowRuntime(
        manifest=manifest, tools=_make_tool_registry(), stage_executor=executor
    )

    result = await runtime.run({"task": "test"})
    assert result.output["done"] == "Done"
    assert result.state.steps == 5


# ------------------------------------------------------------------
# Regression: RunOptions.metadata propagated to ToolContext
# ------------------------------------------------------------------


async def test_run_options_metadata_propagated_to_tool_context() -> None:
    captured_metadata: list[dict[str, Any]] = []

    async def read_meta(*, path: Path, ctx: ToolContext) -> str:
        captured_metadata.append(dict(ctx.metadata))
        return "ok"

    read_tool = Tool(
        name="read_file",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=read_meta,
    )
    registry = ToolRegistry([read_tool])

    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"fs.read"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.read"}),
        policies=(),
        stages=stages,
    )

    class MetadataExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.read", {"path": Path("/tmp/f.txt")})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=MetadataExecutor())
    await runtime.run(
        {"task": "test"}, options=RunOptions(metadata={"run_id": "abc", "env": "test"})
    )
    assert len(captured_metadata) == 1
    assert captured_metadata[0]["run_id"] == "abc"
    assert captured_metadata[0]["env"] == "test"


# ------------------------------------------------------------------
# Regression: concurrent runs with different metadata do not leak
# ------------------------------------------------------------------


async def test_concurrent_runs_do_not_leak_metadata() -> None:
    run_a_metadata: list[dict[str, Any]] = []
    run_b_metadata: list[dict[str, Any]] = []

    async def read_meta(*, path: Path, ctx: ToolContext) -> str:
        return ctx.metadata.get("run_name", "unknown")

    async def write_meta(*, path: Path, content: str, ctx: ToolContext) -> None:
        pass

    read_tool = Tool(
        name="read_file",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=read_meta,
    )
    write_tool = Tool(
        name="write_file",
        capability="fs.write",
        description="w",
        input_schema={"path": Path, "content": str},
        handler=write_meta,
    )
    registry = ToolRegistry([read_tool, write_tool])

    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset({"fs.read", "fs.write"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.read", "fs.write"}),
        policies=(),
        stages=stages,
    )

    class TaggingExecutor:
        def __init__(self, collector: list[dict[str, Any]]) -> None:
            self._collector = collector

        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            meta: dict[str, Any] = {}
            name = await ctx.call_tool("fs.read", {"path": Path("/tmp/x")})
            meta["fs.read"] = name
            await ctx.call_tool("fs.write", {"path": Path("/tmp/x"), "content": "x"})
            self._collector.append(meta)
            return {"out": "done"}

    async def run_with_metadata(run_name: str, collector: list[dict[str, Any]]) -> None:
        opts = RunOptions(metadata={"run_name": run_name})
        await WorkflowRuntime(
            manifest=manifest, tools=registry, stage_executor=TaggingExecutor(collector)
        ).run({"task": "test"}, options=opts)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_with_metadata("runner-A", run_a_metadata))
        tg.create_task(run_with_metadata("runner-B", run_b_metadata))

    assert len(run_a_metadata) == 1
    assert len(run_b_metadata) == 1
    assert run_a_metadata[0]["fs.read"] == "runner-A"
    assert run_b_metadata[0]["fs.read"] == "runner-B"


# ------------------------------------------------------------------
# Manifest construction validation tests (Review Comment 3)
# ------------------------------------------------------------------


def test_construction_fails_on_duplicate_stage_ids() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(),
            requires=frozenset(),
            transitions=(),
        ),
        StageSpec(
            id="A",
            prompt="A2",
            reads=(),
            writes=(),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )
    with pytest.raises(WorkflowValidationError, match="duplicate stage id"):  # type: ignore[reportUnknownMemberType]
        WorkflowRuntime(
            manifest=manifest,
            tools=_make_tool_registry(),
            stage_executor=_scripted_executor({}),
        )


def test_construction_fails_on_missing_entry_stage() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="Bogus",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )
    with pytest.raises(WorkflowValidationError, match="entry stage"):  # type: ignore[reportUnknownMemberType]
        WorkflowRuntime(
            manifest=manifest,
            tools=_make_tool_registry(),
            stage_executor=_scripted_executor({}),
        )


def test_construction_fails_on_missing_exit_stage() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(),
            requires=frozenset(),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"Bogus"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )
    with pytest.raises(WorkflowValidationError, match="exit stage"):  # type: ignore[reportUnknownMemberType]
        WorkflowRuntime(
            manifest=manifest,
            tools=_make_tool_registry(),
            stage_executor=_scripted_executor({}),
        )


def test_construction_fails_on_missing_transition_target() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(),
            requires=frozenset(),
            transitions=(
                TransitionSpec(
                    to="Bogus",
                    priority=0,
                    reason="bad",
                    guard=GuardSpec(kind="always"),
                ),
            ),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )
    with pytest.raises(WorkflowValidationError, match="transition to unknown stage"):  # type: ignore[reportUnknownMemberType]
        WorkflowRuntime(
            manifest=manifest,
            tools=_make_tool_registry(),
            stage_executor=_scripted_executor({}),
        )
