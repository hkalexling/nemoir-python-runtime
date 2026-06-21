"""Tests for WorkflowRuntime.stream(), event ordering, and event fields."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.errors import (
    MaxStepsExceededError,
    NoTransitionMatchedError,
    PolicyDeniedError,
    ToolInvocationError,
)
from nemoir_runtime.events import WorkflowEvent, WorkflowEventEmitter
from nemoir_runtime.runtime import (
    ExprSpec,
    GuardSpec,
    InputSpec,
    PolicySpec,
    RefSpec,
    RequiredCapabilitySpec,
    RunOptions,
    StageContext,
    StageSpec,
    TransitionSpec,
    TriggerSpec,
    WorkflowManifest,
    WorkflowRuntime,
    WriteSpec,
)
from nemoir_runtime.tools import Tool, ToolContext, ToolRegistry

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_tools() -> ToolRegistry:
    async def read_fn(*, path: Path, ctx: ToolContext) -> str:
        return f"read:{path}"

    read_tool = Tool(
        name="read",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=read_fn,
    )
    return ToolRegistry([read_tool])


def _simple_manifest() -> WorkflowManifest:
    return WorkflowManifest(
        workflow_id="TestEvents",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"B"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.read"}),
        policies=(),
        stages=(
            StageSpec(
                id="A",
                prompt="A",
                reads=(),
                writes=(WriteSpec(name="out_a", type="string", optional=False),),
                requires=frozenset(),
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
        ),
    )


def _scripted_exec(
    outputs: dict[str, dict[str, object]],
) -> Any:
    class Exec:
        async def execute(self, ctx: StageContext) -> dict[str, object]:
            return dict(outputs[ctx.stage.id])

    return Exec()


# ------------------------------------------------------------------
# Event ordering tests
# ------------------------------------------------------------------


async def test_stream_yields_events_in_order() -> None:
    manifest = _simple_manifest()
    executor = _scripted_exec({"A": {"out_a": "a"}, "B": {"out_b": "b"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    kinds = [e.kind for e in events]
    assert kinds == [
        "run_started",
        "stage_started",
        "stage_completed",
        "transition_selected",
        "stage_started",
        "stage_completed",
        "run_completed",
    ]
    assert events[0].run_id == events[-1].run_id
    assert events[0].run_id == events[2].run_id


async def test_events_have_monotonic_sequence() -> None:
    manifest = _simple_manifest()
    executor = _scripted_exec({"A": {"out_a": "a"}, "B": {"out_b": "b"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    for i, e in enumerate(events):
        assert e.sequence == i + 1


async def test_stage_started_includes_stage_id() -> None:
    manifest = _simple_manifest()
    executor = _scripted_exec({"A": {"out_a": "a"}, "B": {"out_b": "b"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    started = [e for e in events if e.kind == "stage_started"]
    assert started[0].stage_id == "A"
    assert started[1].stage_id == "B"


async def test_stage_completed_includes_output() -> None:
    manifest = _simple_manifest()
    executor = _scripted_exec({"A": {"out_a": "a"}, "B": {"out_b": "final"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    completed = [e for e in events if e.kind == "stage_completed"]
    assert completed[0].output == {"out_a": "a"}
    assert completed[1].output == {"out_b": "final"}


async def test_transition_selected_includes_target_and_reason() -> None:
    manifest = _simple_manifest()
    executor = _scripted_exec({"A": {"out_a": "a"}, "B": {"out_b": "b"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    ts = [e for e in events if e.kind == "transition_selected"]
    assert len(ts) == 1
    assert ts[0].stage_id == "A"
    assert ts[0].transition_to == "B"


async def test_run_completed_includes_result() -> None:
    manifest = _simple_manifest()
    executor = _scripted_exec({"A": {"out_a": "a"}, "B": {"out_b": "b"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    rc = [e for e in events if e.kind == "run_completed"]
    assert len(rc) == 1
    assert rc[0].result is not None
    assert rc[0].result.output["out_b"] == "b"


async def test_run_started_is_first_event() -> None:
    manifest = _simple_manifest()
    executor = _scripted_exec({"A": {"out_a": "a"}, "B": {"out_b": "b"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    assert events[0].kind == "run_started"


# ------------------------------------------------------------------
# Failure event tests
# ------------------------------------------------------------------


async def test_run_failed_on_max_steps() -> None:
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="x", type="string", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(to="A", priority=0, reason="loop", guard=GuardSpec(kind="always")),
            ),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Loop",
        entry_stage_id="A",
        exit_stage_ids=frozenset(),
        inputs=(),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )
    executor = _scripted_exec({"A": {"x": "x"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    with pytest.raises(MaxStepsExceededError):  # type: ignore[reportUnknownMemberType]  # noqa: PT012
        async for event in runtime.stream({"task": "t"}, options=RunOptions(max_steps=3)):
            events.append(event)

    kinds = [e.kind for e in events]
    assert "run_failed" in kinds
    # After removing the inner emit, there is exactly one run_failed.
    rf_list = [e for e in events if e.kind == "run_failed"]
    assert len(rf_list) == 1
    rf = rf_list[0]
    assert rf.error is not None
    assert "max_steps" in rf.error


async def test_run_failed_on_no_transition() -> None:
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
                        left=None,  # type: ignore[arg-type]
                        right=None,  # type: ignore[arg-type]
                    ),
                ),
            ),
        ),
        StageSpec(id="B", prompt="B", reads=(), writes=(), requires=frozenset(), transitions=()),
    )
    manifest = WorkflowManifest(
        workflow_id="NoTrans",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"B"}),
        inputs=(),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )
    executor = _scripted_exec({"A": {"x": "x"}})
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []
    with pytest.raises(NoTransitionMatchedError):  # type: ignore[reportUnknownMemberType]  # noqa: PT012
        async for event in runtime.stream({"task": "t"}):
            events.append(event)

    kinds = [e.kind for e in events]
    assert "run_failed" in kinds


# ------------------------------------------------------------------
# Live streaming (events arrive before stage completes)
# ------------------------------------------------------------------


async def test_stream_yields_stage_started_before_stage_completes() -> None:
    """Proof that stream yields events live, not after run finishes."""
    release = asyncio.Event()
    seen_before_release = False

    class BlockingExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, object]:  # noqa: ARG002
            await release.wait()
            return {"x": "ok"}

    single_stage = StageSpec(
        id="A",
        prompt="A",
        reads=(),
        writes=(WriteSpec(name="x", type="string", optional=False),),
        requires=frozenset(),
        transitions=(),
    )
    manifest = WorkflowManifest(
        workflow_id="BT",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(),
        capabilities=frozenset(),
        policies=(),
        stages=(single_stage,),
    )

    executor = BlockingExecutor()
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=executor)

    events: list[WorkflowEvent] = []

    async def consume_and_collect() -> None:
        nonlocal seen_before_release
        async for event in runtime.stream({"task": "t"}):
            events.append(event)
            if event.kind == "stage_started":
                seen_before_release = not release.is_set()
                release.set()  # unblock executor
            if event.kind == "run_completed":
                break

    await asyncio.wait_for(consume_and_collect(), timeout=5.0)

    assert seen_before_release, "stage_started must be seen before executor completes"
    assert any(e.kind == "run_completed" for e in events)


# ------------------------------------------------------------------
# Early stream break cancels background run
# ------------------------------------------------------------------


async def test_early_stream_break_cancels_run_cleanly() -> None:
    """Early break from stream cancels background task cleanly."""
    events: list[WorkflowEvent] = []

    class FastExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, object]:  # noqa: ARG002
            return {"out_a": "ok"}

    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset(),
            transitions=(
                TransitionSpec(to="A", priority=0, reason="loop", guard=GuardSpec(kind="always")),
            ),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="CancelTest",
        entry_stage_id="A",
        exit_stage_ids=frozenset(),
        inputs=(),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )
    runtime = WorkflowRuntime(manifest=manifest, tools=_make_tools(), stage_executor=FastExecutor())

    async for event in runtime.stream({"task": "t"}):
        events.append(event)
        if event.kind == "stage_started":
            break  # exit stream early

    assert len(events) > 0
    assert events[0].kind == "run_started"
    assert events[-1].kind == "stage_started"


# ------------------------------------------------------------------
# Event emitter with no sink is harmless
# ------------------------------------------------------------------


async def test_emitter_without_sink_is_cheap() -> None:
    emitter = WorkflowEventEmitter(run_id="test", sink=None)
    assert not emitter.has_sink

    event = await emitter.emit("run_started")
    assert event.kind == "run_started"
    assert event.run_id == "test"
    assert event.sequence == 1


# ------------------------------------------------------------------
# Policy/tool events in stream
# ------------------------------------------------------------------


async def test_stream_includes_policy_and_tool_events() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def read_fn(*, path: Path, ctx: ToolContext) -> str:
        calls.append(("fs.read", {"path": str(path)}))
        return "ok"

    tools = ToolRegistry(
        [
            Tool(
                name="read",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=read_fn,
            )
        ]
    )

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
        workflow_id="PT",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(),
        capabilities=frozenset({"fs.read"}),
        policies=(),
        stages=stages,
    )

    class ToolExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, object]:
            await ctx.call_tool("fs.read", {"path": Path("/tmp/x")})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=tools, stage_executor=ToolExecutor())

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    kinds = [e.kind for e in events]
    assert "tool_call_started" in kinds
    assert "tool_call_completed" in kinds

    tc_started = next(e for e in events if e.kind == "tool_call_started")
    assert tc_started.capability == "fs.read"
    assert tc_started.tool_name == "read"
    assert tc_started.args is not None
    assert tc_started.args["path"] == Path("/tmp/x")


# ------------------------------------------------------------------
# Before-policy required tool events in stream (High-1 regression)
# ------------------------------------------------------------------


async def test_stream_includes_before_policy_required_tool_events() -> None:
    """Policy-required calls (fs.read, user.confirm) emit events in stream."""
    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    async def read_fn(*, path: Path, ctx: ToolContext) -> str:
        tool_calls_log.append(("fs.read", {"path": str(path)}))
        return "read-ok"

    async def write_fn(*, path: Path, content: str, ctx: ToolContext) -> None:
        tool_calls_log.append(("fs.write", {"path": str(path), "content": content}))

    async def confirm_fn(*, message: str, ctx: ToolContext) -> bool:
        tool_calls_log.append(("user.confirm", {"message": message}))
        return True

    tools = ToolRegistry(
        [
            Tool(
                name="read",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=read_fn,
            ),
            Tool(
                name="write",
                capability="fs.write",
                description="w",
                input_schema={"path": Path, "content": str},
                handler=write_fn,
            ),
            Tool(
                name="confirm",
                capability="user.confirm",
                description="c",
                input_schema={"message": str},
                handler=confirm_fn,
            ),
        ]
    )

    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"fs.write"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="BeforePolicyStream",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm"}),
        policies=(
            PolicySpec(
                id="before-fs.write-requires-fs.read-user.confirm",
                kind="before",
                trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
                requires=(
                    RequiredCapabilitySpec(
                        capability="fs.read",
                        args={"path": RefSpec(kind="bound", name="path")},
                    ),
                    RequiredCapabilitySpec(capability="user.confirm", args={}),
                ),
            ),
        ),
        stages=stages,
    )

    class CallExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/x"), "content": "diff"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=tools, stage_executor=CallExecutor())

    events: list[WorkflowEvent] = []
    async for event in runtime.stream({"task": "t"}):
        events.append(event)

    kinds = [e.kind for e in events]
    # Core lifecycle
    assert "run_started" in kinds
    assert "stage_started" in kinds
    assert "stage_completed" in kinds
    assert "run_completed" in kinds

    # Before-policy checked
    pc = [
        e
        for e in events
        if e.kind == "policy_checked" and (e.metadata or {}).get("policy_kind") == "before"
    ]
    assert len(pc) == 1
    assert pc[0].capability == "fs.write"

    # Policy-required fs.read
    read_started = [e for e in events if e.kind == "tool_call_started" and e.tool_name == "read"]
    assert len(read_started) == 1
    assert read_started[0].capability == "fs.read"
    read_completed = [
        e for e in events if e.kind == "tool_call_completed" and e.tool_name == "read"
    ]
    assert len(read_completed) == 1

    # Policy-required user.confirm
    confirm_started = [
        e for e in events if e.kind == "tool_call_started" and e.tool_name == "confirm"
    ]
    assert len(confirm_started) == 1
    assert confirm_started[0].capability == "user.confirm"
    confirm_completed = [
        e for e in events if e.kind == "tool_call_completed" and e.tool_name == "confirm"
    ]
    assert len(confirm_completed) == 1

    # Original fs.write
    write_started = [e for e in events if e.kind == "tool_call_started" and e.tool_name == "write"]
    assert len(write_started) == 1
    assert write_started[0].capability == "fs.write"
    write_completed = [
        e for e in events if e.kind == "tool_call_completed" and e.tool_name == "write"
    ]
    assert len(write_completed) == 1

    # Tool call order: fs.read -> user.confirm -> fs.write
    assert len(tool_calls_log) == 3
    assert tool_calls_log[0][0] == "fs.read"
    assert tool_calls_log[1][0] == "user.confirm"
    assert tool_calls_log[2][0] == "fs.write"


# ------------------------------------------------------------------
# Failure-path event stream tests (Phase 5 follow-up review)
# ------------------------------------------------------------------


async def test_stream_includes_deny_policy_events() -> None:
    """Deny policy emits policy_checked, policy_denied, and run_failed in stream."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def read_fn(*, path: Path, ctx: ToolContext) -> str:
        calls.append(("fs.read", {"path": str(path)}))
        return "ok"

    tools = ToolRegistry(
        [
            Tool(
                name="read",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=read_fn,
            ),
        ]
    )

    deny_policy = PolicySpec(
        id="deny fs.read(path) if not cwd.contains(path)",
        kind="deny",
        trigger=TriggerSpec(capability="fs.read", bind={"path": "path"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="input", name="cwd")),
                method="contains",
                args=(ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),),
            ),
        ),
    )

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
        workflow_id="DenyStream",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="cwd", type="path"),),
        capabilities=frozenset({"fs.read"}),
        policies=(deny_policy,),
        stages=stages,
    )

    class CallExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, object]:
            await ctx.call_tool("fs.read", {"path": Path("/outside/cwd/file")})
            return {"out_a": "never"}

    runtime = WorkflowRuntime(manifest=manifest, tools=tools, stage_executor=CallExecutor())

    events: list[WorkflowEvent] = []
    with pytest.raises(PolicyDeniedError):  # type: ignore[reportUnknownMemberType]  # noqa: PT012
        async for event in runtime.stream({"cwd": Path("/safe/cwd")}):
            events.append(event)

    kinds = [e.kind for e in events]
    assert "policy_checked" in kinds
    assert "policy_denied" in kinds
    assert "run_failed" in kinds

    # policy_checked with denied=true
    pc = [e for e in events if e.kind == "policy_checked"]
    assert len(pc) == 1
    assert pc[0].capability == "fs.read"
    assert (pc[0].metadata or {}).get("denied") is True

    # policy_denied
    pd = [e for e in events if e.kind == "policy_denied"]
    assert len(pd) == 1
    assert pd[0].capability == "fs.read"

    # run_failed present
    rf = [e for e in events if e.kind == "run_failed"]
    assert len(rf) == 1

    # No tool_call_started/completed (handler never invoked)
    assert "tool_call_started" not in kinds
    assert "tool_call_completed" not in kinds
    assert len(calls) == 0


async def test_stream_user_confirm_false_emits_policy_denied() -> None:
    """user.confirm=False emits policy_denied and no original tool event."""
    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    async def confirm_fn(*, message: str, ctx: ToolContext) -> bool:
        tool_calls_log.append(("user.confirm", {"message": message}))
        return False  # deny

    async def write_fn(*, path: Path, content: str, ctx: ToolContext) -> None:
        tool_calls_log.append(("fs.write", {"path": str(path), "content": content}))

    tools = ToolRegistry(
        [
            Tool(
                name="confirm",
                capability="user.confirm",
                description="c",
                input_schema={"message": str},
                handler=confirm_fn,
            ),
            Tool(
                name="write",
                capability="fs.write",
                description="w",
                input_schema={"path": Path, "content": str},
                handler=write_fn,
            ),
        ]
    )

    before_policy = PolicySpec(
        id="before-fs.write-requires-user.confirm",
        kind="before",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        requires=(RequiredCapabilitySpec(capability="user.confirm", args={}),),
    )

    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"fs.write"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="ConfirmFalse",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(),
        capabilities=frozenset({"fs.write", "user.confirm"}),
        policies=(before_policy,),
        stages=stages,
    )

    class CallExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/x"), "content": "diff"})
            return {"out_a": "never"}

    runtime = WorkflowRuntime(manifest=manifest, tools=tools, stage_executor=CallExecutor())

    events: list[WorkflowEvent] = []
    with pytest.raises(PolicyDeniedError):  # type: ignore[reportUnknownMemberType]  # noqa: PT012
        async for event in runtime.stream({"task": "t"}):
            events.append(event)

    kinds = [e.kind for e in events]
    assert "policy_checked" in kinds
    assert "policy_denied" in kinds
    assert "run_failed" in kinds

    # policy_checked for the before policy
    pc = [
        e
        for e in events
        if e.kind == "policy_checked" and (e.metadata or {}).get("policy_kind") == "before"
    ]
    assert len(pc) == 1
    assert pc[0].capability == "fs.write"

    # policy_denied for user.confirm returning False
    pd = [e for e in events if e.kind == "policy_denied"]
    assert len(pd) == 1
    assert pd[0].capability == "fs.write"

    # user.confirm was called and returned False
    assert len(tool_calls_log) == 1
    assert tool_calls_log[0][0] == "user.confirm"

    # No fs.write tool events (original call blocked)
    write_started = [
        e for e in events if e.kind == "tool_call_started" and e.capability == "fs.write"
    ]
    assert len(write_started) == 0


async def test_stream_tool_handler_exception_emits_tool_call_failed() -> None:
    """Tool handler exception emits tool_call_failed and run_failed in stream."""

    async def failing_read(*, path: Path, ctx: ToolContext) -> str:
        msg = "disk full"
        raise RuntimeError(msg)

    tools = ToolRegistry(
        [
            Tool(
                name="read",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=failing_read,
            ),
        ]
    )

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
        workflow_id="ToolFail",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(),
        capabilities=frozenset({"fs.read"}),
        policies=(),
        stages=stages,
    )

    class CallExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, object]:
            await ctx.call_tool("fs.read", {"path": Path("/tmp/x")})
            return {"out_a": "never"}

    runtime = WorkflowRuntime(manifest=manifest, tools=tools, stage_executor=CallExecutor())

    events: list[WorkflowEvent] = []
    with pytest.raises(ToolInvocationError, match="disk full"):  # type: ignore[reportUnknownMemberType]  # noqa: PT012
        async for event in runtime.stream({"task": "t"}):
            events.append(event)

    kinds = [e.kind for e in events]
    assert "tool_call_started" in kinds
    assert "tool_call_failed" in kinds
    assert "run_failed" in kinds

    # tool_call_started
    tcs = [e for e in events if e.kind == "tool_call_started"]
    assert len(tcs) == 1
    assert tcs[0].capability == "fs.read"
    assert tcs[0].tool_name == "read"

    # tool_call_failed
    tcf = [e for e in events if e.kind == "tool_call_failed"]
    assert len(tcf) == 1
    assert tcf[0].capability == "fs.read"
    assert tcf[0].tool_name == "read"
    assert tcf[0].error is not None
    assert "disk full" in tcf[0].error

    # run_failed
    rf = [e for e in events if e.kind == "run_failed"]
    assert len(rf) == 1
    assert "disk full" in (rf[0].error or "")


async def test_workflow_event_channel_includes_reasoning() -> None:
    """The ``reasoning`` channel value is a valid WorkflowEventChannel."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from nemoir_runtime.events import WorkflowEventChannel  # noqa: PLC0415

    # Confirm the literal value is declared.
    assert "reasoning" in WorkflowEventChannel.__args__  # type: ignore[attr-defined]

    event = WorkflowEvent(
        kind="model_delta",
        run_id="r1",
        sequence=1,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        channel="reasoning",
        text="thinking...",
    )
    assert event.channel == "reasoning"
    assert event.text == "thinking..."
