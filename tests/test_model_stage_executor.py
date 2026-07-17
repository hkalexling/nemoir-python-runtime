from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.errors import ModelOutputValidationError
from nemoir_runtime.models import (
    ModelRequest,
    ModelResponse,
    ModelStageExecutor,
    ModelToolCall,
)
from nemoir_runtime.runtime import (
    StageContext,
    StageSpec,
    WriteSpec,
)
from nemoir_runtime.tools import Tool, ToolContext, ToolRegistry

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fake_adapter(responses: list[ModelResponse]) -> Any:
    """Build a fake adapter that returns the given responses in sequence."""

    class Adapter:
        def __init__(self) -> None:
            self.calls: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls.append(request)
            if not responses:
                msg = "no more responses"
                raise RuntimeError(msg)
            return responses.pop(0)

    return Adapter()


def _make_stage_ctx(
    *,
    stage_id: str = "Test",
    prompt: str = "test",
    writes: tuple[WriteSpec, ...] = (),
    allowed_capabilities: frozenset[str] = frozenset(),
    readable_context: dict[str, Any] | None = None,
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    options: dict[str, Any] | None = None,
) -> StageContext:
    tool_calls = tool_calls or []

    async def call_tool(
        capability: str, args: dict[str, Any], *, tool_name: str | None = None
    ) -> str:
        tool_calls.append((capability, args))
        return f"result-for-{capability}"

    opts: dict[str, Any] = options if options is not None else {}

    return StageContext(
        workflow_id="TestWorkflow",
        stage=StageSpec(
            id=stage_id,
            prompt=prompt,
            reads=(),
            writes=writes,
            requires=allowed_capabilities,
            transitions=(),
        ),
        inputs={},
        readable_context=readable_context or {},
        allowed_capabilities=allowed_capabilities,
        options=opts,  # type: ignore[arg-type]
        call_tool=call_tool,  # type: ignore[arg-type]
    )


def _read_tool() -> Tool:
    async def handler(*, path: Path, ctx: ToolContext) -> str:
        return f"read:{path}"

    return Tool(
        name="read",
        capability="fs.read",
        description="Read a file",
        input_schema={"path": Path},
        handler=handler,
    )


def _write_tool() -> Tool:
    async def handler(*, path: Path, content: str, ctx: ToolContext) -> None:
        return None

    return Tool(
        name="write_file",
        capability="fs.write",
        description="Write a file",
        input_schema={"path": Path, "content": str},
        handler=handler,
    )


# ------------------------------------------------------------------
# Content-only stage execution
# ------------------------------------------------------------------


async def test_content_only_stage_returns_output() -> None:
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    adapter = _fake_adapter([ModelResponse(content='{"summary": "done"}')])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
    )
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}
    assert len(adapter.calls) == 1


async def test_content_only_invalid_json_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content="not json")])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        options={"max_model_retries": 0},
    )
    with pytest.raises(ModelOutputValidationError, match="invalid JSON"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_json_array_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content='["not", "object"]')])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        options={"max_model_retries": 0},
    )
    with pytest.raises(ModelOutputValidationError, match="list instead of object"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_json_number_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content="42")])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        options={"max_model_retries": 0},
    )
    with pytest.raises(ModelOutputValidationError, match="int instead of object"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_missing_required_field_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content="{}")])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        options={"max_model_retries": 0},
    )
    with pytest.raises(ModelOutputValidationError, match="missing required"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_unknown_field_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content='{"summary": "ok", "extra": 1}')])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        options={"max_model_retries": 0},
    )
    with pytest.raises(ModelOutputValidationError, match="unknown output"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_path_output_converts_from_str() -> None:
    adapter = _fake_adapter([ModelResponse(content='{"file": "/tmp/x"}')])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="file", type="path", optional=False),),
    )
    result = await executor.execute(ctx)
    assert result == {"file": Path("/tmp/x")}


async def test_content_only_string_array_output_validated() -> None:
    adapter = _fake_adapter([ModelResponse(content='{"items": ["a", "b"]}')])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="items", type="string[]", optional=False),),
    )
    result = await executor.execute(ctx)
    assert result == {"items": ["a", "b"]}


async def test_content_only_string_array_non_string_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content='{"items": [1, 2]}')])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="items", type="string[]", optional=False),),
        options={"max_model_retries": 0},
    )
    with pytest.raises(ModelOutputValidationError, match="expected list"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


# ------------------------------------------------------------------
# Tool-call loop
# ------------------------------------------------------------------


async def test_tool_call_single_then_content() -> None:
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        capability: str, args: dict[str, Any], *, tool_name: str | None = None
    ) -> str:
        tool_calls_log.append((capability, args))
        return "ok"

    ctx = StageContext(
        workflow_id="Test",
        stage=StageSpec(
            id="Test",
            prompt="test",
            reads=(),
            writes=writes,
            requires=frozenset({"fs.read"}),
            transitions=(),
        ),
        inputs={},
        readable_context={},
        allowed_capabilities=frozenset({"fs.read"}),
        options={},  # type: ignore[arg-type]
        call_tool=call_tool,  # type: ignore[arg-type]
    )

    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="call_1",
                        name="read",
                        arguments={"path": "/tmp/test"},
                    ),
                ),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)

    result = await executor.execute(ctx)
    assert result == {"summary": "done"}
    assert len(tool_calls_log) == 1
    assert tool_calls_log[0] == ("fs.read", {"path": Path("/tmp/test")})


async def test_tool_call_path_arg_coerced() -> None:
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        capability: str, args: dict[str, Any], *, tool_name: str | None = None
    ) -> str:
        tool_calls_log.append((capability, args))
        assert isinstance(args["path"], Path)
        return "ok"

    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
        tool_calls=tool_calls_log,  # type: ignore[arg-type]
    )
    # Override call_tool for explicit assertion
    ctx = StageContext(
        workflow_id=ctx.workflow_id,
        stage=ctx.stage,
        inputs=ctx.inputs,
        readable_context=ctx.readable_context,
        allowed_capabilities=ctx.allowed_capabilities,
        options=ctx.options,
        call_tool=call_tool,  # type: ignore[arg-type]
    )

    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="read",
                        arguments={"path": "/tmp/foo"},
                    ),
                ),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)

    await executor.execute(ctx)


async def test_unknown_tool_error_fed_back_then_success() -> None:
    """Unknown tool on first call is fed back as a tool result, not a fatal error.

    Tool-call errors do not consume max_model_retries; the model recovers
    on the next response.
    """
    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(ModelToolCall(id="c1", name="nonexistent", arguments={}),),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        allowed_capabilities=frozenset({"fs.read"}),
        options={"max_model_retries": 0},  # tool errors must NOT be bounded by this
    )
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}


async def test_tool_round_limit_exceeded() -> None:
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    tool_call = ModelResponse(
        content=None,
        tool_calls=(
            ModelToolCall(
                id="c1",
                name="read",
                arguments={"path": "/tmp/x"},
            ),
        ),
    )
    adapter = _fake_adapter([tool_call] * 5)
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools, max_tool_rounds=3)
    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
    )
    with pytest.raises(ModelOutputValidationError, match="exceeded max_tool_rounds"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_persistent_tool_errors_exhaust_max_tool_rounds() -> None:
    """Repeated tool-call errors do NOT consume max_model_retries; only max_tool_rounds."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    bad_call = ModelResponse(
        content=None,
        tool_calls=(ModelToolCall(id="c1", name="nonexistent", arguments={}),),
    )
    # 10 bad responses; max_model_retries=0 must NOT cause early death.
    adapter = _fake_adapter([bad_call] * 10)
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools, max_tool_rounds=3)
    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
        options={"max_model_retries": 0},
    )
    with pytest.raises(ModelOutputValidationError, match="exceeded max_tool_rounds"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)
    assert len(adapter.calls) == 4  # 3 rounds + the one that trips the cap


async def test_disallowed_capability_fed_back_then_success() -> None:
    """Disallowed capability error is fed back; model recovers on next response."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "/tmp/x", "content": "x"},
                    ),
                ),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    tools = ToolRegistry([_read_tool(), _write_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
        options={"max_model_retries": 0},  # tool errors must NOT be bounded by this
    )
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}


async def test_fs_write_triggers_policy_chain() -> None:
    """fs.write through ctx.call_tool triggers before-policy (fs.read + user.confirm).

    This test relies on the real WorkflowRuntime for policy enforcement.
    The model executor calls ctx.call_tool -> runtime enforces policies.
    """
    from nemoir_runtime.runtime import (  # noqa: PLC0415
        PolicySpec,
        RefSpec,
        RequiredCapabilitySpec,
        TriggerSpec,
        WorkflowManifest,
        WorkflowRuntime,
    )

    async def confirm_handler(*, message: str, ctx: ToolContext) -> bool:
        return True

    tools = ToolRegistry(
        [
            _read_tool(),
            _write_tool(),
            Tool(
                name="confirm",
                capability="user.confirm",
                description="confirm",
                input_schema={"message": str},
                handler=confirm_handler,
            ),
        ]
    )
    writes = (WriteSpec(name="summary", type="string", optional=False),)

    manifest = WorkflowManifest(
        workflow_id="PolicyTest",
        entry_stage_id="Apply",
        exit_stage_ids=frozenset({"Apply"}),
        inputs=(),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm"}),
        policies=(
            PolicySpec(
                id="before fs.write(path) requires fs.read(path), user.confirm",
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
        stages=(
            StageSpec(
                id="Apply",
                prompt="apply",
                reads=(),
                writes=writes,
                requires=frozenset({"fs.write"}),
                transitions=(),
            ),
        ),
    )

    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "/tmp/changes.txt", "content": "diff"},
                    ),
                ),
            ),
            ModelResponse(content='{"summary": "applied"}'),
        ]
    )

    executor = ModelStageExecutor(model=adapter, tools=tools)
    runtime = WorkflowRuntime(manifest=manifest, tools=tools, stage_executor=executor)
    result = await runtime.run({})
    assert result.output["summary"] == "applied"


async def test_one_tool_round_then_output_succeeds_with_max_tool_rounds_1() -> None:
    """One tool call then final JSON succeeds with max_tool_rounds=1."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        capability: str, args: dict[str, Any], *, tool_name: str | None = None
    ) -> str:
        tool_calls_log.append((capability, args))
        return "ok"

    ctx = StageContext(
        workflow_id="Test",
        stage=StageSpec(
            id="Test",
            prompt="test",
            reads=(),
            writes=writes,
            requires=frozenset({"fs.read"}),
            transitions=(),
        ),
        inputs={},
        readable_context={},
        allowed_capabilities=frozenset({"fs.read"}),
        options={},  # type: ignore[arg-type]
        call_tool=call_tool,  # type: ignore[arg-type]
    )

    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(ModelToolCall(id="c1", name="read", arguments={"path": "/tmp/x"}),),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools, max_tool_rounds=1)
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}
    assert len(tool_calls_log) == 1


async def test_two_tool_rounds_fail_with_max_tool_rounds_1() -> None:
    """Two consecutive tool-call responses fail with max_tool_rounds=1."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    tool_call = ModelResponse(
        content=None,
        tool_calls=(ModelToolCall(id="c1", name="read", arguments={"path": "/tmp/x"}),),
    )
    adapter = _fake_adapter([tool_call, tool_call])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools, max_tool_rounds=1)
    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
    )
    with pytest.raises(ModelOutputValidationError, match="exceeded max_tool_rounds"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


# ------------------------------------------------------------------
# Multi-fs.write routing + policy (plan Medium-Tests-1)
# ------------------------------------------------------------------


async def test_model_executor_routes_edit_file_through_fs_write_policy(
    tmp_path: Path,
) -> None:
    """Model requests edit_file; executor calls edit_file not write_file,
    and the ``before fs.write`` policy still applies."""
    from nemoir_runtime.official_tools import edit_file, write_file  # noqa: PLC0415
    from nemoir_runtime.runtime import (  # noqa: PLC0415
        InputSpec,
        PolicySpec,
        RefSpec,
        RequiredCapabilitySpec,
        TriggerSpec,
        WorkflowManifest,
        WorkflowRuntime,
    )

    # Pre-create a file for edit_file to operate on.
    target = tmp_path / "target.txt"
    target.write_text("hello old world")

    policy_calls: list[str] = []

    async def read_stub(*, path: Path, ctx: ToolContext) -> str:
        policy_calls.append("fs.read")
        return "ok"

    async def confirm_stub(*, message: str, ctx: ToolContext) -> bool:
        policy_calls.append("user.confirm")
        return True

    read_tool = Tool(
        name="read_file",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=read_stub,
    )
    confirm_tool = Tool(
        name="confirm",
        capability="user.confirm",
        description="c",
        input_schema={"message": str},
        handler=confirm_stub,
    )

    tools = ToolRegistry([read_tool, write_file, edit_file, confirm_tool])

    before_policy = PolicySpec(
        id="before fs.write(path) requires fs.read(path), user.confirm",
        kind="before",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        requires=(
            RequiredCapabilitySpec(
                capability="fs.read",
                args={"path": RefSpec(kind="bound", name="path")},
            ),
            RequiredCapabilitySpec(capability="user.confirm", args={}),
        ),
    )

    stages = (
        StageSpec(
            id="S",
            prompt="p",
            reads=(),
            writes=(WriteSpec(name="out", type="string", optional=False),),
            requires=frozenset({"fs.write"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="TestRoute",
        entry_stage_id="S",
        exit_stage_ids=frozenset({"S"}),
        inputs=(InputSpec(name="cwd", type="path"),),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm"}),
        policies=(before_policy,),
        stages=stages,
    )

    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="edit_file",
                        arguments={
                            "path": str(target),
                            "content": "old",
                            "new_content": "new",
                        },
                    ),
                ),
            ),
            ModelResponse(content='{"out": "done"}'),
        ]
    )

    executor = ModelStageExecutor(model=adapter, tools=tools)
    runtime = WorkflowRuntime(manifest=manifest, tools=tools, stage_executor=executor)
    result = await runtime.run({"cwd": str(tmp_path)})

    assert result.output["out"] == "done"
    # The before-policy chain must have run: fs.read then user.confirm
    # before the fs.write handler.
    assert policy_calls == ["fs.read", "user.confirm"]
    # edit_file replaced "old"→"new".
    assert target.read_text() == "hello new world"


# ------------------------------------------------------------------
# Retry behavior: recoverable model errors
# ------------------------------------------------------------------


async def test_invalid_json_retry_then_success() -> None:
    """Invalid JSON on first attempt, valid JSON on retry succeeds."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    adapter = _fake_adapter(
        [
            ModelResponse(content="not json"),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(writes=writes)
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}
    # Two calls: initial + retry
    assert len(adapter.calls) == 2


async def test_missing_required_output_retry_then_success() -> None:
    """Missing required field on first attempt, valid on retry succeeds."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    adapter = _fake_adapter(
        [
            ModelResponse(content="{}"),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(writes=writes)
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}
    assert len(adapter.calls) == 2


async def test_invalid_json_exhausts_retries_raises() -> None:
    """Persistent invalid JSON after max retries raises."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    adapter = _fake_adapter(
        [
            ModelResponse(content="not json"),
            ModelResponse(content="also not json"),
            ModelResponse(content="still not json"),
            ModelResponse(content="yet again not json"),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=writes,
        options={"max_model_retries": 2},
    )
    with pytest.raises(ModelOutputValidationError, match="invalid JSON"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)
    # Initial + 2 retries = 3 calls
    assert len(adapter.calls) == 3


async def test_invalid_tool_args_retry_then_success() -> None:
    """Invalid tool args on first call, valid on retry succeeds."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    tools = ToolRegistry([_read_tool()])

    adapter = _fake_adapter(
        [
            # First: tool call with missing required arg 'path'
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="read",
                        arguments={"wrong_arg": 1},
                    ),
                ),
            ),
            # Retry: valid tool call then final output
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c2",
                        name="read",
                        arguments={"path": "/tmp/x"},
                    ),
                ),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
    )
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}
    # Initial tool-call + retry tool-call + final output = 3 model calls
    assert len(adapter.calls) == 3


async def test_unknown_tool_retry_then_success() -> None:
    """Unknown tool on first call, valid tool on retry succeeds."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    tools = ToolRegistry([_read_tool()])

    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="nonexistent",
                        arguments={},
                    ),
                ),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
    )
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}
    assert len(adapter.calls) == 2


async def test_tool_invocation_error_retry_then_success() -> None:
    """Tool handler ToolInvocationError on first call, valid on retry succeeds."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)

    call_count = 0

    async def flaky_handler(*, path: Path, ctx: ToolContext) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            msg = "simulated handler failure"
            raise RuntimeError(msg)
        return f"read:{path}"

    flaky_tool = Tool(
        name="flaky",
        capability="fs.read",
        description="Flaky tool",
        input_schema={"path": Path},
        handler=flaky_handler,
    )
    tools = ToolRegistry([flaky_tool])

    # Use a real ToolRegistry.call-based call_tool that exercises the handler
    async def real_call_tool(
        capability: str, args: dict[str, Any], *, tool_name: str | None = None
    ) -> Any:
        return await tools.call(
            capability,
            args,
            ToolContext(
                workflow_id="Test",
                stage_id="Test",
                inputs={},
            ),
            tool_name=tool_name,
        )

    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="flaky",
                        arguments={"path": "/tmp/x"},
                    ),
                ),
            ),
            # Retry: same valid tool call + final output
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c2",
                        name="flaky",
                        arguments={"path": "/tmp/x"},
                    ),
                ),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = StageContext(
        workflow_id="Test",
        stage=StageSpec(
            id="Test",
            prompt="test",
            reads=(),
            writes=writes,
            requires=frozenset({"fs.read"}),
            transitions=(),
        ),
        inputs={},
        readable_context={},
        allowed_capabilities=frozenset({"fs.read"}),
        options={"max_model_retries": 3},  # type: ignore[arg-type]
        call_tool=real_call_tool,  # type: ignore[arg-type]
    )
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}
    assert len(adapter.calls) == 3
    assert call_count == 2


async def test_model_retry_event_emitted_during_stage_output_retry() -> None:
    """model_retry event is emitted when stage output validation retries."""
    from nemoir_runtime.events import WorkflowEvent, WorkflowEventEmitter  # noqa: PLC0415

    writes = (WriteSpec(name="summary", type="string", optional=False),)
    collected: list[WorkflowEvent] = []

    async def sink(event: WorkflowEvent) -> None:
        collected.append(event)

    async def noop_call_tool(*args: Any, **kwargs: Any) -> str:
        return "ok"

    emitter = WorkflowEventEmitter(run_id="r1", sink=sink)
    adapter = _fake_adapter(
        [
            ModelResponse(content="not json"),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = StageContext(
        workflow_id="Test",
        stage=StageSpec(
            id="Test",
            prompt="p",
            reads=(),
            writes=writes,
            requires=frozenset(),
            transitions=(),
        ),
        inputs={},
        readable_context={},
        allowed_capabilities=frozenset(),
        options={"max_model_retries": 3},  # type: ignore[arg-type]
        call_tool=noop_call_tool,  # type: ignore[arg-type]
        event_emitter=emitter,
    )
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}

    retries = [e for e in collected if e.kind == "model_retry"]
    assert len(retries) == 1
    assert retries[0].stage_id == "Test"
    assert retries[0].metadata is not None
    assert retries[0].metadata.get("category") == "stage_output"  # type: ignore[union-attr]
    assert retries[0].metadata.get("attempt") == 1  # type: ignore[union-attr]


async def test_tool_call_retry_emits_model_retry_event() -> None:
    """model_retry event is emitted when a tool-call error triggers retry."""
    from nemoir_runtime.events import WorkflowEvent, WorkflowEventEmitter  # noqa: PLC0415

    writes = (WriteSpec(name="summary", type="string", optional=False),)
    collected: list[WorkflowEvent] = []
    tools = ToolRegistry([_read_tool()])

    async def sink(event: WorkflowEvent) -> None:
        collected.append(event)

    async def noop_call_tool(*args: Any, **kwargs: Any) -> str:
        return "ok"

    emitter = WorkflowEventEmitter(run_id="r1", sink=sink)

    adapter = _fake_adapter(
        [
            ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="read",
                        arguments={"missing": "path"},
                    ),
                ),
            ),
            ModelResponse(content='{"summary": "done"}'),
        ]
    )
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = StageContext(
        workflow_id="Test",
        stage=StageSpec(
            id="Test",
            prompt="p",
            reads=(),
            writes=writes,
            requires=frozenset({"fs.read"}),
            transitions=(),
        ),
        inputs={},
        readable_context={},
        allowed_capabilities=frozenset({"fs.read"}),
        options={"max_model_retries": 3},  # type: ignore[arg-type]
        call_tool=noop_call_tool,  # type: ignore[arg-type]
        event_emitter=emitter,
    )
    result = await executor.execute(ctx)
    assert result == {"summary": "done"}

    retries = [e for e in collected if e.kind == "model_retry"]
    assert len(retries) == 1
    assert retries[0].stage_id == "Test"
    assert retries[0].metadata is not None
    assert retries[0].metadata.get("category") == "tool_call"  # type: ignore[union-attr]


async def test_max_model_retries_zero_preserves_hard_fail() -> None:
    """max_model_retries=0 preserves original hard-fail behavior."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    adapter = _fake_adapter([ModelResponse(content="not json")])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=writes,
        options={"max_model_retries": 0},
    )
    with pytest.raises(ModelOutputValidationError):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)
    # Only one call — no retry attempted
    assert len(adapter.calls) == 1


async def test_retry_respects_default_max_model_retries_from_options() -> None:
    """When options has no max_model_retries key, default 3 is used."""
    writes = (WriteSpec(name="summary", type="string", optional=False),)
    # 5 invalid responses; default max_retries=3 means 4 total attempts then raise
    adapter = _fake_adapter(
        [
            ModelResponse(content="not json 1"),
            ModelResponse(content="not json 2"),
            ModelResponse(content="not json 3"),
            ModelResponse(content="not json 4"),
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(writes=writes, options={})
    with pytest.raises(ModelOutputValidationError):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)
    # Initial + 3 retries = 4 calls
    assert len(adapter.calls) == 4


async def test_number_write_passes_through_model_stage() -> None:
    """Model-backed stage with type=number output passes validation."""
    writes = (WriteSpec(name="score", type="number", optional=False),)
    adapter = _fake_adapter([ModelResponse(content='{"score": 1.5}')])
    executor = ModelStageExecutor(model=adapter, tools=ToolRegistry([]))
    ctx = _make_stage_ctx(writes=writes)
    result = await executor.execute(ctx)
    assert result == {"score": 1.5}


async def test_number_write_rejects_bool_from_model() -> None:
    """Model returns bool for type=number output — rejected."""
    writes = (WriteSpec(name="score", type="number", optional=False),)
    # Default max_model_retries=3; provide enough copies for retries + final.
    adapter = _fake_adapter([ModelResponse(content='{"score": true}')] * 4)
    executor = ModelStageExecutor(model=adapter, tools=ToolRegistry([]))
    ctx = _make_stage_ctx(writes=writes)
    with pytest.raises(ModelOutputValidationError, match="expected int or float"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)
