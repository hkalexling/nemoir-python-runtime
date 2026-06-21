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
) -> StageContext:
    tool_calls = tool_calls or []

    async def call_tool(
        capability: str, args: dict[str, Any], *, tool_name: str | None = None
    ) -> str:
        tool_calls.append((capability, args))
        return f"result-for-{capability}"

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
        options={},  # type: ignore[arg-type]
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
    )
    with pytest.raises(ModelOutputValidationError, match="invalid JSON"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_json_array_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content='["not", "object"]')])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
    )
    with pytest.raises(ModelOutputValidationError, match="list instead of object"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_json_number_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content="42")])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
    )
    with pytest.raises(ModelOutputValidationError, match="int instead of object"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_missing_required_field_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content="{}")])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
    )
    with pytest.raises(ModelOutputValidationError, match="missing required"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


async def test_content_only_unknown_field_raises() -> None:
    adapter = _fake_adapter([ModelResponse(content='{"summary": "ok", "extra": 1}')])
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
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


async def test_unknown_tool_name_raises() -> None:
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
        ]
    )
    tools = ToolRegistry([_read_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        allowed_capabilities=frozenset({"fs.read"}),
    )
    with pytest.raises(ModelOutputValidationError, match="unknown tool"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


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


async def test_tool_outside_allowed_capabilities_raises() -> None:
    """A model requests a tool whose capability is NOT in stage.allowed_capabilities."""
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
        ]
    )
    tools = ToolRegistry([_read_tool(), _write_tool()])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    ctx = _make_stage_ctx(
        writes=writes,
        allowed_capabilities=frozenset({"fs.read"}),
    )

    # The executor's capability visibility check fires BEFORE calling ctx.call_tool.
    with pytest.raises(ModelOutputValidationError, match="not allowed"):  # type: ignore[reportUnknownMemberType]
        await executor.execute(ctx)


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
