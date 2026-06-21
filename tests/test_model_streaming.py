"""Tests for model streaming through ModelStageExecutor and fake adapters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemoir_runtime.events import WorkflowEvent, WorkflowEventEmitter
from nemoir_runtime.models import (
    ModelRequest,
    ModelResponse,
    ModelStageExecutor,
    ModelStreamChunk,
    ModelToolCall,
    supports_streaming,
)
from nemoir_runtime.runtime import StageContext, StageSpec, WriteSpec
from nemoir_runtime.tools import Tool, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fake_streaming_adapter(
    chunks: list[ModelStreamChunk],
) -> Any:
    """Build a fake streaming adapter that yields the given chunks."""

    class Adapter:
        def __init__(self) -> None:
            self.calls: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            """Required by ModelAdapter contract. Not used in streaming tests."""
            msg = "complete() not implemented; use stream() for this fake adapter"
            raise NotImplementedError(msg)

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
            self.calls.append(request)
            for chunk in chunks:
                yield chunk

    return Adapter()


def _fake_non_streaming_adapter(responses: list[ModelResponse]) -> Any:
    """Build a non-streaming adapter returning responses in sequence."""

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
    writes: tuple[WriteSpec, ...] = (),
    emitter: WorkflowEventEmitter | None = None,
) -> StageContext:
    async def call_tool(capability: str, args: dict[str, Any]) -> str:
        return f"ok-{capability}"

    return StageContext(
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
        options={},  # type: ignore[arg-type]
        call_tool=call_tool,  # type: ignore[arg-type]
        event_emitter=emitter,
    )


# ------------------------------------------------------------------
# supports_streaming
# ------------------------------------------------------------------


def test_supports_streaming_true_for_streaming_adapter() -> None:
    adapter = _fake_streaming_adapter([])
    assert supports_streaming(adapter)


def test_supports_streaming_false_for_complete_only_adapter() -> None:
    adapter = _fake_non_streaming_adapter([])
    assert not supports_streaming(adapter)


# ------------------------------------------------------------------
# Streaming adapter yields model_delta events
# ------------------------------------------------------------------


async def test_streaming_adapter_yields_model_deltas() -> None:
    collected: list[WorkflowEvent] = []

    async def sink(event: WorkflowEvent) -> None:
        collected.append(event)

    emitter = WorkflowEventEmitter(run_id="r1", sink=sink)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        emitter=emitter,
    )

    adapter = _fake_streaming_adapter(
        [
            ModelStreamChunk(kind="delta", channel="assistant", text="Hello "),
            ModelStreamChunk(kind="delta", channel="assistant", text="world"),
            ModelStreamChunk(
                kind="completed", response=ModelResponse(content='{"summary": "done"}')
            ),
        ]
    )
    tools = ToolRegistry([])
    executor = ModelStageExecutor(model=adapter, tools=tools)

    result = await executor.execute(ctx)
    assert result == {"summary": "done"}

    deltas = [e for e in collected if e.kind == "model_delta"]
    assert len(deltas) == 2
    assert deltas[0].text == "Hello "
    assert deltas[0].channel == "assistant"
    assert deltas[1].text == "world"

    completed = [e for e in collected if e.kind == "model_completed"]
    assert len(completed) == 1


async def test_streaming_adapter_yields_reasoning_channel() -> None:
    collected: list[WorkflowEvent] = []

    async def sink(event: WorkflowEvent) -> None:
        collected.append(event)

    emitter = WorkflowEventEmitter(run_id="r1", sink=sink)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        emitter=emitter,
    )

    adapter = _fake_streaming_adapter(
        [
            ModelStreamChunk(kind="delta", channel="reasoning_summary", text="I should say hello"),
            ModelStreamChunk(kind="completed", response=ModelResponse(content='{"summary": "hi"}')),
        ]
    )
    tools = ToolRegistry([])
    executor = ModelStageExecutor(model=adapter, tools=tools)

    await executor.execute(ctx)

    deltas = [e for e in collected if e.kind == "model_delta"]
    assert len(deltas) == 1
    assert deltas[0].channel == "reasoning_summary"
    assert deltas[0].text == "I should say hello"


async def test_streaming_adapter_with_tool_calls() -> None:
    collected: list[WorkflowEvent] = []

    async def sink(event: WorkflowEvent) -> None:
        collected.append(event)

    emitter = WorkflowEventEmitter(run_id="r1", sink=sink)

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
            prompt="p",
            reads=(),
            writes=(WriteSpec(name="summary", type="string", optional=False),),
            requires=frozenset({"fs.read"}),
            transitions=(),
        ),
        inputs={},
        readable_context={},
        allowed_capabilities=frozenset({"fs.read"}),
        options={},  # type: ignore[arg-type]
        call_tool=call_tool,  # type: ignore[arg-type]
        event_emitter=emitter,
    )

    # Use a streaming adapter that returns responses in sequence
    responses_seq = [
        ModelStreamChunk(
            kind="completed",
            response=ModelResponse(
                content=None,
                tool_calls=(ModelToolCall(id="c1", name="read", arguments={"path": "/tmp/x"}),),
            ),
        ),
        ModelStreamChunk(kind="completed", response=ModelResponse(content='{"summary": "ok"}')),
    ]

    class SeqAdapter:
        def __init__(self) -> None:
            self._idx = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            """Required by ModelAdapter contract. Not used in this test."""
            msg = "complete() not implemented; use stream() for this fake adapter"
            raise NotImplementedError(msg)

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:  # noqa: ARG002
            chunk = responses_seq[self._idx]
            self._idx += 1
            yield chunk

    adapter = SeqAdapter()

    async def handler(*, path: Path, ctx: ToolContext) -> str:
        return "read-ok"

    tools = ToolRegistry(
        [
            Tool(
                name="read",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=handler,
            )
        ]
    )
    executor = ModelStageExecutor(model=adapter, tools=tools)

    result = await executor.execute(ctx)
    assert result == {"summary": "ok"}
    assert len(tool_calls_log) == 1
    assert tool_calls_log[0][0] == "fs.read"


# ------------------------------------------------------------------
# Non-streaming adapters still work via complete()
# ------------------------------------------------------------------


async def test_non_streaming_adapter_emits_model_completed() -> None:
    collected: list[WorkflowEvent] = []

    async def sink(event: WorkflowEvent) -> None:
        collected.append(event)

    emitter = WorkflowEventEmitter(run_id="r1", sink=sink)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        emitter=emitter,
    )

    adapter = _fake_non_streaming_adapter([ModelResponse(content='{"summary": "ns"}')])
    tools = ToolRegistry([])
    executor = ModelStageExecutor(model=adapter, tools=tools)

    result = await executor.execute(ctx)
    assert result == {"summary": "ns"}

    deltas = [e for e in collected if e.kind == "model_delta"]
    assert len(deltas) == 0

    completed = [e for e in collected if e.kind == "model_completed"]
    assert len(completed) == 1


async def test_no_sink_falls_back_to_complete_quietly() -> None:
    """When no event sink is attached, don't activate streaming at all."""
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
    )
    # Use a complete-only adapter: even though it could be wrapped, the
    # executor should use complete() because emitter.has_sink is False.
    adapter = _fake_non_streaming_adapter([ModelResponse(content='{"summary": "ok"}')])
    tools = ToolRegistry([])
    executor = ModelStageExecutor(model=adapter, tools=tools)

    result = await executor.execute(ctx)
    assert result == {"summary": "ok"}


async def test_emitter_without_sink_falls_back_to_complete() -> None:
    """emitter exists but has no sink — doesn't switch to streaming mode."""
    emitter = WorkflowEventEmitter(run_id="r1", sink=None)
    ctx = _make_stage_ctx(
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        emitter=emitter,
    )
    # Use a complete-only adapter: emitter.has_sink is False, so streaming
    # is not activated. The executor calls complete() as normal.
    adapter = _fake_non_streaming_adapter([ModelResponse(content='{"summary": "fallback"}')])
    tools = ToolRegistry([])
    executor = ModelStageExecutor(model=adapter, tools=tools)
    result = await executor.execute(ctx)
    assert result == {"summary": "fallback"}
