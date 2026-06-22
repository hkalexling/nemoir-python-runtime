"""Workflow event primitives for live streaming, UI observation, and debugging.

Defines the event types, channels, emitter, and sink that the runtime uses
to expose stage lifecycle, model deltas, tool calls, policy decisions, and
run-level outcomes as an observable async stream.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

WorkflowEventKind = Literal[
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
]

WorkflowEventChannel = Literal[
    "assistant",
    "progress",
    "reasoning",
    "reasoning_summary",
    "debug",
]


@dataclass(frozen=True)
class WorkflowEvent:
    kind: WorkflowEventKind
    run_id: str
    sequence: int
    timestamp: datetime
    stage_id: str | None = None
    channel: WorkflowEventChannel | None = None
    text: str | None = None
    capability: str | None = None
    tool_name: str | None = None
    args: Mapping[str, Any] | None = None
    output: Mapping[str, Any] | None = None
    result: Any = None
    error: str | None = None
    transition_to: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


WorkflowEventSink = Callable[[WorkflowEvent], Awaitable[None]]


class WorkflowEventEmitter:
    """Per-run event emitter with monotonic sequencing.

    The emitter is cheap when the sink is ``None``: ``emit()`` still
    constructs and returns the event (for tests), but no I/O happens.
    """

    def __init__(self, *, run_id: str, sink: WorkflowEventSink | None = None) -> None:
        self._run_id = run_id
        self._sink = sink
        self._seq = 0

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def sequence(self) -> int:
        return self._seq

    @property
    def has_sink(self) -> bool:
        """True when an actual consumer is attached.

        Use this to decide whether to activate provider-side streaming:
        only stream when someone is listening.
        """
        return self._sink is not None

    async def emit(
        self,
        kind: WorkflowEventKind,
        **kwargs: Any,
    ) -> WorkflowEvent:
        self._seq += 1
        event = WorkflowEvent(
            kind=kind,
            run_id=self._run_id,
            sequence=self._seq,
            timestamp=datetime.now(tz=UTC),
            **kwargs,
        )
        if self._sink is not None:
            await self._sink(event)
        return event
