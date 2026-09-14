"""Phase 4 taped replay: re-execute recorded paths with no live effects.

The end-to-end tests run a real workflow under ``profile="replay"`` with
counting fakes, then call :func:`replay_trace` with **no model, no tools,
and no human** — only the archive and passphrase. Matching proves the
replay core is self-contained; the call-count assertions prove the live
fakes were never touched during replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nemoir_runtime.errors import PolicyDeniedError
from nemoir_runtime.models import ModelRequest, ModelResponse, ModelStageExecutor, ModelToolCall
from nemoir_runtime.replay import (
    ReplayReport,
    TapedModelAdapter,
    TapedReplayError,
    TapedToolRegistry,
    _unmark_fixture,  # type: ignore[reportPrivateUsage]
    replay_trace,
)
from nemoir_runtime.runtime import (
    ExprSpec,
    GuardSpec,
    InputSpec,
    PolicySpec,
    RefSpec,
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
from nemoir_runtime.trace import HostProvenance, TraceRecorder


class _StageAdapter:
    """Fake provider: counts live calls so replay can prove independence."""

    def __init__(self, content: str = '{"score": 0.9}') -> None:
        self.content = content
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return ModelResponse(content=self.content)


def _manifest(*, deny: bool) -> WorkflowManifest:
    literal = ExprSpec(kind="literal", value=deny)
    return WorkflowManifest(
        workflow_id="ReplayE2E",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"C"}),
        inputs=(InputSpec(name="topic", type="string"),),
        capabilities=frozenset({"fs.read", "fs.write"}),
        policies=(
            PolicySpec(
                id="deny fs.write if guard",
                kind="deny",
                trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
                condition=literal,
            ),
        ),
        stages=(
            StageSpec(
                id="A",
                prompt="",
                reads=(),
                writes=(WriteSpec(name="text", type="string", optional=False),),
                requires=frozenset({"fs.read"}),
                transitions=(
                    TransitionSpec(
                        to="B",
                        priority=0,
                        reason="explicit_transition",
                        guard=GuardSpec(kind="always"),
                    ),
                ),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.read",
                    args={"path": ExprSpec(kind="literal", value="/work/a.txt", type="path")},
                ),
            ),
            StageSpec(
                id="B",
                prompt="",
                reads=(),
                writes=(WriteSpec(name="score", type="number", optional=False),),
                requires=frozenset(),
                transitions=(
                    TransitionSpec(
                        to="C",
                        priority=0,
                        reason="explicit_transition",
                        guard=GuardSpec(kind="always"),
                    ),
                ),
                execution=StageExecutionSpec(kind="model"),
            ),
            StageSpec(
                id="C",
                prompt="",
                reads=(),
                writes=(WriteSpec(name="done", type="bool", optional=False),),
                requires=frozenset({"fs.write"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.write",
                    args={
                        "path": ExprSpec(kind="literal", value="/work/out.txt", type="path"),
                        "content": ExprSpec(kind="literal", value="data", type="string"),
                    },
                ),
            ),
        ),
    )


def _registry(calls: list[tuple[str, Any]]) -> ToolRegistry:
    async def _read(*, path: Path, ctx: ToolContext) -> dict[str, Any]:
        calls.append(("fs.read", str(path)))
        return {"text": "hello"}

    async def _write(*, path: Path, content: str, ctx: ToolContext) -> dict[str, Any]:
        calls.append(("fs.write", str(path)))
        return {"done": True}

    return ToolRegistry(
        [
            Tool(
                name="reader",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=_read,
                output_schema={"text": str},
            ),
            Tool(
                name="writer",
                capability="fs.write",
                description="w",
                input_schema={"path": Path, "content": str},
                handler=_write,
                output_schema={"done": bool},
            ),
        ]
    )


async def _record(
    tmp_path: Path, *, deny: bool
) -> tuple[Path, _StageAdapter, list[tuple[str, Any]]]:
    adapter = _StageAdapter()
    calls: list[tuple[str, Any]] = []
    registry = _registry(calls)
    runtime = WorkflowRuntime(
        manifest=_manifest(deny=deny),
        tools=registry,
        stage_executor=ModelStageExecutor(model=adapter, tools=registry),
    )
    recorder = TraceRecorder.create(
        tmp_path / "run.nemotrace",
        profile="replay",
        provenance=HostProvenance(
            frontend="replay-e2e",
            target="python",
            compiler_version="replay-e2e",
            ir_version="0.1",
            ir_sha256="sha256:" + "cd" * 32,
        ),
        path_aliases={"$workspace": Path("/work")},
        safe_path_aliases=frozenset({"$workspace"}),
        vault_passphrase="replay-e2e-passphrase",  # noqa: S106
    )
    if deny:
        with pytest.raises(PolicyDeniedError):
            await runtime.run({"topic": "replay-e2e"}, trace_recorder=recorder)
    else:
        result = await runtime.run({"topic": "replay-e2e"}, trace_recorder=recorder)
        assert result.output["done"] is True
    return tmp_path / "run.nemotrace", adapter, calls


async def test_taped_replay_matches_recorded_path(tmp_path: Path) -> None:
    path, adapter, calls = await _record(tmp_path, deny=False)
    assert adapter.calls
    assert len(calls) == 2
    live_model_calls = len(adapter.calls)
    live_tool_calls = len(calls)
    # Replay receives no model, no tools, no human: archive + passphrase only.
    report: ReplayReport = await replay_trace(path, "replay-e2e-passphrase")
    assert report.matched, report.divergences
    assert report.divergences == ()
    assert report.replayed_status == "complete"
    assert report.steps == 3
    assert report.verification.semantic == "passed"
    # The live fakes prove replay independence: untouched by the replay.
    assert len(adapter.calls) == live_model_calls
    assert len(calls) == live_tool_calls


async def test_taped_replay_matches_failure_path(tmp_path: Path) -> None:
    path, _, _ = await _record(tmp_path, deny=True)
    report = await replay_trace(path, "replay-e2e-passphrase")
    assert report.matched, report.divergences
    assert report.replayed_status == "failed"


async def test_taped_replay_refuses_audit_archive(tmp_path: Path) -> None:
    adapter = _StageAdapter()
    calls: list[tuple[str, Any]] = []
    registry = _registry(calls)
    runtime = WorkflowRuntime(
        manifest=_manifest(deny=False),
        tools=registry,
        stage_executor=ModelStageExecutor(model=adapter, tools=registry),
    )
    recorder = TraceRecorder.create(tmp_path / "audit.nemotrace")
    result = await runtime.run({"topic": "replay-e2e"}, trace_recorder=recorder)
    assert result.output["done"] is True
    report = await replay_trace(tmp_path / "audit.nemotrace", "replay-e2e-passphrase")
    assert report.matched is False
    assert any("no replay vault" in d for d in report.divergences)
    assert report.verification.replayability == "playback-only"


async def test_taped_replay_wrong_passphrase(tmp_path: Path) -> None:
    path, _, _ = await _record(tmp_path, deny=False)
    report = await replay_trace(path, "wrong-passphrase")
    assert report.matched is False
    assert report.divergences == ("vault unlock failed",)


async def test_taped_fixtures_fail_closed() -> None:
    model = TapedModelAdapter({})
    with pytest.raises(TapedReplayError, match="no model fixture"):
        await model.complete(
            ModelRequest(stage_id="B", messages=(), tools=(), output_schema={}, options={})
        )

    tools = TapedToolRegistry([])
    with pytest.raises(TapedReplayError, match="no tool fixture"):
        await tools.call("fs.read", {}, ToolContext(workflow_id="w", stage_id="A", inputs={}))


class _ToolCallAdapter:
    """Fake provider: first response requests fs.read, then final content."""

    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        if len(self.calls) == 1:
            return ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="call-1",
                        name="run_shell",
                        arguments={"command": "echo /home/u/build-ok python"},
                    ),
                ),
            )
        return ModelResponse(content='{"done": true}')


def _scoped_registry(calls: list[tuple[str, Any]]) -> ToolRegistry:
    """Official shell tool; the live command is a harmless echo."""
    from nemoir_runtime.official_tools import run_shell  # noqa: PLC0415

    calls.append(("os.shell", "echo"))
    return ToolRegistry([run_shell])


def _scoped_manifest() -> WorkflowManifest:
    """One model stage whose tool call is gated by a path-scoped policy."""
    return WorkflowManifest(
        workflow_id="ReplayTape",
        entry_stage_id="M",
        exit_stage_ids=frozenset({"M"}),
        inputs=(),
        capabilities=frozenset({"os.shell"}),
        policies=(
            PolicySpec(
                id="deny os.shell without python",
                kind="deny",
                trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
                condition=ExprSpec(
                    kind="not",
                    expr=ExprSpec(
                        kind="method_call",
                        method="contains",
                        receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
                        args=(ExprSpec(kind="literal", value="python"),),
                    ),
                ),
            ),
        ),
        stages=(
            StageSpec(
                id="M",
                prompt="",
                reads=(),
                writes=(WriteSpec(name="done", type="bool", optional=False),),
                requires=frozenset({"os.shell"}),
                transitions=(),
                execution=StageExecutionSpec(kind="model"),
            ),
        ),
    )


async def test_taped_replay_reproduces_policy_outcomes(tmp_path: Path) -> None:
    """End-to-end: taped policy tape + recorded stub schemas + name map.

    The live model requests ``os.shell`` with a command the policy allows.
    The vault scrubs that command (embedded home path), so naive
    re-evaluation would deny it on replay. The run matches only when the
    runtime reproduces the recorded allow outcome, the stub schema accepts
    the recorded argument shape, and the recorded tool name resolves.
    """

    adapter = _ToolCallAdapter()
    calls: list[tuple[str, Any]] = []
    registry = _scoped_registry(calls)
    runtime = WorkflowRuntime(
        manifest=_scoped_manifest(),
        tools=registry,
        stage_executor=ModelStageExecutor(model=adapter, tools=registry),
    )
    recorder = TraceRecorder.create(
        tmp_path / "scoped.nemotrace",
        profile="replay",
        provenance=HostProvenance(
            frontend="replay-e2e",
            target="python",
            compiler_version="replay-e2e",
            ir_version="0.1",
            ir_sha256="sha256:" + "cd" * 32,
        ),
        path_aliases={"$workspace": Path("/work")},
        safe_path_aliases=frozenset({"$workspace"}),
        vault_passphrase="replay-e2e-passphrase",  # noqa: S106
    )
    result = await runtime.run({}, trace_recorder=recorder)
    assert result.output["done"] is True
    assert len(adapter.calls) == 2
    live_model_calls = len(adapter.calls)
    live_tool_calls = len(calls)

    report = await replay_trace(tmp_path / "scoped.nemotrace", "replay-e2e-passphrase")
    assert report.matched, report.divergences
    assert report.replayed_status == "complete"
    assert report.verification.semantic == "passed"
    assert len(adapter.calls) == live_model_calls
    assert len(calls) == live_tool_calls


def test_fixture_placeholders_cover_marker_types() -> None:
    """Typed placeholders satisfy validation without trusting values."""

    def marker(value_type: str, length: Any = None) -> dict[str, Any]:
        inner: dict[str, Any] = {"token": "r-1", "reason": "x", "value_type": value_type}
        if length is not None:
            inner["length"] = length
        return {"$redacted": inner}

    assert _unmark_fixture(marker("string", 3)) == "???"
    assert _unmark_fixture(marker("string")) == ""
    assert _unmark_fixture(marker("number", 5)) == 0
    assert _unmark_fixture(marker("boolean")) is False
    assert _unmark_fixture(marker("array", 2)) == []
    assert _unmark_fixture(marker("object")) == {}
    assert _unmark_fixture(marker("null")) is None
    nested = {"a": [marker("string", 1)], "b": {"c": marker("number")}, "d": "kept"}
    assert _unmark_fixture(nested) == {"a": ["?"], "b": {"c": 0}, "d": "kept"}
