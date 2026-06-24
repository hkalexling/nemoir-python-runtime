"""Phase 3 cross-language end-to-end test.

Generates the ``coding_agent`` Python package from ``coding-agent.nemo`` via
``nemo compile --target python``, then imports it against the shared
``nemoir-runtime`` and runs the workflow to completion with a scripted
``StageExecutor`` that never touches an LLM.

This proves the generated ``_manifest.py`` is consumable by Phase 2's
``WorkflowRuntime`` end-to-end.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime import Tool, ToolContext, ToolRegistry
from nemoir_runtime.errors import PolicyDeniedError
from nemoir_runtime.models import ModelResponse, ModelStreamChunk, ModelToolCall

REPO_ROOT = Path(__file__).parents[3]
NEMO_BIN = REPO_ROOT / "compiler" / "target" / "debug" / "nemo"
CODING_AGENT_NEMO = REPO_ROOT / "demos" / "coding-agent" / "coding-agent.nemo"
RUNTIME_SRC = REPO_ROOT / "python" / "nemoir-runtime" / "src"


def _nemo_available() -> bool:
    return NEMO_BIN.exists()


pytestmark: Any = pytest.mark.skipif(  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
    not _nemo_available(),
    reason=(
        "requires built `nemo` binary at compiler/target/debug/nemo. "
        "Run `cargo build` in compiler/ first."
    ),
)


# Per-stage scripted outputs. The happy path is
# Triage -> Plan -> Propose(ok=True) -> Apply -> Fin, skipping Clarify and
# Verify (no optional outputs are populated).
_SCRIPTED_OUTPUTS: dict[str, dict[str, Any]] = {
    "Triage": {"summary": "Analyzed the task"},
    "Clarify": {"answers": []},
    "Plan": {"plan": "do the thing"},  # skip Verify by omitting verification_plan
    "Propose": {"ok": True},  # ok=True branches to Apply
    "Apply": {"summary": "applied changes"},
    "Verify": {"ok": True, "summary": "verified"},
    "Fin": {"summary": "done"},
}


class _ScriptedExecutor:
    """Returns a hard-coded valid output for each stage id."""

    def __init__(self) -> None:
        self.stages_seen: list[str] = []

    async def execute(self, ctx: Any) -> dict[str, Any]:
        self.stages_seen.append(ctx.stage.id)
        if ctx.stage.id not in _SCRIPTED_OUTPUTS:
            msg = f"unhandled stage {ctx.stage.id}"
            raise RuntimeError(msg)
        return dict(_SCRIPTED_OUTPUTS[ctx.stage.id])


def _make_tools() -> ToolRegistry:
    async def read_fn(*, path: Path, ctx: ToolContext) -> str:
        return f"read:{path}"

    async def write_fn(*, path: Path, content: str, ctx: ToolContext) -> None:
        return None

    async def confirm_fn(*, message: str, ctx: ToolContext) -> bool:
        return True

    async def shell_fn(*, command: str, ctx: ToolContext) -> str:
        return "ok"

    async def elicit_fn(*, question: str, ctx: ToolContext) -> str:
        return "yes"

    read_tool = Tool(
        name="read_file",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=read_fn,
    )
    write_tool = Tool(
        name="write_file",
        capability="fs.write",
        description="w",
        input_schema={"path": Path, "content": str},
        handler=write_fn,
    )
    confirm_tool = Tool(
        name="confirm",
        capability="user.confirm",
        description="confirm",
        input_schema={"message": str},
        handler=confirm_fn,
    )
    shell_tool = Tool(
        name="shell",
        capability="os.shell",
        description="shell",
        input_schema={"command": str},
        handler=shell_fn,
    )
    elicit_tool = Tool(
        name="elicit",
        capability="user.elicit",
        description="elicit",
        input_schema={"question": str},
        handler=elicit_fn,
    )
    return ToolRegistry([read_tool, write_tool, confirm_tool, shell_tool, elicit_tool])


def _generate_package(out_dir: Path) -> None:
    """Invoke `nemo compile --target python` against coding-agent.nemo."""
    subprocess.run(  # noqa: S603 - binary path is controlled by the test
        [
            str(NEMO_BIN),
            "compile",
            str(CODING_AGENT_NEMO),
            "--target",
            "python",
            "-o",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
    )


def _import_generated_package(out_dir: Path) -> Any:
    # Defensive: clear any previously-cached `coding_agent` modules so each
    # test imports a freshly generated package. Without this, the second
    # `_import_generated_package` call silently reuses the first test's
    # cached module even when `out_dir` differs.
    for mod_name in list(sys.modules):
        if mod_name == "coding_agent" or mod_name.startswith("coding_agent."):
            del sys.modules[mod_name]

    sys.path.insert(0, str(out_dir))
    if str(RUNTIME_SRC) not in sys.path:
        sys.path.insert(0, str(RUNTIME_SRC))
    import coding_agent  # type: ignore[import-not-found,reportMissingImports]  # noqa: PLC0415

    return coding_agent


def test_generated_package_runs_through_phase2_runtime(tmp_path: Path) -> None:
    out_dir = tmp_path / "gen"
    out_dir.mkdir()

    _generate_package(out_dir)

    assert (out_dir / "coding_agent" / "__init__.py").exists()
    assert (out_dir / "coding_agent" / "_manifest.py").exists()
    assert (out_dir / "pyproject.toml").exists()

    coding_agent = _import_generated_package(out_dir)

    assert coding_agent.Agent.workflow_id == "CodingAgent"
    assert coding_agent.Agent.required_capabilities == frozenset(
        {"fs.read", "fs.write", "os.shell", "user.elicit", "user.confirm"}
    )

    agent = coding_agent.Agent(model="bogus-for-phase-3", tools=_make_tools())
    executor = _ScriptedExecutor()
    result = asyncio.run(
        agent._run_with_executor(  # noqa: SLF001 - the test exercises the backdoor
            coding_agent.AgentInput(task="t", cwd=Path("/tmp")),
            executor=executor,
        )
    )

    assert isinstance(result, coding_agent.AgentResult)
    assert isinstance(result.output, coding_agent.AgentOutput)
    assert result.output.summary == "done"
    # Happy path skips Clarify (no unclear_points) and skips Verify
    # (no verification_plan).
    assert executor.stages_seen == ["Triage", "Plan", "Propose", "Apply", "Fin"]


def test_generated_package_run_with_fake_adapter_succeeds(tmp_path: Path) -> None:
    """Generated Agent.run() executes the workflow with a content-only fake adapter.

    The fake adapter returns plain JSON for every stage. The coding-agent
    happy path is Triage -> Plan -> Propose -> Apply -> Fin (5 stages).
    The last stage (Fin) writes summary, so the final AgentOutput.summary
    matches what the fake adapter returned.
    """
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)

    coding_agent = _import_generated_package(out_dir)

    from nemoir_runtime.models import ModelResponse  # noqa: PLC0415

    class StageAwareAdapter:
        """Returns stage-appropriate content-only responses.

        The coding-agent stages write different output fields, so we need
        valid per-stage JSON responses.
        """

        def __init__(self) -> None:
            self.calls: list[Any] = []
            self._responses: dict[str, ModelResponse] = {
                "Triage": ModelResponse(content='{"summary": "triage-done"}'),
                "Plan": ModelResponse(content='{"plan": "the plan"}'),
                "Propose": ModelResponse(content='{"ok": true}'),
                "Apply": ModelResponse(content='{"summary": "apply-done"}'),
                "Fin": ModelResponse(content='{"summary": "fin-done"}'),
            }

        async def complete(self, request: Any) -> ModelResponse:
            self.calls.append(request)
            stage_id = request.stage_id
            return self._responses.get(stage_id, ModelResponse(content='{"summary": "unknown"}'))

    fake = StageAwareAdapter()
    agent = coding_agent.Agent(model=fake, tools=_make_tools())
    result = asyncio.run(agent.run(coding_agent.AgentInput(task="t", cwd=Path("/tmp"))))

    assert isinstance(result, coding_agent.AgentResult)
    assert isinstance(result.output, coding_agent.AgentOutput)
    assert result.output.summary == "fin-done"
    assert len(fake.calls) > 0


def test_generated_package_run_with_model_router(tmp_path: Path) -> None:
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)

    coding_agent = _import_generated_package(out_dir)

    from nemoir_runtime import ModelRouter  # noqa: PLC0415
    from nemoir_runtime.models import ModelAdapter, ModelRequest, ModelResponse  # noqa: PLC0415

    class StageAwareAdapter(ModelAdapter):
        def __init__(self) -> None:
            self.calls: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.calls.append(request)
            responses: dict[str, str] = {
                "Triage": '{"summary": "triage"}',
                "Plan": '{"plan": "the plan"}',
                "Propose": '{"ok": true}',
                "Apply": '{"summary": "apply"}',
                "Fin": '{"summary": "router-done"}',
            }
            return ModelResponse(content=responses.get(request.stage_id, '{"summary": "ok"}'))

    mock_default = StageAwareAdapter()
    mock_plan = StageAwareAdapter()
    router = ModelRouter(default=mock_default, stages={"Plan": mock_plan})
    agent = coding_agent.Agent(model=router, tools=_make_tools())
    result = asyncio.run(agent.run(coding_agent.AgentInput(task="t", cwd=Path("/tmp"))))
    assert result.output.summary == "router-done"
    assert len(mock_default.calls) > 0


class _TriageEmptyArrayExecutor:
    """Regression: Triage returns unclear_points=[] (empty optional array).

    This must normalize to None so the has_value guard evaluates False
    and the workflow proceeds to Plan, not Clarify.
    """

    def __init__(self) -> None:
        self.stages_seen: list[str] = []

    async def execute(self, ctx: Any) -> dict[str, Any]:
        self.stages_seen.append(ctx.stage.id)
        if ctx.stage.id == "Triage":
            return {"summary": "clear", "unclear_points": []}
        if ctx.stage.id == "Plan":
            return {"plan": "the plan"}
        if ctx.stage.id == "Propose":
            return {"ok": True}
        if ctx.stage.id == "Apply":
            return {"summary": "applied"}
        if ctx.stage.id == "Fin":
            return {"summary": "done"}
        if ctx.stage.id == "Clarify":
            return {"answers": []}
        msg = f"unhandled stage {ctx.stage.id}"
        raise RuntimeError(msg)


def test_empty_optional_array_skips_clarify(tmp_path: Path) -> None:
    """Triage with unclear_points=[] must skip Clarify (empty array -> None).

    Regression: before the fix, unclear_points=[] was treated as "has value"
    causing an infinite Triage<->Clarify loop.
    """
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)
    coding_agent = _import_generated_package(out_dir)

    executor = _TriageEmptyArrayExecutor()
    agent = coding_agent.Agent(model="bogus", tools=_make_tools())
    result = asyncio.run(
        agent._run_with_executor(  # noqa: SLF001
            coding_agent.AgentInput(task="t", cwd=Path("/tmp")),
            executor=executor,
        )
    )
    assert result.output.summary == "done"
    # Must skip Clarify entirely.
    assert "Clarify" not in executor.stages_seen
    assert "Plan" in executor.stages_seen
    assert executor.stages_seen == ["Triage", "Plan", "Propose", "Apply", "Fin"]


def test_generated_package_run_with_executor_backdoor_still_works(tmp_path: Path) -> None:
    """_run_with_executor backdoor remains intact."""
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)
    coding_agent = _import_generated_package(out_dir)

    executor = _ScriptedExecutor()
    agent = coding_agent.Agent(model="bogus", tools=_make_tools())
    result = asyncio.run(
        agent._run_with_executor(  # noqa: SLF001
            coding_agent.AgentInput(task="t", cwd=Path("/tmp")),
            executor=executor,
        )
    )
    assert isinstance(result, coding_agent.AgentResult)
    assert result.output.summary == "done"


class _PolicyExercisingExecutor:
    """Walks Triage -> Plan -> Propose(ok=True) -> Apply(fs.write) -> Fin.

    Apply's ``fs.write`` call triggers the
    ``before fs.write(path) requires fs.read(path), user.confirm`` policy
    chain, exercising all three policy-gated capabilities in one stage.
    """

    def __init__(self) -> None:
        self.stages_seen: list[str] = []

    async def execute(self, ctx: Any) -> dict[str, Any]:
        self.stages_seen.append(ctx.stage.id)
        if ctx.stage.id == "Triage":
            # fs.read with cwd=/tmp and path=/tmp/README.md is inside cwd,
            # so the deny policy should allow the call.
            await ctx.call_tool("fs.read", {"path": Path("/tmp/README.md")})
            return {"summary": "triaged"}
        if ctx.stage.id == "Plan":
            # No verification_plan -> Apply's `missing` guard fires and we
            # skip Verify (returns directly to Fin through Apply).
            return {"plan": "do the thing"}
        if ctx.stage.id == "Propose":
            # ok=True branches to Apply.
            return {"ok": True}
        if ctx.stage.id == "Apply":
            # fs.write triggers the `before fs.write(path) requires
            # fs.read(path), user.confirm` policy chain: the runtime first
            # calls fs.read with the bound `path`, then user.confirm, then
            # fs.write proper. All three happen inside this single tool call.
            await ctx.call_tool(
                "fs.write",
                {"path": Path("/tmp/changes.txt"), "content": "diff"},
            )
            return {"summary": "applied"}
        if ctx.stage.id == "Fin":
            return {"summary": "done"}
        msg = f"unhandled stage {ctx.stage.id}"
        raise RuntimeError(msg)


def test_generated_package_policy_gated_tool_calls_succeed(tmp_path: Path) -> None:
    # F1 regression: this test exercises the generated `coding_agent` package's
    # deny and before policies through real `ctx.call_tool` invocations.
    # Before F1, evaluating `deny fs.read(path) if not cwd.contains(path)`
    # raised `TypeError: 'ExprSpec' object is not iterable` because the
    # single-arg `cwd.contains(path)` method call was emitted as
    # `args=(ExprSpec(...))` (a parenthesized ExprSpec, not a tuple).
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)
    coding_agent = _import_generated_package(out_dir)

    tool_calls: list[tuple[str, dict[str, Any]]] = []

    async def read_recorder(*, path: Path, ctx: ToolContext) -> str:
        tool_calls.append(("fs.read", {"path": str(path)}))
        return "ok"

    async def write_recorder(*, path: Path, content: str, ctx: ToolContext) -> None:
        tool_calls.append(("fs.write", {"path": str(path), "content": content}))

    async def confirm_recorder(*, message: str, ctx: ToolContext) -> bool:
        tool_calls.append(("user.confirm", {"message": message}))
        return True

    async def elicit_recorder(*, question: str, ctx: ToolContext) -> str:
        return "y"

    async def shell_recorder(*, command: str, ctx: ToolContext) -> str:
        return "ok"

    tools = ToolRegistry(
        [
            Tool(
                name="read",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=read_recorder,
            ),
            Tool(
                name="write",
                capability="fs.write",
                description="w",
                input_schema={"path": Path, "content": str},
                handler=write_recorder,
            ),
            Tool(
                name="confirm",
                capability="user.confirm",
                description="c",
                input_schema={"message": str},
                handler=confirm_recorder,
            ),
            Tool(
                name="elicit",
                capability="user.elicit",
                description="e",
                input_schema={"question": str},
                handler=elicit_recorder,
            ),
            Tool(
                name="shell",
                capability="os.shell",
                description="s",
                input_schema={"command": str},
                handler=shell_recorder,
            ),
        ]
    )

    agent = coding_agent.Agent(model="bogus", tools=tools)
    executor = _PolicyExercisingExecutor()
    result = asyncio.run(
        agent._run_with_executor(  # noqa: SLF001 - tests the backdoor
            coding_agent.AgentInput(task="t", cwd=Path("/tmp")),
            executor=executor,
        )
    )

    assert result.output.summary == "done"
    assert executor.stages_seen == ["Triage", "Plan", "Propose", "Apply", "Fin"]

    # Triage: 1x fs.read (Triage's own call).
    # Apply: 3x fs.write-policy chain (fs.read, user.confirm, fs.write proper).
    assert tool_calls[0] == ("fs.read", {"path": "/tmp/README.md"})
    assert tool_calls[1] == ("fs.read", {"path": "/tmp/changes.txt"})  # before: fs.read
    assert tool_calls[2][0] == "user.confirm"  # before: user.confirm
    assert tool_calls[3] == (
        "fs.write",
        {"path": "/tmp/changes.txt", "content": "diff"},
    )


def test_generated_package_agent_run_policy_gated_tool_calls(tmp_path: Path) -> None:
    """Agent.run() exercises model-requested policy-gated tool calls.

    Uses a fake adapter that returns fs.read for Triage and fs.write for
    Apply, so the ModelStageExecutor tool-call loop drives the before-policy
    chain through ctx.call_tool().
    """
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)

    coding_agent = _import_generated_package(out_dir)

    from nemoir_runtime.models import ModelResponse, ModelToolCall  # noqa: PLC0415

    class PolicyChainAdapter:
        """Returns tool calls for Triage/Apply, else stage-appropriate JSON."""

        def __init__(self) -> None:
            self.calls: list[Any] = []
            self._tool_call_made: set[str] = set()

        async def complete(self, request: Any) -> ModelResponse:
            self.calls.append(request)
            stage_id = request.stage_id

            if stage_id == "Triage" and "Triage" not in self._tool_call_made:
                self._tool_call_made.add("Triage")
                return ModelResponse(
                    content=None,
                    tool_calls=(
                        ModelToolCall(
                            id="call_t1",
                            name="read",
                            arguments={"path": "/tmp/README.md"},
                        ),
                    ),
                )
            if stage_id == "Apply" and "Apply" not in self._tool_call_made:
                self._tool_call_made.add("Apply")
                return ModelResponse(
                    content=None,
                    tool_calls=(
                        ModelToolCall(
                            id="call_a1",
                            name="write_file",
                            arguments={"path": "/tmp/changes.txt", "content": "diff"},
                        ),
                    ),
                )

            outputs: dict[str, str] = {
                "Triage": '{"summary": "triaged"}',
                "Plan": '{"plan": "do the thing"}',
                "Propose": '{"ok": true}',
                "Apply": '{"summary": "applied"}',
                "Fin": '{"summary": "done"}',
            }
            return ModelResponse(content=outputs.get(stage_id, '{"summary": "ok"}'))

    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    async def read_recorder(*, path: Path, ctx: ToolContext) -> str:
        tool_calls_log.append(("fs.read", {"path": str(path)}))
        return "ok"

    async def write_recorder(*, path: Path, content: str, ctx: ToolContext) -> None:
        tool_calls_log.append(("fs.write", {"path": str(path), "content": content}))

    async def confirm_recorder(*, message: str, ctx: ToolContext) -> bool:
        tool_calls_log.append(("user.confirm", {"message": message}))
        return True

    async def shell_recorder(*, command: str, ctx: ToolContext) -> str:
        return "ok"

    async def elicit_recorder(*, question: str, ctx: ToolContext) -> str:
        return "y"

    tools = ToolRegistry(
        [
            Tool(
                name="read",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=read_recorder,
            ),
            Tool(
                name="write_file",
                capability="fs.write",
                description="w",
                input_schema={"path": Path, "content": str},
                handler=write_recorder,
            ),
            Tool(
                name="confirm",
                capability="user.confirm",
                description="c",
                input_schema={"message": str},
                handler=confirm_recorder,
            ),
            Tool(
                name="elicit",
                capability="user.elicit",
                description="e",
                input_schema={"question": str},
                handler=elicit_recorder,
            ),
            Tool(
                name="shell",
                capability="os.shell",
                description="s",
                input_schema={"command": str},
                handler=shell_recorder,
            ),
        ]
    )

    agent = coding_agent.Agent(model=PolicyChainAdapter(), tools=tools)
    result = asyncio.run(agent.run(coding_agent.AgentInput(task="t", cwd=Path("/tmp"))))

    assert isinstance(result, coding_agent.AgentResult)
    assert isinstance(result.output, coding_agent.AgentOutput)
    assert result.output.summary == "done"

    # Triage: 1x fs.read (model-requested).
    # Apply: before-policy chain (fs.read + user.confirm) then fs.write.
    assert tool_calls_log[0] == ("fs.read", {"path": "/tmp/README.md"})
    assert tool_calls_log[1] == ("fs.read", {"path": "/tmp/changes.txt"})
    assert tool_calls_log[2][0] == "user.confirm"
    assert tool_calls_log[3] == (
        "fs.write",
        {"path": "/tmp/changes.txt", "content": "diff"},
    )
    assert len(tool_calls_log) == 4


# ------------------------------------------------------------------
# Phase 5: generated Agent.stream() integration tests
# ------------------------------------------------------------------


class _StreamingFakeAdapter:
    """Fake adapter with both complete() and stream() for generated-package tests."""

    def __init__(self, stage_responses: dict[str, str] | None = None) -> None:
        self.calls: list[Any] = []
        self._responses = stage_responses or {
            "Triage": '{"summary": "triaged"}',
            "Plan": '{"plan": "the plan"}',
            "Propose": '{"ok": true}',
            "Apply": '{"summary": "applied"}',
            "Fin": '{"summary": "streaming-done"}',
        }

    async def complete(self, request: Any) -> ModelResponse:
        self.calls.append(request)
        content = self._responses.get(request.stage_id, '{"summary": "ok"}')
        return ModelResponse(content=content)

    async def stream(self, request: Any) -> Any:
        self.calls.append(request)
        content = self._responses.get(request.stage_id, '{"summary": "ok"}')
        # Emit content as deltas then a completed chunk.
        words = content.split()
        for i, word in enumerate(words):
            text = word if i == 0 else " " + word
            yield ModelStreamChunk(kind="delta", channel="assistant", text=text)
        yield ModelStreamChunk(kind="completed", response=ModelResponse(content=content))


class _ToolCallStreamingAdapter:
    """Streaming adapter that returns tool calls for Triage + Apply."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self._tool_call_made: set[str] = set()

    async def complete(self, request: Any) -> ModelResponse:
        self.calls.append(request)
        return self._make_response(request.stage_id)

    async def stream(self, request: Any) -> Any:
        self.calls.append(request)
        resp = self._make_response(request.stage_id)
        # Yield a single completed chunk with the full response.
        yield ModelStreamChunk(kind="completed", response=resp)

    def _make_response(self, stage_id: str) -> ModelResponse:
        if stage_id == "Triage" and "Triage" not in self._tool_call_made:
            self._tool_call_made.add("Triage")
            return ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(id="call_s1", name="read", arguments={"path": "/tmp/README.md"}),
                ),
            )
        if stage_id == "Apply" and "Apply" not in self._tool_call_made:
            self._tool_call_made.add("Apply")
            return ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="call_s2",
                        name="write_file",
                        arguments={"path": "/tmp/changes.txt", "content": "diff"},
                    ),
                ),
            )
        outputs: dict[str, str] = {
            "Triage": '{"summary": "triaged"}',
            "Plan": '{"plan": "do the thing"}',
            "Propose": '{"ok": true}',
            "Apply": '{"summary": "applied"}',
            "Fin": '{"summary": "s-done"}',
        }
        return ModelResponse(content=outputs.get(stage_id, '{"summary": "ok"}'))


def test_generated_package_stream_yields_events(tmp_path: Path) -> None:
    """Agent.stream() yields lifecycle + model_delta events with typed result."""
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)
    coding_agent = _import_generated_package(out_dir)

    adapter = _StreamingFakeAdapter()
    agent = coding_agent.Agent(model=adapter, tools=_make_tools())

    events: list[Any] = []  # WorkflowEvent — dynamic import, typed as Any

    async def collect() -> None:
        async for event in agent.stream(coding_agent.AgentInput(task="t", cwd=Path("/tmp"))):
            events.append(event)

    asyncio.run(collect())

    kinds = [e.kind for e in events]
    assert "run_started" in kinds
    assert "stage_started" in kinds
    assert "model_delta" in kinds
    assert "model_completed" in kinds
    assert "stage_completed" in kinds
    assert "run_completed" in kinds

    # Typed run_completed result.
    rc = [e for e in events if e.kind == "run_completed"]
    assert len(rc) == 1
    assert isinstance(rc[0].result, coding_agent.AgentResult)
    assert isinstance(rc[0].result.output, coding_agent.AgentOutput)
    assert rc[0].result.output.summary == "streaming-done"

    # At least one model_delta on the assistant channel.
    deltas = [e for e in events if e.kind == "model_delta"]
    assert len(deltas) >= 1
    assistant_deltas = [d for d in deltas if d.channel == "assistant"]
    assert len(assistant_deltas) >= 1


def test_generated_package_stream_policy_required_tool_events(tmp_path: Path) -> None:
    """Agent.stream() includes policy-required tool events (High-1 regression)."""
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)
    coding_agent = _import_generated_package(out_dir)

    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    async def read_recorder(*, path: Path, ctx: ToolContext) -> str:
        tool_calls_log.append(("fs.read", {"path": str(path)}))
        return "ok"

    async def write_recorder(*, path: Path, content: str, ctx: ToolContext) -> None:
        tool_calls_log.append(("fs.write", {"path": str(path), "content": content}))

    async def confirm_recorder(*, message: str, ctx: ToolContext) -> bool:
        tool_calls_log.append(("user.confirm", {"message": message}))
        return True

    async def shell_recorder(*, command: str, ctx: ToolContext) -> str:
        return "ok"

    async def elicit_recorder(*, question: str, ctx: ToolContext) -> str:
        return "y"

    tools = ToolRegistry(
        [
            Tool(
                name="read",
                capability="fs.read",
                description="r",
                input_schema={"path": Path},
                handler=read_recorder,
            ),
            Tool(
                name="write_file",
                capability="fs.write",
                description="w",
                input_schema={"path": Path, "content": str},
                handler=write_recorder,
            ),
            Tool(
                name="confirm",
                capability="user.confirm",
                description="c",
                input_schema={"message": str},
                handler=confirm_recorder,
            ),
            Tool(
                name="elicit",
                capability="user.elicit",
                description="e",
                input_schema={"question": str},
                handler=elicit_recorder,
            ),
            Tool(
                name="shell",
                capability="os.shell",
                description="s",
                input_schema={"command": str},
                handler=shell_recorder,
            ),
        ]
    )

    adapter = _ToolCallStreamingAdapter()
    agent = coding_agent.Agent(model=adapter, tools=tools)

    events: list[Any] = []  # WorkflowEvent — dynamic import, typed as Any

    async def collect() -> None:
        async for event in agent.stream(coding_agent.AgentInput(task="t", cwd=Path("/tmp"))):
            events.append(event)

    asyncio.run(collect())

    # Triage: 1x fs.read (model-requested via tool_calls).
    # Apply: policy-required chain (fs.read + user.confirm) then fs.write proper.
    assert len(tool_calls_log) == 4
    assert tool_calls_log[0] == ("fs.read", {"path": "/tmp/README.md"})
    assert tool_calls_log[1] == ("fs.read", {"path": "/tmp/changes.txt"})
    assert tool_calls_log[2][0] == "user.confirm"
    assert tool_calls_log[3] == (
        "fs.write",
        {"path": "/tmp/changes.txt", "content": "diff"},
    )

    # Check that policy-required tool_call_started events are in the stream.
    tool_call_started_names = [e.tool_name for e in events if e.kind == "tool_call_started"]
    # The list should include: read (Triage), read (before-policy),
    # confirm (before-policy), write_file (Apply).
    assert tool_call_started_names.count("read") == 2, (
        f"expected 2 read tool_call_started events "
        f"(Triage + before-policy), got: {tool_call_started_names}"
    )
    assert "confirm" in tool_call_started_names, (
        f"expected confirm tool_call_started event (before-policy), got: {tool_call_started_names}"
    )
    assert "write_file" in tool_call_started_names

    tool_call_completed_names = [e.tool_name for e in events if e.kind == "tool_call_completed"]
    assert tool_call_completed_names.count("read") == 2
    assert "confirm" in tool_call_completed_names
    assert "write_file" in tool_call_completed_names


def test_nemo_binary_is_present() -> None:
    # Sanity guard: ensures the skip guard is exercised even when the runtime
    # import succeeds, and documents the prerequisite path.
    assert _nemo_available(), (
        f"expected `nemo` binary at {NEMO_BIN}; run `cargo build` in compiler/ first"
    )
    assert shutil.which("python3") is not None, "python3 not found on PATH"


# ------------------------------------------------------------------
# Plan Medium-Tests-1: official tools wired through generated package
# ------------------------------------------------------------------


def test_generated_package_run_with_official_tools(tmp_path: Path) -> None:
    """Generated Agent.run() succeeds with all six official tools registered."""
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)

    coding_agent = _import_generated_package(out_dir)

    from nemoir_runtime.official_tools import (  # noqa: PLC0415
        ask_user,
        confirm_user,
        edit_file,
        read_file,
        run_shell,
        write_file,
    )

    official_tools = ToolRegistry(
        [read_file, write_file, edit_file, run_shell, ask_user, confirm_user]
    )

    class StageAwareAdapter:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def _output_for(self, stage_id: str) -> str:
            outputs: dict[str, str] = {
                "Triage": '{"summary": "triaged"}',
                "Plan": '{"plan": "do the thing"}',
                "Propose": '{"ok": true}',
                "Apply": '{"summary": "applied"}',
                "Fin": '{"summary": "official-tools-done"}',
            }
            return outputs.get(stage_id, '{"summary": "ok"}')

        async def complete(self, request: Any) -> ModelResponse:
            self.calls.append(request)
            return ModelResponse(content=self._output_for(request.stage_id))

    fake = StageAwareAdapter()
    agent = coding_agent.Agent(model=fake, tools=official_tools)
    result = asyncio.run(agent.run(coding_agent.AgentInput(task="t", cwd=Path("/tmp"))))
    assert isinstance(result, coding_agent.AgentResult)
    assert isinstance(result.output, coding_agent.AgentOutput)
    assert result.output.summary == "official-tools-done"
    # Happy path: Triage → Plan → Propose → Apply → Fin (5 stages).
    assert len(fake.calls) == 5


def test_generated_package_stream_with_official_tools(tmp_path: Path) -> None:
    """Agent.stream() yields events with official tools in registry."""
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)

    coding_agent = _import_generated_package(out_dir)

    from nemoir_runtime.official_tools import (  # noqa: PLC0415
        ask_user,
        confirm_user,
        edit_file,
        read_file,
        run_shell,
        write_file,
    )

    official_tools = ToolRegistry(
        [read_file, write_file, edit_file, run_shell, ask_user, confirm_user]
    )

    stage_responses = {
        "Triage": '{"summary": "triaged"}',
        "Plan": '{"plan": "do the thing"}',
        "Propose": '{"ok": true}',
        "Apply": '{"summary": "applied"}',
        "Fin": '{"summary": "official-stream-done"}',
    }

    class StreamingAdapter:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def complete(self, request: Any) -> ModelResponse:
            self.calls.append(request)
            content = stage_responses.get(request.stage_id, '{"summary": "ok"}')
            return ModelResponse(content=content)

        async def stream(self, request: Any) -> Any:
            self.calls.append(request)
            content = stage_responses.get(request.stage_id, '{"summary": "ok"}')
            yield ModelStreamChunk(kind="delta", channel="assistant", text=content)
            yield ModelStreamChunk(kind="completed", response=ModelResponse(content=content))

    adapter = StreamingAdapter()
    agent = coding_agent.Agent(model=adapter, tools=official_tools)

    events: list[Any] = []

    async def collect() -> None:
        async for event in agent.stream(coding_agent.AgentInput(task="t", cwd=Path("/tmp"))):
            events.append(event)

    asyncio.run(collect())

    kinds = [e.kind for e in events]
    assert "run_started" in kinds
    assert "stage_started" in kinds
    assert "model_delta" in kinds
    assert "model_completed" in kinds
    assert "stage_completed" in kinds
    assert "run_completed" in kinds

    rc = [e for e in events if e.kind == "run_completed"]
    assert len(rc) == 1
    assert isinstance(rc[0].result, coding_agent.AgentResult)
    assert rc[0].result.output.summary == "official-stream-done"


def test_generated_package_stream_reasoning_channel(tmp_path: Path) -> None:  # noqa: C901
    """Generated Agent.stream() forwards 'reasoning' channel model_delta events."""
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    _generate_package(out_dir)

    coding_agent = _import_generated_package(out_dir)

    stage_responses = {
        "Triage": '{"summary": "triaged"}',
        "Plan": '{"plan": "do the thing"}',
        "Propose": '{"ok": true}',
        "Apply": '{"summary": "applied"}',
        "Fin": '{"summary": "reasoning-stream-done"}',
    }

    class StreamingReasoningAdapter:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def complete(self, request: Any) -> ModelResponse:
            self.calls.append(request)
            content = stage_responses.get(request.stage_id, '{"summary": "ok"}')
            return ModelResponse(content=content)

        async def stream(self, request: Any) -> Any:
            self.calls.append(request)
            content = stage_responses.get(request.stage_id, '{"summary": "ok"}')
            yield ModelStreamChunk(kind="delta", channel="reasoning", text="thinking...")
            yield ModelStreamChunk(kind="delta", channel="assistant", text=content)
            yield ModelStreamChunk(kind="completed", response=ModelResponse(content=content))

    adapter = StreamingReasoningAdapter()

    # coding_agent requires several capabilities; provide minimal fake tools.
    async def fake_elicit(*, question: str, ctx: ToolContext) -> str:
        return "fake answer"

    async def fake_confirm(*, message: str, ctx: ToolContext) -> bool:
        return True

    async def fake_read(*, path: Path, ctx: ToolContext) -> str:
        return "fake content"

    async def fake_write(*, path: Path, content: str, ctx: ToolContext) -> None:
        return None

    async def fake_shell(*, command: str, ctx: ToolContext) -> str:
        return "fake output"

    tools = ToolRegistry(
        [
            Tool(
                name="fake_elicit",
                capability="user.elicit",
                description="f",
                input_schema={"question": str},
                handler=fake_elicit,
            ),
            Tool(
                name="fake_confirm",
                capability="user.confirm",
                description="f",
                input_schema={"message": str},
                handler=fake_confirm,
            ),
            Tool(
                name="fake_read",
                capability="fs.read",
                description="f",
                input_schema={"path": Path},
                handler=fake_read,
            ),
            Tool(
                name="fake_write",
                capability="fs.write",
                description="f",
                input_schema={"path": Path, "content": str},
                handler=fake_write,
            ),
            Tool(
                name="fake_shell",
                capability="os.shell",
                description="f",
                input_schema={"command": str},
                handler=fake_shell,
            ),
        ]
    )
    agent = coding_agent.Agent(model=adapter, tools=tools)

    events: list[Any] = []

    async def collect() -> None:
        async for event in agent.stream(coding_agent.AgentInput(task="t", cwd=Path("/tmp"))):
            events.append(event)

    asyncio.run(collect())

    deltas = [e for e in events if e.kind == "model_delta"]
    assert len(deltas) >= 1

    reasoning_deltas = [d for d in deltas if d.channel == "reasoning"]
    assert len(reasoning_deltas) >= 1
    assert reasoning_deltas[0].text == "thinking..."

    assistant_deltas = [d for d in deltas if d.channel == "assistant"]
    assert len(assistant_deltas) >= 1

    rc = [e for e in events if e.kind == "run_completed"]
    assert len(rc) == 1
    assert rc[0].result.output.summary == "reasoning-stream-done"


# ---------------------------------------------------------------------------
# Policy predicate generated-package test (Phase 6 / review follow-up)
# ---------------------------------------------------------------------------

def _compile_and_import(nemo_path: Path, package_name: str, out_dir: Path) -> Any:
    """Compile a .nemo file and import the generated Python package."""
    result = subprocess.run(  # noqa: S603
        [str(NEMO_BIN), "compile", str(nemo_path), "--target", "python", "-o", str(out_dir)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode() if result.stderr else "(no stderr)\n")
        sys.stderr.write(result.stdout.decode() if result.stdout else "(no stdout)\n")
        result.check_returncode()
    # Clear any previously-cached module.
    for mod_name in list(sys.modules):
        if mod_name == package_name or mod_name.startswith(f"{package_name}."):
            del sys.modules[mod_name]
    sys.path.insert(0, str(out_dir))
    if str(RUNTIME_SRC) not in sys.path:
        sys.path.insert(0, str(RUNTIME_SRC))
    return __import__(package_name)  # type: ignore[reportUnknownVariableType]


def test_generated_package_policy_literal_whitespace_preserved(tmp_path: Path) -> None:  # noqa: C901
    """Compile a policy-allowlist workflow and assert trailing-space literals.

    Regression test for the {process_string} trimming bug: policy string
    literals used by {command.starts_with} and {command.contains} must
    preserve significant leading/trailing whitespace.
    """
    nemo_src = dedent("""\
    workflow TestAllowlist {
      input {
        candidate_path: path
      }

      policy {
        deny os.shell(command) if not (
          command.eq("python harness/preflight.py")
          or command.starts_with("python harness/run_trial.py ")
          or command.starts_with("git commit -m ")
        )
        deny os.shell(command) if command.contains("&&") or command.contains("; ")

        deny fs.write(path) if not path.eq(candidate_path)
        deny fs.write(path) if path.eq("harness/eval.py") or path.eq("harness/train.py")
      }

      stage@entry Start {
        prompt: "start"
        requires: os.shell, fs.write
        output: { x: string }
      }
      stage@exit Fin {
        prompt: "done"
        input: Start.x
        output: { summary: string }
      }
    }
    """)
    nemo_path = tmp_path / "test_allowlist.nemo"
    nemo_path.write_text(nemo_src)

    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    pkg = _compile_and_import(nemo_path, "test_allowlist", out_dir)

    manifest = pkg._manifest.WORKFLOW_MANIFEST  # noqa: SLF001  # type: ignore[reportUnknownVariableType]

    # Find the os.shell allowlist policy and assert literal values
    shell_policy = None
    for p in manifest.policies:
        if p.trigger.capability == "os.shell" and "starts_with" in p.id:
            shell_policy = p
            break
    assert shell_policy is not None, "os.shell allowlist policy not found"

    # Collect all string literals from the condition tree
    def collect_literals(expr: Any) -> list[Any]:  # type: ignore[reportUnknownVariableType]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if expr.kind == "literal" and expr.type == "string":
            return [expr.value]
        if expr.kind == "or":
            result: list[Any] = []
            for e in expr.exprs:
                result.extend(collect_literals(e))  # pyright: ignore[reportUnknownMemberType]
            return result
        if expr.kind == "not":
            return collect_literals(expr.expr) if expr.expr else []
        if expr.kind == "method_call":
            result: list[Any] = []
            if expr.receiver:
                result.extend(collect_literals(expr.receiver))  # pyright: ignore[reportUnknownMemberType]
            for a in expr.args:
                result.extend(collect_literals(a))  # pyright: ignore[reportUnknownMemberType]
            return result
        return []

    literals = collect_literals(shell_policy.condition)
    # The shell allowlist condition should contain these exact literals:
    assert "python harness/preflight.py" in literals
    assert "python harness/run_trial.py " in literals, (
        f"trailing space preserved: {literals}"
    )
    assert "git commit -m " in literals, (
        f"trailing space preserved: {literals}"
    )

    # Find the metacharacter deny policy
    meta_policy = None
    for p in manifest.policies:
        if p.trigger.capability == "os.shell" and "contains" in p.id:
            meta_policy = p
            break
    assert meta_policy is not None, "metacharacter deny policy not found"
    meta_literals = collect_literals(meta_policy.condition)
    assert "; " in meta_literals, f"trailing space preserved: {meta_literals}"
    assert "&&" in meta_literals


# ------------------------------------------------------------------
# End-to-end enforcement test (review Medium/Tests 1)
# ------------------------------------------------------------------


class _PolicyEnforcementExecutor:
    """Stage-aware executor for exercising allow/deny policy enforcement."""

    def __init__(
        self,
        start_cmd: str | None = None,
        error_on_start: type[BaseException] | None = None,
    ) -> None:
        self.start_cmd = start_cmd
        self.error_on_start = error_on_start
        self.stages_seen: list[str] = []  # type: ignore[reportUnknownVariableType]

    async def execute(self, ctx: Any) -> dict[str, Any]:
        self.stages_seen.append(ctx.stage.id)
        if ctx.stage.id == "Start" and self.start_cmd is not None:
            try:
                await ctx.call_tool("os.shell", {"command": self.start_cmd})
            except BaseException as exc:
                if self.error_on_start is not None and isinstance(exc, self.error_on_start):
                    raise
                raise
            return {"x": "done"}
        if ctx.stage.id == "Fin":
            return {"summary": "done"}
        return {"x": "done"}


def test_generated_package_policy_enforcement_allowed(tmp_path: Path) -> None:
    """Allowed shell command passes policy and completes the workflow."""
    nemo_path = (
        REPO_ROOT / "compiler" / "crates" / "nemoir-dsl-fe"
        / "tests" / "fixtures" / "policy_command_allowlist.nemo"
    )
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    pkg = _compile_and_import(nemo_path, "test_allowlist", out_dir)

    executor = _PolicyEnforcementExecutor(start_cmd="python harness/preflight.py")
    agent = pkg.Agent(model="bogus", tools=_make_tools())
    result = asyncio.run(
        agent._run_with_executor(  # noqa: SLF001
            pkg.AgentInput(cwd=Path("/tmp/work"), candidate_path=Path("/tmp/work/candidate.py")),
            executor=executor,
        )
    )
    assert isinstance(result, pkg.AgentResult)
    assert result.output.summary == "done"


def test_generated_package_policy_enforcement_denied_shell(tmp_path: Path) -> None:
    """Denied shell command raises PolicyDeniedError and emits a policy_denied event."""
    nemo_path = (
        REPO_ROOT / "compiler" / "crates" / "nemoir-dsl-fe"
        / "tests" / "fixtures" / "policy_command_allowlist.nemo"
    )
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    pkg = _compile_and_import(nemo_path, "test_allowlist", out_dir)

    executor = _PolicyEnforcementExecutor(
        start_cmd="rm -rf /",
        error_on_start=PolicyDeniedError,
    )
    agent = pkg.Agent(model="bogus", tools=_make_tools())

    events: list[Any] = []

    async def collect_and_run() -> None:
        try:
            target_input = pkg.AgentInput(
                cwd=Path("/tmp/work"), candidate_path=Path("/tmp/work/candidate.py")
            )
            async for ev in agent._stream_with_executor(  # noqa: SLF001
                target_input,
                executor=executor,
            ):
                events.append(ev)
        except PolicyDeniedError:
            pass

    asyncio.run(collect_and_run())
    assert any(e.kind == "policy_denied" for e in events), (
        f"expected policy_denied event in {[e.kind for e in events]}"
    )


def test_generated_package_policy_enforcement_denied_write(tmp_path: Path) -> None:
    """Write outside the allowlist (harness/eval.py) raises PolicyDeniedError."""
    nemo_path = (
        REPO_ROOT / "compiler" / "crates" / "nemoir-dsl-fe"
        / "tests" / "fixtures" / "policy_command_allowlist.nemo"
    )
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    pkg = _compile_and_import(nemo_path, "test_allowlist", out_dir)

    class WriteDenyExecutor:
        def __init__(self) -> None:
            self.stages_seen: list[str] = []  # type: ignore[reportUnknownVariableType]

        async def execute(self, ctx: Any) -> dict[str, Any]:
            self.stages_seen.append(ctx.stage.id)  # pyright: ignore[reportUnknownMemberType]
            if ctx.stage.id == "Start":
                await ctx.call_tool(
                    "fs.write",
                    {"path": Path("harness/eval.py"), "content": "x"},
                )
                return {"x": "done"}
            if ctx.stage.id == "Fin":
                return {"summary": "done"}
            return {"x": "done"}

    agent = pkg.Agent(model="bogus", tools=_make_tools())
    denied_input = pkg.AgentInput(
        cwd=Path("/tmp/work"), candidate_path=Path("/tmp/work/candidate.py")
    )
    with pytest.raises(PolicyDeniedError, match="denied"):
        asyncio.run(agent._run_with_executor(  # noqa: SLF001
            denied_input, executor=WriteDenyExecutor(),
        ))


# ------------------------------------------------------------------
# Deterministic stage: generated-package e2e (Medium #2)
# ------------------------------------------------------------------


def test_deterministic_package_runs_with_official_tool(tmp_path: Path) -> None:
    """Compile exec_shell.nemo → run Agent with official run_shell tool.

    Verifies: tool_call_started/completed events fire, no model_completed
    on the deterministic stage, and the final output is correct.
    """
    nemo_path = (
        REPO_ROOT
        / "compiler"
        / "crates"
        / "nemoir-dsl-fe"
        / "tests"
        / "fixtures"
        / "exec_shell.nemo"
    )
    pkg = _compile_and_import(nemo_path, "exec_shell", tmp_path)

    from nemoir_runtime.official_tools import run_shell  # noqa: PLC0415

    tools = ToolRegistry([run_shell])

    calls: list[str] = []

    class FakeModel:
        async def complete(self, _request: Any) -> Any:
            calls.append("model")
            return ModelResponse(content='{"summary": "processed"}')

    agent = pkg.Agent(model=FakeModel(), tools=tools)

    result = asyncio.run(agent.run(pkg.AgentInput()))
    assert result.output.summary == "processed"
    assert calls == ["model"]
