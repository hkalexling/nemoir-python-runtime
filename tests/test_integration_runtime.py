from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.errors import PolicyDeniedError
from nemoir_runtime.runtime import (
    ExprSpec,
    GuardSpec,
    InputSpec,
    PolicySpec,
    ReadSpec,
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


def _make_coding_agent_policies() -> tuple[PolicySpec, ...]:
    return (
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
        PolicySpec(
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
        ),
        PolicySpec(
            id="deny fs.write(path) if not cwd.contains(path)",
            kind="deny",
            trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
            condition=ExprSpec(
                kind="not",
                expr=ExprSpec(
                    kind="method_call",
                    receiver=ExprSpec(kind="ref", ref=RefSpec(kind="input", name="cwd")),
                    method="contains",
                    args=(ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),),
                ),
            ),
        ),
    )


# ------------------------------------------------------------------
# Compact integration: Triage -> Plan -?-> Fin  (conditional optional-skip)
# ------------------------------------------------------------------

_STAGES = (
    StageSpec(
        id="Triage",
        prompt="Triage prompt",
        reads=(),
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        requires=frozenset({"fs.read"}),
        transitions=(
            TransitionSpec(
                to="Plan",
                priority=0,
                reason="fallthrough",
                guard=GuardSpec(kind="always"),
            ),
        ),
    ),
    StageSpec(
        id="Plan",
        prompt="Plan prompt",
        reads=(
            ReadSpec(
                ref=RefSpec(kind="node_output", node="Triage", field="summary"),
                optional=False,
            ),
        ),
        writes=(
            WriteSpec(name="plan", type="string", optional=False),
            WriteSpec(name="verification_plan", type="string", optional=True),
        ),
        requires=frozenset({"fs.read", "fs.write"}),
        transitions=(
            TransitionSpec(
                to="Verify",
                priority=0,
                reason="next_stage_required_input_available",
                guard=GuardSpec(
                    kind="has_value",
                    ref=RefSpec(kind="node_output", node="Plan", field="verification_plan"),
                ),
            ),
            TransitionSpec(
                to="Fin",
                priority=1,
                reason="skip_next_stage_required_input_missing",
                guard=GuardSpec(
                    kind="missing",
                    ref=RefSpec(kind="node_output", node="Plan", field="verification_plan"),
                ),
            ),
        ),
    ),
    StageSpec(
        id="Verify",
        prompt="Verify prompt",
        reads=(
            ReadSpec(
                ref=RefSpec(kind="node_output", node="Plan", field="verification_plan"),
                optional=False,
            ),
        ),
        writes=(WriteSpec(name="ok", type="bool", optional=False),),
        requires=frozenset({"fs.read", "os.shell"}),
        transitions=(
            TransitionSpec(
                to="Fin",
                priority=0,
                reason="fallthrough",
                guard=GuardSpec(kind="always"),
            ),
        ),
    ),
    StageSpec(
        id="Fin",
        prompt="Fin prompt",
        reads=(
            ReadSpec(
                ref=RefSpec(kind="node_output", node="Plan", field="plan"),
                optional=False,
            ),
        ),
        writes=(WriteSpec(name="summary", type="string", optional=False),),
        requires=frozenset(),
        transitions=(),
    ),
)


async def test_integration_workflow_runs_with_policies(make_registry_with_log: Any) -> None:
    manifest = WorkflowManifest(
        workflow_id="CodingAgentCompact",
        entry_stage_id="Triage",
        exit_stage_ids=frozenset({"Fin"}),
        inputs=(
            InputSpec(name="task", type="string"),
            InputSpec(name="cwd", type="path"),
        ),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm", "os.shell"}),
        policies=_make_coding_agent_policies(),
        stages=_STAGES,
    )
    registry, calls = make_registry_with_log()

    cwd = Path("/tmp/integration_test")

    class IntegrationExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            if ctx.stage.id == "Triage":
                await ctx.call_tool("fs.read", {"path": cwd / "README.md"})
                return {"summary": "Analyzed task"}
            if ctx.stage.id == "Plan":
                await ctx.call_tool("fs.write", {"path": cwd / "output.txt", "content": "changes"})
                return {
                    "plan": "Apply changes to output.txt",
                    "verification_plan": "Run tests",
                }
            if ctx.stage.id == "Verify":
                await ctx.call_tool("os.shell", {"command": "pytest"})
                return {"ok": True}
            if ctx.stage.id == "Fin":
                return {"summary": "Done: Applied changes to output.txt"}
            msg = f"Unknown stage {ctx.stage.id}"
            raise RuntimeError(msg)

    runtime = WorkflowRuntime(
        manifest=manifest, tools=registry, stage_executor=IntegrationExecutor()
    )
    result = await runtime.run(
        {"task": "Fix the bug", "cwd": cwd},
        options=RunOptions(max_steps=10),
    )

    assert result.output["summary"] == "Done: Applied changes to output.txt"
    assert result.state.steps == 4

    # Triage: fs.read only (no before policy on fs.read)
    # Plan: fs.write triggers fs.read + user.confirm + fs.write
    assert calls[0][0] == "fs.read"  # Triage read
    assert calls[1][0] == "fs.read"  # before: fs.read (from write policy)
    assert calls[2][0] == "user.confirm"  # before: user.confirm
    assert calls[3][0] == "fs.write"  # Plan write


async def test_integration_deny_write_outside_cwd(make_registry_with_log: Any) -> None:
    manifest = WorkflowManifest(
        workflow_id="CodingAgentCompact",
        entry_stage_id="Triage",
        exit_stage_ids=frozenset({"Fin"}),
        inputs=(
            InputSpec(name="task", type="string"),
            InputSpec(name="cwd", type="path"),
        ),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm", "os.shell"}),
        policies=_make_coding_agent_policies(),
        stages=_STAGES,
    )
    registry, calls = make_registry_with_log()

    class WriteOutsideCwdExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            if ctx.stage.id == "Triage":
                return {"summary": "Analyzed"}
            if ctx.stage.id == "Plan":
                # Attempt write outside cwd — deny policy should stop before handler
                await ctx.call_tool("fs.write", {"path": Path("/etc/passwd"), "content": "x"})
                return {"plan": "x"}
            if ctx.stage.id == "Fin":
                return {"summary": "done"}
            msg = f"Unknown stage {ctx.stage.id}"
            raise RuntimeError(msg)

    runtime = WorkflowRuntime(
        manifest=manifest, tools=registry, stage_executor=WriteOutsideCwdExecutor()
    )
    with pytest.raises(PolicyDeniedError, match="denied"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test", "cwd": Path("/tmp/work")})
    # The before policy's fs.read should not have run either
    assert len(calls) == 0
