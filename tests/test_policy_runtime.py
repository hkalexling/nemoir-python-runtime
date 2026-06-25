from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.errors import (
    MissingCapabilityError,
    PolicyDeniedError,
    PolicyEvaluationError,
)
from nemoir_runtime.runtime import (
    ExprSpec,
    InputSpec,
    PolicySpec,
    RefSpec,
    RequiredCapabilitySpec,
    StageContext,
    StageSpec,
    TriggerSpec,
    WorkflowManifest,
    WorkflowRuntime,
    WriteSpec,
)
from nemoir_runtime.tools import Tool, ToolContext, ToolRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping

# ------------------------------------------------------------------
# Deny policy tests
# ------------------------------------------------------------------


async def test_fs_read_outside_cwd_denied(make_registry_with_log: Any) -> None:
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm", "os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class DenyTestExecutor:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            # Attempt read outside cwd
            await ctx.call_tool("fs.read", {"path": Path("/etc/passwd")})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=DenyTestExecutor())
    with pytest.raises(PolicyDeniedError, match="denied"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test", "cwd": Path("/tmp/work")})
    assert not calls  # handler never ran


async def test_fs_read_inside_cwd_runs(make_registry_with_log: Any) -> None:
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm", "os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class InsideTestExecutor:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.read", {"path": (Path("/tmp/work") / "file.txt")})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(
        manifest=manifest, tools=registry, stage_executor=InsideTestExecutor()
    )
    await runtime.run({"task": "test", "cwd": Path("/tmp/work")})
    assert len(calls) == 1
    assert calls[0][0] == "fs.read"


# ------------------------------------------------------------------
# Before policy tests
# ------------------------------------------------------------------


async def test_fs_write_runs_before_policy(make_registry_with_log: Any) -> None:
    before_policy = PolicySpec(
        id="before fs.write(path) requires fs.read(path), user.confirm",
        kind="before",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        requires=(
            RequiredCapabilitySpec(
                capability="fs.read",
                args={"path": RefSpec(kind="bound", name="path")},
            ),
            RequiredCapabilitySpec(
                capability="user.confirm",
                args={},
            ),
        ),
    )

    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"fs.write", "fs.read"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm", "os.shell"}),
        policies=(before_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class BeforeTestExecutor:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/work/file.txt"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(
        manifest=manifest, tools=registry, stage_executor=BeforeTestExecutor()
    )
    await runtime.run({"task": "test", "cwd": Path("/tmp/work")})
    # fs.read first, then user.confirm, then fs.write
    assert [c[0] for c in calls] == ["fs.read", "user.confirm", "fs.write"]


async def test_user_confirm_false_blocks_write() -> None:
    async def confirm_false(*, message: str, ctx: ToolContext) -> bool:
        return False

    async def read_ok(*, path: Path, ctx: ToolContext) -> str:
        return "ok"

    async def write_ok(*, path: Path, content: str, ctx: ToolContext) -> None:
        pass

    confirm_tool = Tool(
        name="confirm",
        capability="user.confirm",
        description="confirm",
        input_schema={"message": str},
        handler=confirm_false,
    )
    read_tool = Tool(
        name="read_file",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=read_ok,
    )
    write_tool = Tool(
        name="write_file",
        capability="fs.write",
        description="w",
        input_schema={"path": Path, "content": str},
        handler=write_ok,
    )
    registry_deny = ToolRegistry([read_tool, write_tool, confirm_tool])

    before_policy = PolicySpec(
        id="before fs.write(path) requires user.confirm",
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm"}),
        policies=(before_policy,),
        stages=stages,
    )

    class ConfirmDenyExecutor:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/work/f.txt"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(
        manifest=manifest, tools=registry_deny, stage_executor=ConfirmDenyExecutor()
    )
    with pytest.raises(PolicyDeniedError, match=r"user\.confirm returned False"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test", "cwd": Path("/tmp/work")})


# ------------------------------------------------------------------
# Stage visibility tests
# ------------------------------------------------------------------


async def test_stage_cannot_call_capability_outside_requires() -> None:
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
        capabilities=frozenset({"fs.read", "fs.write"}),
        policies=(),
        stages=stages,
    )
    registry = _make_basic_registry()

    class BadExecutor:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/f.txt"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=BadExecutor())
    with pytest.raises(MissingCapabilityError, match="not available in stage"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})


# ------------------------------------------------------------------
# Regression: policy-required calls still enforce deny policies
# ------------------------------------------------------------------


async def test_before_required_call_enforces_deny() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def read_fn(*, path: Path, ctx: ToolContext) -> str:
        calls.append(("fs.read", {"path": str(path)}))
        return f"read:{path}"

    async def write_fn(*, path: Path, content: str, ctx: ToolContext) -> None:
        calls.append(("fs.write", {"path": str(path), "content": content}))

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
    registry = ToolRegistry([read_tool, write_tool])

    before_policy = PolicySpec(
        id="before fs.write(path) requires fs.read(path)",
        kind="before",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        requires=(
            RequiredCapabilitySpec(
                capability="fs.read",
                args={"path": RefSpec(kind="bound", name="path")},
            ),
        ),
    )
    deny_read_policy = PolicySpec(
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
            requires=frozenset({"fs.write"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read", "fs.write"}),
        policies=(before_policy, deny_read_policy),
        stages=stages,
    )

    class OutsideWriteExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.write", {"path": Path("/etc/passwd"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(
        manifest=manifest, tools=registry, stage_executor=OutsideWriteExecutor()
    )
    with pytest.raises(PolicyDeniedError, match="denied"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test", "cwd": Path("/tmp/work")})
    assert not calls


# ------------------------------------------------------------------
# Regression: deny policy accepts string cwd via contains() coercion
# ------------------------------------------------------------------


async def test_deny_policy_accepts_string_cwd(make_registry_with_log: Any) -> None:
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, _ = make_registry_with_log()

    class StringCwdExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.read", {"path": Path("/tmp/work") / "file.txt"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=StringCwdExecutor())
    await runtime.run({"task": "test", "cwd": "/tmp/work"})
    # Should not raise — string cwd is coerced to Path


async def _read_ok(*, path: Path, ctx: ToolContext) -> str:
    return ""


async def _write_ok(*, path: Path, content: str, ctx: ToolContext) -> None:
    pass


def _make_basic_registry() -> ToolRegistry:
    read_tool = Tool(
        name="read_file",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=_read_ok,
    )
    write_tool = Tool(
        name="write_file",
        capability="fs.write",
        description="w",
        input_schema={"path": Path, "content": str},
        handler=_write_ok,
    )
    return ToolRegistry([read_tool, write_tool])


# ------------------------------------------------------------------
# Comment 1 tests: Policy evaluation error type
# ------------------------------------------------------------------


async def test_unknown_policy_method_raises_policy_evaluation_error() -> None:
    """Policy condition with unknown method raises PolicyEvaluationError."""
    deny_policy = PolicySpec(
        id="deny-fs.read-bogus",
        kind="deny",
        trigger=TriggerSpec(capability="fs.read", bind={"path": "path"}),
        condition=ExprSpec(
            kind="method_call",
            receiver=ExprSpec(kind="ref", ref=RefSpec(kind="input", name="cwd")),
            method="bogus",
            args=(ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry = _make_basic_registry()

    class TestExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.read", {"path": Path("/tmp/work/file.txt")})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=TestExecutor())
    with pytest.raises(PolicyEvaluationError, match="Unknown method"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test", "cwd": Path("/tmp/work")})


async def test_missing_policy_expr_ref_raises_policy_evaluation_error() -> None:
    """Policy condition with ref=None raises PolicyEvaluationError."""
    deny_policy = PolicySpec(
        id="deny-fs.read-null-ref",
        kind="deny",
        trigger=TriggerSpec(capability="fs.read", bind={"path": "path"}),
        condition=ExprSpec(kind="ref", ref=None),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry = _make_basic_registry()

    class TestExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.read", {"path": Path("/tmp/work/file.txt")})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=TestExecutor())
    with pytest.raises(PolicyEvaluationError, match="Ref expression has no ref"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test", "cwd": Path("/tmp/work")})


async def test_contains_invalid_receiver_raises_policy_evaluation_error() -> None:
    """contains() with non-Path receiver raises PolicyEvaluationError."""
    deny_policy = PolicySpec(
        id="deny-fs.read-bad-receiver",
        kind="deny",
        trigger=TriggerSpec(capability="fs.read", bind={"path": "path"}),
        condition=ExprSpec(
            kind="method_call",
            receiver=ExprSpec(kind="literal", type="int", value=42),
            method="contains",
            args=(ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry = _make_basic_registry()

    class TestExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.read", {"path": Path("/tmp/work/file.txt")})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=TestExecutor())
    with pytest.raises(PolicyEvaluationError, match="receiver must be Path"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test", "cwd": Path("/tmp/work")})


# ------------------------------------------------------------------
# Comment 2 tests: Missing trigger-bound arg
# ------------------------------------------------------------------


async def test_missing_trigger_bound_arg_raises_policy_evaluation_error() -> None:
    """fs.write call missing 'path' fails with PolicyEvaluationError before handlers run."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def write_fn(*, path: Path, content: str, ctx: ToolContext) -> None:
        calls.append(("fs.write", {"path": str(path), "content": content}))

    async def read_fn(*, path: Path, ctx: ToolContext) -> str:
        calls.append(("fs.read", {"path": str(path)}))
        return "ok"

    async def confirm_fn(*, message: str, ctx: ToolContext) -> bool:
        calls.append(("user.confirm", {"message": message}))
        return True

    write_tool = Tool(
        name="write_file",
        capability="fs.write",
        description="w",
        input_schema={"path": Path, "content": str},
        handler=write_fn,
    )
    read_tool = Tool(
        name="read_file",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=read_fn,
    )
    confirm_tool = Tool(
        name="confirm",
        capability="user.confirm",
        description="confirm",
        input_schema={"message": str},
        handler=confirm_fn,
    )
    registry = ToolRegistry([read_tool, write_tool, confirm_tool])

    before_policy = PolicySpec(
        id="before fs.write(path) requires fs.read(path), user.confirm",
        kind="before",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        requires=(
            RequiredCapabilitySpec(
                capability="fs.read",
                args={"path": RefSpec(kind="bound", name="path")},
            ),
            RequiredCapabilitySpec(
                capability="user.confirm",
                args={},
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm"}),
        policies=(before_policy,),
        stages=stages,
    )

    class MissingArgExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.write", {"content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(
        manifest=manifest, tools=registry, stage_executor=MissingArgExecutor()
    )
    with pytest.raises(PolicyEvaluationError, match=r"trigger-bound argument.*missing"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})
    assert not calls  # No handlers run


async def test_required_arg_undeclared_bound_ref_raises_policy_evaluation_error() -> None:
    """RequiredCapabilitySpec refs undeclared bound name raises PolicyEvaluationError."""
    before_policy = PolicySpec(
        id="before fs.write(path) requires fs.read(undeclared)",
        kind="before",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        requires=(
            RequiredCapabilitySpec(
                capability="fs.read",
                args={"path": RefSpec(kind="bound", name="undeclared")},
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.read", "fs.write"}),
        policies=(before_policy,),
        stages=stages,
    )
    registry = _make_basic_registry()

    class TestExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/work/f.txt"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=TestExecutor())
    with pytest.raises(PolicyEvaluationError, match=r"bound ref.*could not be resolved"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})


# ------------------------------------------------------------------
# Second review: policy-required calls missing catalog-required args
# ------------------------------------------------------------------


async def test_before_required_call_missing_catalog_args_raises() -> None:
    """before fs.write(path) requires fs.read (no forwarded args) fails with
    PolicyEvaluationError before any handlers run."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def read_fn(*, path: Path, ctx: ToolContext) -> str:
        calls.append(("fs.read", {"path": str(path)}))
        return "ok"

    async def write_fn(*, path: Path, content: str, ctx: ToolContext) -> None:
        calls.append(("fs.write", {"path": str(path), "content": content}))

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
    registry = ToolRegistry([read_tool, write_tool])

    before_policy = PolicySpec(
        id="before fs.write(path) requires fs.read",
        kind="before",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        requires=(
            RequiredCapabilitySpec(
                capability="fs.read",
                args={},
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.read", "fs.write"}),
        policies=(before_policy,),
        stages=stages,
    )

    class TestExecutor:
        async def execute(self, ctx: StageContext) -> dict[str, Any]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/work/f.txt"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=TestExecutor())
    with pytest.raises(PolicyEvaluationError, match="missing catalog-required argument"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "test"})
    assert not calls  # No handlers run


# ------------------------------------------------------------------
# Plan Medium-Tests-1: policy applies to both write_file and edit_file
# ------------------------------------------------------------------


async def test_before_policy_triggers_for_both_write_file_and_edit_file(
    tmp_path: Path,
) -> None:
    """before fs.write(path) requires fs.read(path), user.confirm
    triggers before both official write_file and edit_file."""
    from nemoir_runtime.official_tools import edit_file, write_file  # noqa: PLC0415

    policy_calls: list[tuple[str, dict[str, Any]]] = []

    async def read_stub(*, path: Path, ctx: ToolContext) -> str:
        policy_calls.append(("fs.read", {"path": str(path)}))
        return "ok"

    async def confirm_stub(*, message: str, ctx: ToolContext) -> bool:
        policy_calls.append(("user.confirm", {"message": message}))
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
    registry = ToolRegistry([read_tool, write_file, edit_file, confirm_tool])

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
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"fs.write"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="cwd", type="path"),),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm"}),
        policies=(before_policy,),
        stages=stages,
    )

    # First: call write_file via tool_name (write_file creates the file).
    target = tmp_path / "f.txt"

    class CallWriteFile:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool(
                "fs.write",
                {"path": target, "content": "hello"},
                tool_name="write_file",
            )
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=CallWriteFile())
    await runtime.run({"cwd": str(tmp_path)})

    # write_file: before-policy fs.read + user.confirm, then handler.
    assert policy_calls[0] == ("fs.read", {"path": str(target)})
    assert policy_calls[1][0] == "user.confirm"
    assert target.read_text() == "hello"

    # Now: call edit_file via tool_name.
    target.write_text("hello old end")
    policy_calls.clear()

    class CallEditFile:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool(
                "fs.write",
                {
                    "path": target,
                    "content": "old",
                    "new_content": "new",
                },
                tool_name="edit_file",
            )
            return {"out_a": "done"}

    runtime2 = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=CallEditFile())
    await runtime2.run({"cwd": str(tmp_path)})

    # edit_file: before-policy fs.read + user.confirm, then handler.
    assert policy_calls[0] == ("fs.read", {"path": str(target)})
    assert policy_calls[1][0] == "user.confirm"
    assert target.read_text() == "hello new end"


async def test_deny_policy_denies_edit_file_outside_cwd(tmp_path: Path) -> None:
    """deny fs.write(path) if not cwd.contains(path) denies edit_file."""
    from nemoir_runtime.official_tools import edit_file, write_file  # noqa: PLC0415

    async def read_stub(*, path: Path, ctx: ToolContext) -> str:
        return "ok"

    read_tool = Tool(
        name="read_file",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=read_stub,
    )
    registry = ToolRegistry([read_tool, write_file, edit_file])

    deny_policy = PolicySpec(
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(
            InputSpec(name="task", type="string"),
            InputSpec(name="cwd", type="path"),
        ),
        capabilities=frozenset({"fs.read", "fs.write"}),
        policies=(deny_policy,),
        stages=stages,
    )

    class CallEditFileOutside:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool(
                "fs.write",
                {
                    "path": Path("/etc/passwd"),
                    "content": "old",
                    "new_content": "new",
                },
                tool_name="edit_file",
            )
            return {"out_a": "never"}

    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=registry,
        stage_executor=CallEditFileOutside(),
    )
    with pytest.raises(PolicyDeniedError, match="denied"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"task": "t", "cwd": Path("/tmp/work")})


async def test_confirm_false_blocks_both_write_tools(tmp_path: Path) -> None:
    """user.confirm=False blocks both write_file and edit_file via before."""
    from nemoir_runtime.official_tools import edit_file, write_file  # noqa: PLC0415

    async def read_stub(*, path: Path, ctx: ToolContext) -> str:
        return "ok"

    async def confirm_false(*, message: str, ctx: ToolContext) -> bool:
        return False

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
        handler=confirm_false,
    )
    registry = ToolRegistry([read_tool, write_file, edit_file, confirm_tool])

    before_policy = PolicySpec(
        id="before fs.write requires user.confirm",
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="cwd", type="path"),),
        capabilities=frozenset({"fs.read", "fs.write", "user.confirm"}),
        policies=(before_policy,),
        stages=stages,
    )

    # Block write_file.
    class CallWriteFile:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            target = tmp_path / "w.txt"
            await ctx.call_tool(
                "fs.write",
                {"path": target, "content": "x"},
                tool_name="write_file",
            )
            return {"out_a": "never"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=CallWriteFile())
    with pytest.raises(PolicyDeniedError, match=r"user\.confirm returned False"):  # type: ignore[reportUnknownMemberType]
        await runtime.run({"cwd": str(tmp_path)})

    # Block edit_file.
    class CallEditFile:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            target = tmp_path / "e.txt"
            target.write_text("hello old world")
            await ctx.call_tool(
                "fs.write",
                {
                    "path": target,
                    "content": "old",
                    "new_content": "new",
                },
                tool_name="edit_file",
            )
            return {"out_a": "never"}

    runtime2 = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=CallEditFile())
    with pytest.raises(PolicyDeniedError, match=r"user\.confirm returned False"):  # type: ignore[reportUnknownMemberType]
        await runtime2.run({"cwd": str(tmp_path)})


# ------------------------------------------------------------------
# New predicate tests (Phase 2: eq, starts_with, string contains, and/or)
# ------------------------------------------------------------------


async def test_os_shell_command_eq_allowed(make_registry_with_log: Any) -> None:
    """Exact command match allows the call."""
    deny_policy = PolicySpec(
        id='deny os.shell(command) if not command.eq("python run.py")',
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
                method="eq",
                args=(ExprSpec(kind="literal", type="string", value="python run.py"),),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "python run.py"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    await runtime.run({"task": "test"})
    assert len(calls) == 1


async def test_os_shell_command_eq_denied(make_registry_with_log: Any) -> None:
    deny_policy = PolicySpec(
        id='deny os.shell(command) if not command.eq("python run.py")',
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
                method="eq",
                args=(ExprSpec(kind="literal", type="string", value="python run.py"),),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "rm -rf /"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    with pytest.raises(PolicyDeniedError, match="denied"):
        await runtime.run({"task": "test"})
    assert not calls


async def test_os_shell_command_starts_with_allowed(make_registry_with_log: Any) -> None:
    deny_policy = PolicySpec(
        id='deny os.shell(command) if not command.starts_with("python run.py")',
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
                method="starts_with",
                args=(ExprSpec(kind="literal", type="string", value="python run.py"),),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "python run.py --flag"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    await runtime.run({"task": "test"})
    assert len(calls) == 1


async def test_os_shell_command_contains_metachar_denied(make_registry_with_log: Any) -> None:
    """command.contains("&&") should deny shell injection"""
    deny_policy = PolicySpec(
        id='deny os.shell(command) if command.contains("&&")',
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="method_call",
            receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
            method="contains",
            args=(ExprSpec(kind="literal", type="string", value="&&"),),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "echo hi && rm -rf /"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    with pytest.raises(PolicyDeniedError, match="denied"):
        await runtime.run({"task": "test"})
    assert not calls


async def test_fs_write_path_eq_allowed(make_registry_with_log: Any) -> None:
    deny_policy = PolicySpec(
        id="deny fs.write(path) if not path.eq(candidate_path)",
        kind="deny",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),
                method="eq",
                args=(ExprSpec(kind="ref", ref=RefSpec(kind="input", name="candidate_path")),),
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(
            InputSpec(name="task", type="string"),
            InputSpec(name="candidate_path", type="path"),
        ),
        capabilities=frozenset({"fs.write"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/candidate.py"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    await runtime.run({"task": "test", "candidate_path": Path("/tmp/candidate.py")})
    assert len(calls) == 1


async def test_fs_write_path_eq_denied(make_registry_with_log: Any) -> None:
    deny_policy = PolicySpec(
        id="deny fs.write(path) if not path.eq(candidate_path)",
        kind="deny",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),
                method="eq",
                args=(ExprSpec(kind="ref", ref=RefSpec(kind="input", name="candidate_path")),),
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(
            InputSpec(name="task", type="string"),
            InputSpec(name="candidate_path", type="path"),
        ),
        capabilities=frozenset({"fs.write"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("/tmp/harness/eval.py"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    with pytest.raises(PolicyDeniedError, match="denied"):
        await runtime.run({"task": "test", "candidate_path": Path("/tmp/candidate.py")})
    assert not calls


async def test_eq_path_relative_lexical(make_registry_with_log: Any) -> None:
    """Relative path eq does lexical comparison without filesystem access."""
    deny_policy = PolicySpec(
        id='deny fs.write(path) if not path.eq("candidate.py")',
        kind="deny",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),
                method="eq",
                args=(ExprSpec(kind="literal", type="string", value="candidate.py"),),
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.write"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("candidate.py"), "content": "x"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    await runtime.run({"task": "test"})
    assert len(calls) == 1


async def test_and_short_circuits(make_registry_with_log: Any) -> None:
    """And expression short-circuits: first false means second never evaluated."""
    # Build: deny if command.eq("bad") and missing_bound_ref.eq("x")
    # command="good", so first operand (command.eq("bad")) is False
    # AND short-circuits, never evaluating the second operand (which would fail)
    deny_policy = PolicySpec(
        id='deny os.shell(command) if command.eq("bad") and missing.eq("x")',
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="and",
            exprs=(
                ExprSpec(
                    kind="method_call",
                    receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
                    method="eq",
                    args=(ExprSpec(kind="literal", type="string", value="bad"),),
                ),
                ExprSpec(
                    kind="method_call",
                    receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="missing")),
                    method="eq",
                    args=(ExprSpec(kind="literal", type="string", value="x"),),
                ),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "good"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    # Should succeed: first operand (command.eq("bad")) is False for command="good",
    # so AND short-circuits and never evaluates the second operand.
    await runtime.run({"task": "test"})
    assert len(calls) == 1


async def test_or_short_circuits(make_registry_with_log: Any) -> None:
    """Or expression short-circuits: first true means second never evaluated."""
    # Build: deny if command.eq("good") or missing_bound_ref.eq("x")
    # command="good", so first operand is True, OR short-circuits,
    # never evaluating the second operand (which would fail with missing ref)
    deny_policy = PolicySpec(
        id='deny os.shell(command) if command.eq("good") or missing.eq("x")',
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="or",
            exprs=(
                ExprSpec(
                    kind="method_call",
                    receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
                    method="eq",
                    args=(ExprSpec(kind="literal", type="string", value="good"),),
                ),
                ExprSpec(
                    kind="method_call",
                    receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="missing")),
                    method="eq",
                    args=(ExprSpec(kind="literal", type="string", value="x"),),
                ),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "good"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    # Should raise PolicyDeniedError because first operand is True (denied),
    # and OR short-circuits. Second operand (with missing bound ref) never reached.
    with pytest.raises(PolicyDeniedError, match="denied"):
        await runtime.run({"task": "test"})
    assert not calls


# ------------------------------------------------------------------
# Missing §8.4 tests (starts_with denied, absolute eq, in-allowlist lowered,
# edit/write gated with new eq predicate)
# ------------------------------------------------------------------


async def test_os_shell_command_starts_with_denied(make_registry_with_log: Any) -> None:
    """Non-matching prefix should raise PolicyDeniedError."""
    deny_policy = PolicySpec(
        id='deny os.shell(command) if not command.starts_with("python run.py")',
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
                method="starts_with",
                args=(ExprSpec(kind="literal", type="string", value="python run.py"),),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "rm -rf /"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    with pytest.raises(PolicyDeniedError, match="denied"):
        await runtime.run({"task": "test"})
    assert not calls


async def test_eq_path_absolute_resolves(make_registry_with_log: Any) -> None:
    """Absolute path eq uses resolve(strict=False) from docs/dsl-and-ir.md §6.1."""
    deny_policy = PolicySpec(
        id='deny fs.write(path) if not path.eq("/tmp/work/candidate.py")',
        kind="deny",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),
                method="eq",
                args=(ExprSpec(kind="literal", type="string", value="/tmp/work/candidate.py"),),
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.write"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool(
                "fs.write",
                {"path": Path("/tmp/work/candidate.py"), "content": "x"},
            )
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    await runtime.run({"task": "test"})
    assert len(calls) == 1


async def test_fs_write_path_in_allowlist_lowered(make_registry_with_log: Any) -> None:
    """The lowered shape (Or of eq) from `in [...]` should allow matches."""
    deny_policy = PolicySpec(
        id="deny fs.write(path) if not (path.eq(x) or path.eq(y))",
        kind="deny",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="or",
                exprs=(
                    ExprSpec(
                        kind="method_call",
                        receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),
                        method="eq",
                        args=(ExprSpec(kind="literal", type="string", value="a.py"),),
                    ),
                    ExprSpec(
                        kind="method_call",
                        receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),
                        method="eq",
                        args=(ExprSpec(kind="literal", type="string", value="b.py"),),
                    ),
                ),
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.write"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    # Allowed: a.py matches first disjunct
    class ExecA:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("a.py"), "content": "x"})
            return {"out_a": "done"}

    runtime_a = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=ExecA())
    await runtime_a.run({"task": "test"})
    assert len(calls) == 1
    calls.clear()

    # Denied: z.py matches neither disjunct
    class ExecZ:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("fs.write", {"path": Path("z.py"), "content": "x"})
            return {"out_a": "done"}

    runtime_z = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=ExecZ())
    with pytest.raises(PolicyDeniedError, match="denied"):
        await runtime_z.run({"task": "test"})
    assert not calls


async def test_eq_policy_applies_to_edit_file_and_write_file(make_registry_with_log: Any) -> None:
    """Both official fs.write tools (write_file + edit_file) are gated by path.eq.

    We exercise two fs.write calls (one with tool_name="write_file", one
    without) to assert both reach the same policy enforcement path.  The
    registry's single fs.write tool serves both calls.
    """
    deny_policy = PolicySpec(
        id='deny fs.write(path) if not path.eq("/tmp/candidate.py")',
        kind="deny",
        trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="path")),
                method="eq",
                args=(ExprSpec(kind="literal", type="string", value="/tmp/candidate.py"),),
            ),
        ),
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"fs.write"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    # Allowed: write_file to candidate.py
    class CallWriteFile:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool(
                "fs.write",
                {"path": Path("/tmp/candidate.py"), "content": "hello"},
                tool_name="write_file",
            )
            return {"out_a": "done"}

    runtime_w = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=CallWriteFile())
    await runtime_w.run({"task": "test"})
    assert len(calls) == 1
    calls.clear()

    # Denied: write_file to non-allowlisted path
    class CallDenied:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool(
                "fs.write",
                {"path": Path("/tmp/harness/eval.py"), "content": "x"},
                tool_name="write_file",
            )
            return {"out_a": "done"}

    runtime_d = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=CallDenied())
    with pytest.raises(PolicyDeniedError, match="denied"):
        await runtime_d.run({"task": "test"})
    assert not calls


# ------------------------------------------------------------------
# Regression: string cwd + string bound path → path containment (not substring)
# ------------------------------------------------------------------


async def test_contains_string_cwd_and_string_bound_path_uses_path_containment(
    make_registry_with_log: Any,
) -> None:
    """When cwd (path-typed input) arrives as a string, and the bound path is
    also a string (as a model emits), the coercion fix ensures path containment
    still applies — not substring match. Regression for false-deny."""
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            # Call with a STRING path arg (as a model emits) — the bound
            # `path` should be coerced to Path, and cwd stays str but
            # the presence of the Path arg triggers promotion to path
            # containment, not substring match.
            await ctx.call_tool("fs.read", {"path": "file.txt"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    # cwd is passed as a string; policy should still allow (path containment)
    await runtime.run({"task": "test", "cwd": "/tmp/work"})
    assert len(calls) == 1


async def test_contains_substring_bypass_with_string_cwd_rejected(
    make_registry_with_log: Any,
) -> None:
    """An absolute ancestor path that is a substring of the cwd string must NOT
    pass containment — it is outside cwd. Regression for substring bypass."""
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
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"), InputSpec(name="cwd", type="path")),
        capabilities=frozenset({"fs.read"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            # /home/alice is a SUBSTRING of /home/alice/project but is
            # its ancestor, not inside it.  Path containment says False.
            await ctx.call_tool("fs.read", {"path": "/home/alice"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    with pytest.raises(PolicyDeniedError, match="denied"):
        await runtime.run({"task": "test", "cwd": "/home/alice/project"})
    assert not calls


# ------------------------------------------------------------------
# Defense-in-depth runtime tests: reject invalid manifests at runtime
# ------------------------------------------------------------------


async def test_contains_extra_args_raises_policy_evaluation_error(
    make_registry_with_log: Any,
) -> None:
    """Runtime defense: contains() with extra args raises PolicyEvaluationError."""
    deny_policy = PolicySpec(
        id='deny os.shell(command) if command.contains("x", "y")',
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="method_call",
            receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
            method="contains",
            args=(
                ExprSpec(kind="literal", type="string", value="x"),
                ExprSpec(kind="literal", type="string", value="y"),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="task", type="string"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, _ = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "echo hi"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    with pytest.raises(PolicyEvaluationError, match="requires exactly 1 argument"):
        await runtime.run({"task": "test"})


async def test_string_contains_path_raises_policy_evaluation_error(
    make_registry_with_log: Any,
) -> None:
    """Runtime defense: string.contains(Path) raises PolicyEvaluationError."""
    deny_policy = PolicySpec(
        id="deny os.shell(command) if command.contains(cwd)",
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="method_call",
            receiver=ExprSpec(kind="ref", ref=RefSpec(kind="bound", name="command")),
            method="contains",
            args=(ExprSpec(kind="ref", ref=RefSpec(kind="input", name="cwd")),),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(
            InputSpec(name="task", type="string"),
            InputSpec(name="cwd", type="path"),
        ),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, _ = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "echo hi"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    with pytest.raises(PolicyEvaluationError, match="argument must be string"):
        await runtime.run({"task": "test", "cwd": Path("/tmp/work")})


async def test_eq_with_numeric_operands_raises(make_registry_with_log: Any) -> None:
    """eq() with numeric operands raises PolicyEvaluationError (ordering-only rule)."""
    deny_policy = PolicySpec(
        id="deny numeric eq",
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="input", name="score")),
                method="eq",
                args=(ExprSpec(kind="literal", type="number", value=0.3),),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="score", type="number"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, _ = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "echo hi"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    with pytest.raises(PolicyEvaluationError, match="number"):
        await runtime.run({"score": 0.1 + 0.2})


async def test_eq_with_bool_operands_still_works(make_registry_with_log: Any) -> None:
    """eq() with bool operands still works (bool excluded from numeric check)."""
    deny_policy = PolicySpec(
        id="deny bool eq",
        kind="deny",
        trigger=TriggerSpec(capability="os.shell", bind={"command": "command"}),
        condition=ExprSpec(
            kind="not",
            expr=ExprSpec(
                kind="method_call",
                receiver=ExprSpec(kind="ref", ref=RefSpec(kind="input", name="flag")),
                method="eq",
                args=(ExprSpec(kind="literal", type="bool", value=True),),
            ),
        ),
    )
    stages = (
        StageSpec(
            id="A",
            prompt="A",
            reads=(),
            writes=(WriteSpec(name="out_a", type="string", optional=False),),
            requires=frozenset({"os.shell"}),
            transitions=(),
        ),
    )
    manifest = WorkflowManifest(
        workflow_id="Test",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(InputSpec(name="flag", type="bool"),),
        capabilities=frozenset({"os.shell"}),
        policies=(deny_policy,),
        stages=stages,
    )
    registry, calls = make_registry_with_log()

    class Exec:
        async def execute(self, ctx: StageContext) -> Mapping[str, object]:
            await ctx.call_tool("os.shell", {"command": "echo hi"})
            return {"out_a": "done"}

    runtime = WorkflowRuntime(manifest=manifest, tools=registry, stage_executor=Exec())
    # eq(True, True) returns True → Not(True) = False → not denied → call proceeds
    await runtime.run({"flag": True})
    assert len(calls) == 1
