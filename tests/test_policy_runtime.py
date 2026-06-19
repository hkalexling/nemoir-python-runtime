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
