from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.capabilities import CapabilityParamType
from nemoir_runtime.errors import (
    MissingCapabilityError,
    ToolInvocationError,
    ToolValidationError,
)
from nemoir_runtime.tools import (
    CATALOG_TYPE_MAP,
    Tool,
    ToolContext,
    ToolRegistry,
    tool,
)


def test_tool_decorator_captures_metadata() -> None:
    @tool(capability="fs.read", description="Read a file under cwd.")
    async def read_file(*, path: Path, ctx: ToolContext) -> str:
        return f"read {path}"

    assert isinstance(read_file, Tool)
    assert read_file.name == "read_file"
    assert read_file.capability == "fs.read"
    assert read_file.description == "Read a file under cwd."
    assert "path" in read_file.input_schema
    assert read_file.input_schema["path"] is Path


def test_tool_unknown_capability_rejected_at_registry() -> None:
    @tool(capability="made.up", description="bad")
    async def bad_tool(*, ctx: ToolContext) -> None:  # type: ignore[no-untyped-def]
        pass

    with pytest.raises(ToolValidationError, match="unknown capability"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([bad_tool])


def test_tool_sync_function_rejected_at_registry() -> None:
    def sync_fn(*, path: Path, ctx: ToolContext) -> str:  # type: ignore[misc]
        return str(path)

    t = Tool(
        name="sync_fn",
        capability="fs.read",
        description="sync",
        input_schema={"path": Path},
        handler=sync_fn,
    )
    with pytest.raises(ToolValidationError, match="must be async"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([t])


def test_tool_missing_required_param_rejected_at_registry() -> None:
    @tool(capability="fs.read", description="missing path")
    async def no_path(*, ctx: ToolContext) -> str:  # type: ignore[no-untyped-def]
        return ""

    with pytest.raises(ToolValidationError, match="missing required parameter"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([no_path])


def test_tool_wrong_required_param_type_rejected_at_registry() -> None:
    @tool(capability="fs.read", description="wrong type")
    async def wrong_type(*, path: str, ctx: ToolContext) -> str:
        return path

    with pytest.raises(ToolValidationError, match="has type"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([wrong_type])


def test_tool_missing_param_annotation_rejected() -> None:
    t = Tool(
        name="read_file",
        capability="fs.read",
        description="no annotation on path",
        input_schema={},
        handler=_missing_annotation_handler(),
    )
    with pytest.raises(ToolValidationError, match="missing type annotation"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([t])


def _missing_annotation_handler() -> Any:  # type: ignore[reportUnknownParameterType]
    async def handler(*, path, ctx: ToolContext) -> str:  # type: ignore[no-untyped-def, misc]
        return str(path)  # type: ignore[reportUnknownArgumentType]

    return handler  # type: ignore[reportUnknownVariableType]


def test_tool_missing_ctx_rejected_at_registry() -> None:
    t = Tool(
        name="no_ctx",
        capability="fs.read",
        description="no ctx",
        input_schema={"path": Path},
        handler=_no_ctx_factory(),
    )
    with pytest.raises(ToolValidationError, match="must have a 'ctx' parameter"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([t])


def _no_ctx_factory() -> Any:
    async def no_ctx(*, path: Path) -> str:  # type: ignore[no-untyped-def,misc]
        return str(path)

    return no_ctx


def test_tool_extra_optional_param_allowed() -> None:
    @tool(capability="fs.read", description="has extra optional")
    async def with_extra(*, path: Path, ctx: ToolContext, extra: int = 0) -> str:
        return str(extra)

    ToolRegistry([with_extra])


def test_tool_extra_required_param_accepted_at_registry() -> None:
    @tool(capability="fs.read", description="extra required")
    async def with_req_extra(*, path: Path, ctx: ToolContext, extra: int) -> str:
        return str(extra)

    ToolRegistry([with_req_extra])


def test_tool_return_annotation_ignored() -> None:
    @tool(capability="fs.read", description="returns int but that is fine")
    async def returns_int(*, path: Path, ctx: ToolContext) -> int:
        return 42

    ToolRegistry([returns_int])


def test_registry_multiple_tools_per_capability_accepted() -> None:
    @tool(capability="fs.read", description="first")
    async def read1(*, path: Path, ctx: ToolContext) -> str:
        return ""

    @tool(capability="fs.read", description="second")
    async def read2(*, path: Path, ctx: ToolContext) -> str:
        return ""

    registry = ToolRegistry([read1, read2])
    assert registry.get("fs.read") is not None
    assert len(registry.tools_for_capabilities({"fs.read"})) == 2


def test_registry_require_capabilities_missing_rejected() -> None:
    @tool(capability="fs.read", description="r")
    async def read_file(*, path: Path, ctx: ToolContext) -> str:
        return ""

    registry = ToolRegistry([read_file])
    with pytest.raises(MissingCapabilityError, match="Required capability"):  # type: ignore[reportUnknownMemberType]
        registry.require_capabilities(["os.shell"])


def test_registry_require_capabilities_satisfied() -> None:
    @tool(capability="fs.read", description="r")
    async def read_file(*, path: Path, ctx: ToolContext) -> str:
        return ""

    registry = ToolRegistry([read_file])
    registry.require_capabilities(["fs.read"])


async def test_registry_call_invokes_handler() -> None:
    called: list[tuple[str, str]] = []

    @tool(capability="fs.read", description="r")
    async def read_file(*, path: Path, ctx: ToolContext) -> str:
        called.append(("read_file", str(path)))
        return "ok"

    registry = ToolRegistry([read_file])
    ctx = ToolContext(workflow_id="w", stage_id="s", inputs={"cwd": Path.cwd()})
    result = await registry.call("fs.read", {"path": Path("/tmp")}, ctx)

    assert result == "ok"
    assert called == [("read_file", "/tmp")]


async def test_registry_call_missing_capability_raises() -> None:
    @tool(capability="fs.read", description="r")
    async def read_file(*, path: Path, ctx: ToolContext) -> str:
        return ""

    registry = ToolRegistry([read_file])
    ctx = ToolContext(workflow_id="w", stage_id="s", inputs={})

    with pytest.raises(MissingCapabilityError, match="No tool registered"):  # type: ignore[reportUnknownMemberType]
        await registry.call("os.shell", {"command": "ls"}, ctx)


async def test_registry_call_wraps_handler_errors() -> None:
    @tool(capability="fs.read", description="r")
    async def read_file(*, path: Path, ctx: ToolContext) -> str:
        msg = "boom"
        raise ValueError(msg)

    registry = ToolRegistry([read_file])
    ctx = ToolContext(workflow_id="w", stage_id="s", inputs={})

    with pytest.raises(ToolInvocationError, match="failed"):  # type: ignore[reportUnknownMemberType]
        await registry.call("fs.read", {"path": Path("/tmp")}, ctx)


def test_tool_context_fields() -> None:
    ctx = ToolContext(
        workflow_id="wf1",
        stage_id="s1",
        inputs={"task": "hello", "cwd": Path("/app")},
        metadata={"run": 1},
    )
    assert ctx.workflow_id == "wf1"
    assert ctx.stage_id == "s1"
    assert ctx.inputs["task"] == "hello"
    assert ctx.metadata["run"] == 1


def test_catalog_type_map_covers_all_types() -> None:
    assert CATALOG_TYPE_MAP[CapabilityParamType.STRING] is str
    assert CATALOG_TYPE_MAP[CapabilityParamType.PATH] is Path
    assert CATALOG_TYPE_MAP[CapabilityParamType.BOOL] is bool


def test_tool_context_is_frozen() -> None:
    ctx = ToolContext(workflow_id="w", stage_id="s", inputs={})
    with pytest.raises(Exception):  # type: ignore[reportUnknownMemberType]
        ctx.workflow_id = "x"  # type: ignore[misc]


# ------------------------------------------------------------------
# ToolRegistry: duplicate-name rejection + inspection helpers
# ------------------------------------------------------------------


def test_registry_rejects_duplicate_tool_name() -> None:
    async def handler1(*, path: Path, ctx: ToolContext) -> str:
        return "ok"

    async def handler2(*, command: str, ctx: ToolContext) -> str:
        return "ok"

    t1 = Tool(
        name="read",
        capability="fs.read",
        description="r1",
        input_schema={"path": Path},
        handler=handler1,
    )
    t2 = Tool(
        name="read",
        capability="os.shell",
        description="r2",
        input_schema={"command": str},
        handler=handler2,
    )
    with pytest.raises(ToolValidationError, match="Duplicate tool name"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([t1, t2])


def test_get_by_name_returns_tool() -> None:
    async def handler(*, path: Path, ctx: ToolContext) -> str:
        return "ok"

    t = Tool(
        name="reader",
        capability="fs.read",
        description="reads files",
        input_schema={"path": Path},
        handler=handler,
    )
    registry = ToolRegistry([t])
    found = registry.get_by_name("reader")
    assert found is not None
    assert found.name == "reader"
    assert found.capability == "fs.read"


def test_get_by_name_returns_none_for_unknown() -> None:
    async def handler(*, path: Path, ctx: ToolContext) -> str:
        return "ok"

    t = Tool(
        name="reader",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=handler,
    )
    registry = ToolRegistry([t])
    assert registry.get_by_name("nonexistent") is None


def test_tools_for_capabilities_filters_to_allowed() -> None:
    async def h1(*, path: Path, ctx: ToolContext) -> str:
        return "read"

    async def h2(*, path: Path, content: str, ctx: ToolContext) -> None:
        return None

    async def h3(*, command: str, ctx: ToolContext) -> str:
        return "ok"

    t1 = Tool(
        name="read",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=h1,
    )
    t2 = Tool(
        name="write",
        capability="fs.write",
        description="w",
        input_schema={"path": Path, "content": str},
        handler=h2,
    )
    t3 = Tool(
        name="shell",
        capability="os.shell",
        description="s",
        input_schema={"command": str},
        handler=h3,
    )
    registry = ToolRegistry([t1, t2, t3])

    visible = registry.tools_for_capabilities({"fs.read"})
    assert len(visible) == 1
    assert visible[0].name == "read"

    visible_two = registry.tools_for_capabilities({"fs.read", "fs.write"})
    assert len(visible_two) == 2
    names = {t.name for t in visible_two}
    assert names == {"read", "write"}


def test_tools_for_capabilities_empty_returns_empty() -> None:
    async def h(*, path: Path, ctx: ToolContext) -> str:
        return "ok"

    t = Tool(
        name="read",
        capability="fs.read",
        description="r",
        input_schema={"path": Path},
        handler=h,
    )
    registry = ToolRegistry([t])
    assert registry.tools_for_capabilities(set()) == ()


# ------------------------------------------------------------------
# Medium-2 regression: unannotated tool-specific params rejected
# ------------------------------------------------------------------


def test_unannotated_required_extra_param_rejected() -> None:
    """A required tool-specific param without annotation is rejected."""

    async def handler(
        *,
        path: Path,
        ctx: ToolContext,
        extra,  # type: ignore[no-untyped-def]
    ) -> str:
        return str(extra)  # type: ignore[reportUnknownArgumentType]

    t = Tool(
        name="bad",
        capability="fs.read",
        description="bad",
        input_schema={"path": Path},
        handler=handler,  # type: ignore[reportUnknownArgumentType]
    )
    with pytest.raises(ToolValidationError, match="missing a type annotation"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([t])


def test_unannotated_optional_extra_param_rejected() -> None:
    """An optional tool-specific param without annotation is rejected."""

    async def handler(
        *,
        path: Path,
        ctx: ToolContext,
        extra="fallback",  # type: ignore[no-untyped-def]
    ) -> str:
        return str(extra)  # type: ignore[reportUnknownArgumentType]

    t = Tool(
        name="bad",
        capability="fs.read",
        description="bad",
        input_schema={"path": Path},
        handler=handler,  # type: ignore[reportUnknownArgumentType]
    )
    with pytest.raises(ToolValidationError, match="missing a type annotation"):  # type: ignore[reportUnknownMemberType]
        ToolRegistry([t])
