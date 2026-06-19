from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.tools import Tool, ToolContext, ToolRegistry


@pytest.fixture  # type: ignore[reportUnknownMemberType,reportUntypedFunctionDecorator]
def make_registry_with_log() -> Any:
    def _make() -> tuple[ToolRegistry, list[tuple[str, dict[str, Any]]]]:
        calls: list[tuple[str, dict[str, Any]]] = []

        async def read_fn(*, path: Path, ctx: ToolContext) -> str:
            calls.append(("fs.read", {"path": str(path)}))
            return f"read:{path}"

        async def write_fn(*, path: Path, content: str, ctx: ToolContext) -> None:
            calls.append(("fs.write", {"path": str(path), "content": content}))

        async def confirm_fn(*, message: str, ctx: ToolContext) -> bool:
            calls.append(("user.confirm", {"message": message}))
            return True

        async def shell_fn(*, command: str, ctx: ToolContext) -> str:
            calls.append(("os.shell", {"command": command}))
            return "ok"

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
        return ToolRegistry([read_tool, write_tool, confirm_tool, shell_tool]), calls

    return _make
