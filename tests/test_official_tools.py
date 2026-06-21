"""Tests for the official tools module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.models import (
    ModelRequest,
    ModelResponse,
    ModelStageExecutor,
    ModelToolCall,
    tool_schema,
)
from nemoir_runtime.official_tools import (
    FileEditResult,
    FileReadResult,
    FileWriteResult,
    ShellResult,
    ask_user,
    confirm_user,
    edit_file,
    read_file,
    run_shell,
    write_file,
)
from nemoir_runtime.runtime import StageContext, StageSpec, WriteSpec
from nemoir_runtime.tools import Tool, ToolContext, ToolRegistry

# ------------------------------------------------------------------
# Registry acceptance tests
# ------------------------------------------------------------------


def test_all_six_tools_accepted_in_one_registry() -> None:
    registry = ToolRegistry([read_file, write_file, edit_file, run_shell, ask_user, confirm_user])
    assert registry.get("fs.read") is not None
    assert registry.get("fs.write") is not None
    assert registry.get("os.shell") is not None
    assert registry.get("user.elicit") is not None
    assert registry.get("user.confirm") is not None
    # Two fs.write tools coexist
    write_tools = registry.tools_for_capabilities({"fs.write"})
    assert len(write_tools) == 2


def test_each_tool_is_a_tool_with_expected_capability() -> None:
    for t, cap in [
        (read_file, "fs.read"),
        (write_file, "fs.write"),
        (edit_file, "fs.write"),
        (run_shell, "os.shell"),
        (ask_user, "user.elicit"),
        (confirm_user, "user.confirm"),
    ]:
        assert isinstance(t, Tool), f"{t.name} is not a Tool"
        assert t.capability == cap, f"{t.name} capability mismatch"


def test_tool_schemas_are_well_formed() -> None:
    for t in [read_file, write_file, edit_file, run_shell, ask_user, confirm_user]:
        s = tool_schema(t)
        assert s["type"] == "function"
        assert s["function"]["name"] == t.name
        assert "description" in s["function"]
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        assert isinstance(params.get("required", []), list)


# ------------------------------------------------------------------
# read_file tests
# ------------------------------------------------------------------


def _ctx(cwd: Path | None = None) -> ToolContext:
    inputs: dict[str, Any] = {}
    if cwd is not None:
        inputs["cwd"] = str(cwd)
    return ToolContext(workflow_id="w", stage_id="s", inputs=inputs)


async def test_read_file_reads_entire_small_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("line1\nline2\nline3")
    result = await read_file.handler(
        path=Path("hello.txt"), ctx=_ctx(tmp_path), offset=0, limit=2000
    )
    assert isinstance(result, FileReadResult)
    assert result.content == "line1\nline2\nline3"
    assert result.lines_returned == 3
    assert not result.truncated


async def test_read_file_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("a\nb\nc\nd\ne")
    result = await read_file.handler(path=Path("data.txt"), ctx=_ctx(tmp_path), offset=1, limit=2)
    assert result.content == "b\nc"
    assert result.lines_returned == 2
    assert result.truncated


async def test_read_file_truncated_flag(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("\n".join(str(i) for i in range(100)))
    result = await read_file.handler(path=Path("big.txt"), ctx=_ctx(tmp_path), offset=0, limit=3)
    assert result.lines_returned == 3
    assert result.truncated


async def test_read_file_absolute_path_outside_cwd(tmp_path: Path) -> None:
    """read_file does not check containment; policy handles that."""
    f = tmp_path / "outside.txt"
    f.write_text("hello")
    result = await read_file.handler(path=f, ctx=_ctx(Path("/other/cwd")))
    assert result.content == "hello"


async def test_read_file_rejects_negative_offset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="offset must be"):  # type: ignore[reportUnknownMemberType]
        await read_file.handler(path=Path("x.txt"), ctx=_ctx(tmp_path), offset=-1, limit=10)


async def test_read_file_rejects_zero_limit(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("hi")
    with pytest.raises(ValueError, match="limit must be"):  # type: ignore[reportUnknownMemberType]
        await read_file.handler(path=Path("x.txt"), ctx=_ctx(tmp_path), offset=0, limit=0)


async def test_read_file_raises_on_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directory"):  # type: ignore[reportUnknownMemberType]
        await read_file.handler(path=Path(), ctx=_ctx(tmp_path))


async def test_read_file_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):  # type: ignore[reportUnknownMemberType]
        await read_file.handler(path=Path("nope.txt"), ctx=_ctx(tmp_path))


# ------------------------------------------------------------------
# write_file tests
# ------------------------------------------------------------------


async def test_write_file_creates_new_file(tmp_path: Path) -> None:
    result = await write_file.handler(
        path=Path("new.txt"), content="hello world", ctx=_ctx(tmp_path)
    )
    assert isinstance(result, FileWriteResult)
    assert result.created
    assert (tmp_path / "new.txt").read_text() == "hello world"


async def test_write_file_overwrites_existing(tmp_path: Path) -> None:
    (tmp_path / "exist.txt").write_text("old")
    result = await write_file.handler(path=Path("exist.txt"), content="new", ctx=_ctx(tmp_path))
    assert not result.created
    assert (tmp_path / "exist.txt").read_text() == "new"


async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    result = await write_file.handler(path=Path("a/b/c/f.txt"), content="deep", ctx=_ctx(tmp_path))
    assert (tmp_path / "a" / "b" / "c" / "f.txt").read_text() == "deep"
    assert result.created


# ------------------------------------------------------------------
# edit_file tests
# ------------------------------------------------------------------


async def test_edit_file_replaces_single_occurrence(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello old world")
    result = await edit_file.handler(
        path=Path("f.txt"),
        content="old",
        new_content="new",
        ctx=_ctx(tmp_path),
        replace_all=False,
    )
    assert isinstance(result, FileEditResult)
    assert result.occurrences_replaced == 1
    assert (tmp_path / "f.txt").read_text() == "hello new world"


async def test_edit_file_replaces_all(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x x x")
    result = await edit_file.handler(
        path=Path("f.txt"),
        content="x",
        new_content="y",
        ctx=_ctx(tmp_path),
        replace_all=True,
    )
    assert result.occurrences_replaced == 3
    assert (tmp_path / "f.txt").read_text() == "y y y"


async def test_edit_file_raises_on_zero_occurrences(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("abc")
    with pytest.raises(ValueError, match="content not found"):  # type: ignore[reportUnknownMemberType]
        await edit_file.handler(
            path=Path("f.txt"),
            content="xyz",
            new_content="new",
            ctx=_ctx(tmp_path),
        )


async def test_edit_file_raises_on_multiple_without_replace_all(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x y x")
    with pytest.raises(ValueError, match="replace_all=True"):  # type: ignore[reportUnknownMemberType]
        await edit_file.handler(
            path=Path("f.txt"),
            content="x",
            new_content="z",
            ctx=_ctx(tmp_path),
            replace_all=False,
        )


async def test_edit_file_rejects_empty_content(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content must be non-empty"):  # type: ignore[reportUnknownMemberType]
        await edit_file.handler(
            path=Path("f.txt"),
            content="",
            new_content="x",
            ctx=_ctx(tmp_path),
        )


async def test_edit_file_schema_requires_new_content() -> None:
    s = tool_schema(edit_file)
    required = s["function"]["parameters"]["required"]
    assert "path" in required
    assert "content" in required
    assert "new_content" in required
    assert "replace_all" not in required


# ------------------------------------------------------------------
# run_shell tests
# ------------------------------------------------------------------


async def test_run_shell_echo(tmp_path: Path) -> None:
    result = await run_shell.handler(command="echo hello", ctx=_ctx(tmp_path), timeout=10)
    assert isinstance(result, ShellResult)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert not result.timed_out


async def test_run_shell_non_zero_exit(tmp_path: Path) -> None:
    result = await run_shell.handler(
        command="exit 1",
        ctx=_ctx(tmp_path),
        timeout=10,
    )
    assert result.exit_code == 1


async def test_run_shell_timeout(tmp_path: Path) -> None:
    result = await run_shell.handler(
        command="sleep 10",
        ctx=_ctx(tmp_path),
        timeout=0.5,
    )
    assert result.timed_out
    assert result.exit_code is None


async def test_run_shell_uses_ctx_cwd(tmp_path: Path) -> None:
    child = tmp_path / "sub"
    child.mkdir()
    (child / "marker").write_text("found")
    result = await run_shell.handler(command="cat marker", ctx=_ctx(child), timeout=10)
    assert "found" in result.stdout


async def test_run_shell_truncates_large_output(tmp_path: Path) -> None:
    result = await run_shell.handler(
        command="python3 -c \"print('x' * 250000)\"",
        ctx=_ctx(tmp_path),
        timeout=30,
    )
    assert result.truncated


async def test_run_shell_rejects_empty_command(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="command must be non-empty"):  # type: ignore[reportUnknownMemberType]
        await run_shell.handler(command="   ", ctx=_ctx(tmp_path))


# ------------------------------------------------------------------
# ask_user tests
# ------------------------------------------------------------------


async def test_ask_user_freeform_returns_trimmed_input() -> None:
    with (
        patch("nemoir_runtime.official_tools.input", return_value="  hello  "),
        patch("nemoir_runtime.official_tools.print"),
    ):
        result = await ask_user.handler(
            question="What?",
            ctx=_ctx(),
        )
    assert result == "hello"


async def test_ask_user_options_selects_by_number() -> None:
    with (
        patch("nemoir_runtime.official_tools.input", return_value="2"),
        patch("nemoir_runtime.official_tools.print"),
    ):
        result = await ask_user.handler(
            question="Pick one",
            ctx=_ctx(),
            options=["a", "b", "c"],
        )
    assert result == "b"


async def test_ask_user_options_selects_by_exact_match() -> None:
    with (
        patch("nemoir_runtime.official_tools.input", return_value="b"),
        patch("nemoir_runtime.official_tools.print"),
    ):
        result = await ask_user.handler(
            question="Pick",
            ctx=_ctx(),
            options=["a", "b", "c"],
        )
    assert result == "b"


async def test_ask_user_empty_options_treated_as_none() -> None:
    with (
        patch("nemoir_runtime.official_tools.input", return_value="yep"),
        patch("nemoir_runtime.official_tools.print"),
    ):
        result = await ask_user.handler(
            question="ok?",
            ctx=_ctx(),
            options=[],
        )
    assert result == "yep"


async def test_ask_user_raises_after_retries() -> None:
    with (
        patch("nemoir_runtime.official_tools.input", side_effect=["x", "x"]),
        patch("nemoir_runtime.official_tools.print"),
        pytest.raises(ValueError, match="invalid"),  # type: ignore[reportUnknownMemberType]
    ):
        await ask_user.handler(
            question="Pick",
            ctx=_ctx(),
            options=["a"],
        )


async def test_ask_user_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="question must be non-empty"):  # type: ignore[reportUnknownMemberType]
        await ask_user.handler(question="   ", ctx=_ctx())


# ------------------------------------------------------------------
# confirm_user tests
# ------------------------------------------------------------------


async def test_confirm_user_yes_returns_true() -> None:
    with (
        patch("nemoir_runtime.official_tools.input", return_value="yes"),
        patch("nemoir_runtime.official_tools.print"),
    ):
        result = await confirm_user.handler(
            message="ok?",
            ctx=_ctx(),
        )
    assert result is True


async def test_confirm_user_no_returns_false() -> None:
    with (
        patch("nemoir_runtime.official_tools.input", return_value=" n "),
        patch("nemoir_runtime.official_tools.print"),
    ):
        result = await confirm_user.handler(
            message="sure?",
            ctx=_ctx(),
        )
    assert result is False


async def test_confirm_user_after_retries_returns_false() -> None:
    with (
        patch("nemoir_runtime.official_tools.input", side_effect=["maybe", "huh"]),
        patch("nemoir_runtime.official_tools.print"),
    ):
        result = await confirm_user.handler(
            message="?",
            ctx=_ctx(),
        )
    assert result is False


async def test_confirm_user_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="message must be non-empty"):  # type: ignore[reportUnknownMemberType]
        await confirm_user.handler(message="   ", ctx=_ctx())


# ------------------------------------------------------------------
# ToolRegistry named routing + multi-tool per capability
# ------------------------------------------------------------------


async def test_registry_call_with_tool_name_routes_correctly() -> None:
    """call(..., tool_name=...) invokes the named tool, not the first tool."""
    called: list[str] = []

    async def h1(*, path: Path, ctx: ToolContext) -> str:
        called.append("first")
        return "first"

    async def h2(*, path: Path, ctx: ToolContext) -> str:
        called.append("second")
        return "second"

    t1 = Tool(
        name="read_a",
        capability="fs.read",
        description="a",
        input_schema={"path": Path},
        handler=h1,
    )
    t2 = Tool(
        name="read_b",
        capability="fs.read",
        description="b",
        input_schema={"path": Path},
        handler=h2,
    )
    registry = ToolRegistry([t1, t2])
    ctx = _ctx()

    result = await registry.call("fs.read", {"path": Path("/tmp")}, ctx, tool_name="read_b")
    assert result == "second"
    assert called == ["second"]


async def test_registry_call_tool_name_capability_mismatch_raises() -> None:
    registry = ToolRegistry([read_file])
    ctx = _ctx()
    with pytest.raises(Exception):
        await registry.call(
            "fs.write", {"path": Path("/tmp"), "content": "x"}, ctx, tool_name="read_file"
        )


# ------------------------------------------------------------------
# Model executor: multi-tool disambiguation
# ------------------------------------------------------------------


async def test_model_executor_disambiguates_two_fs_write_tools() -> None:
    """Model requests edit_file; executor calls edit_file, not write_file."""
    tool_calls_log: list[str] = []

    registry = ToolRegistry([write_file, edit_file])

    writes = (WriteSpec(name="summary", type="string", optional=False),)

    class FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[ModelRequest] = []

        async def complete(self, _request: ModelRequest) -> ModelResponse:
            self.calls.append(_request)
            return ModelResponse(
                content=None,
                tool_calls=(
                    ModelToolCall(
                        id="c1",
                        name="edit_file",
                        arguments={
                            "path": "/tmp/test.txt",
                            "content": "old",
                            "new_content": "new",
                        },
                    ),
                ),
            )

    fake = FakeAdapter()

    async def call_tool(
        capability: str, args: dict[str, Any], *, tool_name: str | None = None
    ) -> str:
        tool_calls_log.append(tool_name or capability)
        return "ok"

    ctx = StageContext(
        workflow_id="Test",
        stage=StageSpec(
            id="Test",
            prompt="p",
            reads=(),
            writes=writes,
            requires=frozenset({"fs.write"}),
            transitions=(),
        ),
        inputs={},
        readable_context={},
        allowed_capabilities=frozenset({"fs.write"}),
        options={},  # type: ignore[arg-type]
        call_tool=call_tool,  # type: ignore[arg-type]
    )

    # Override setTimeout to avoid timing-based test failures
    executor = ModelStageExecutor(model=fake, tools=registry)
    executor.execute = executor.execute  # no-op, just for clarity
    # We need a second response with content so the loop terminates.
    fake.calls = []

    class TwoResponseAdapter:
        def __init__(self) -> None:
            self._step = 0

        async def complete(self, _request: ModelRequest) -> ModelResponse:
            self._step += 1
            if self._step == 1:
                return ModelResponse(
                    content=None,
                    tool_calls=(
                        ModelToolCall(
                            id="c1",
                            name="edit_file",
                            arguments={
                                "path": "/tmp/test.txt",
                                "content": "old",
                                "new_content": "new",
                            },
                        ),
                    ),
                )
            return ModelResponse(content='{"summary": "done"}')

    adapter2 = TwoResponseAdapter()
    registry2 = ToolRegistry([write_file, edit_file])
    executor2 = ModelStageExecutor(model=adapter2, tools=registry2)

    result = await executor2.execute(ctx)
    assert result == {"summary": "done"}
    # edit_file was the tool that got called
    assert "edit_file" in tool_calls_log


# ------------------------------------------------------------------
# ask_user schema: list[str] | None
# ------------------------------------------------------------------


def test_ask_user_schema_options_is_array_not_required() -> None:
    s = tool_schema(ask_user)
    required = s["function"]["parameters"]["required"]
    assert "options" not in required
    props = s["function"]["parameters"]["properties"]
    assert props["options"]["type"] == "array"
    assert props["options"]["items"] == {"type": "string"}
