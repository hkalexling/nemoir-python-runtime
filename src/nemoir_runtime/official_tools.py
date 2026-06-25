"""Official, importable tool implementations for every capability in the catalog.

These are ordinary ``Tool`` objects that ship with the runtime package.  Users
pick the tools they need and add them to their own ``ToolRegistry``::

    from nemoir_runtime import ToolRegistry
    from nemoir_runtime.official_tools import read_file, write_file, run_shell

    tools = ToolRegistry([read_file, write_file, run_shell])

Official tools validate inputs and perform the operation.  They do **not**
enforce workflow policy — path containment, write confirmation, shell
allowlists, and similar authorization remain owned by NemoIR policies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nemoir_runtime.tools import ToolContext, tool

if TYPE_CHECKING:
    from collections.abc import MutableSequence

# ------------------------------------------------------------------
# Result dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class FileReadResult:
    path: str
    content: str
    offset: int
    limit: int
    lines_returned: int
    truncated: bool


@dataclass(frozen=True)
class FileWriteResult:
    path: str
    bytes_written: int
    created: bool


@dataclass(frozen=True)
class FileEditResult:
    path: str
    occurrences_replaced: int
    bytes_written: int


@dataclass(frozen=True)
class ShellResult:
    command: str
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _resolve_path(path: Path, ctx: ToolContext) -> Path:
    """Resolve a relative path against ``ctx.inputs[\"cwd\"]`` if present.

    Absolute paths are returned as-is.  This is a convenience — containment
    is enforced by workflow policies, not by the tool.
    """
    cwd = ctx.inputs.get("cwd")
    if cwd is not None and not path.is_absolute():
        return (Path(cwd) / path).resolve()
    return path.resolve()


# ------------------------------------------------------------------
# Official tools
# ------------------------------------------------------------------


@tool(capability="fs.read", description="Read a text file, or a slice of it, as lines.")
async def read_file(
    *,
    path: Path,
    ctx: ToolContext,
    offset: int = 0,
    limit: int = 2000,
) -> FileReadResult:
    if offset < 0:
        msg = f"offset must be >= 0, got {offset}"
        raise ValueError(msg)
    if limit <= 0:
        msg = f"limit must be > 0, got {limit}"
        raise ValueError(msg)

    resolved = _resolve_path(path, ctx)

    if not resolved.exists():
        msg = f"file not found: {resolved}"
        raise FileNotFoundError(msg)
    if resolved.is_dir():
        msg = f"path is a directory: {resolved}"
        raise ValueError(msg)

    lines: MutableSequence[str] = resolved.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    start = offset
    end = offset + limit
    selected = lines[start:end]
    content = "\n".join(selected)

    return FileReadResult(
        path=str(resolved),
        content=content,
        offset=offset,
        limit=limit,
        lines_returned=len(selected),
        truncated=end < total,
    )


@tool(
    capability="fs.write",
    description="Create or overwrite a text file with the given content.",
)
async def write_file(
    *,
    path: Path,
    content: str,
    ctx: ToolContext,
) -> FileWriteResult:
    resolved = _resolve_path(path, ctx)
    created = not resolved.exists()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    resolved.write_bytes(encoded)
    return FileWriteResult(
        path=str(resolved),
        bytes_written=len(encoded),
        created=created,
    )


@tool(
    capability="fs.write",
    description="Edit a text file by replacing an exact string occurrence.",
)
async def edit_file(
    *,
    path: Path,
    content: str,
    new_content: str,
    ctx: ToolContext,
    replace_all: bool = False,
) -> FileEditResult:
    if not content:
        msg = "content must be non-empty"
        raise ValueError(msg)

    resolved = _resolve_path(path, ctx)
    text = resolved.read_text(encoding="utf-8")

    if replace_all:
        count = text.count(content)
        if count == 0:
            msg = f"content not found in {resolved}"
            raise ValueError(msg)
        text = text.replace(content, new_content)
    else:
        count = text.count(content)
        if count == 0:
            msg = f"content not found in {resolved}"
            raise ValueError(msg)
        if count > 1:
            msg = (
                f"content found {count} times in {resolved}; "
                f"use replace_all=True to replace all occurrences"
            )
            raise ValueError(msg)
        text = text.replace(content, new_content, 1)

    encoded = text.encode("utf-8")
    resolved.write_bytes(encoded)
    return FileEditResult(
        path=str(resolved),
        occurrences_replaced=count if replace_all else 1,
        bytes_written=len(encoded),
    )


@tool(
    capability="os.shell",
    description="Run a shell command and return stdout, stderr, and exit code.",
)
async def run_shell(
    *,
    command: str,
    ctx: ToolContext,
    timeout: float = 30.0,  # noqa: ASYNC109
) -> ShellResult:
    if not command.strip():
        msg = "command must be non-empty"
        raise ValueError(msg)

    # Clamp timeout to a sane maximum.
    max_timeout = 300.0
    effective_timeout = min(timeout, max_timeout) if timeout > 0 else max_timeout

    cwd = ctx.inputs.get("cwd")
    cwd_str = str(cwd) if cwd is not None else str(Path.cwd())

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd_str,
        )
    except OSError as e:
        msg = f"failed to spawn process: {e}"
        raise RuntimeError(msg) from e

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=effective_timeout
        )
        timed_out = False
        exit_code = proc.returncode
    except TimeoutError:
        proc.kill()
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        except Exception:
            stdout_bytes, stderr_bytes = (b"", b"")
        timed_out = True
        exit_code = None

    max_output = 200_000
    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    truncated = len(stdout) + len(stderr) > max_output
    if truncated and (len(stdout) + len(stderr)) > 0:
        ratio = max_output / (len(stdout) + len(stderr))
        stdout = stdout[: int(len(stdout) * ratio)]
        stderr = stderr[: int(len(stderr) * ratio)]

    return ShellResult(
        command=command,
        cwd=cwd_str,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=truncated,
    )


@tool(
    capability="user.elicit",
    description="Ask the user a question and return their answer.",
)
async def ask_user(
    *,
    question: str,
    ctx: ToolContext,  # noqa: ARG001
    options: list[str] | None = None,
) -> str:
    if not question.strip():
        msg = "question must be non-empty"
        raise ValueError(msg)

    if options is not None and len(options) == 0:
        options = None

    import sys  # noqa: PLC0415

    def _io() -> str:
        print(question, file=sys.stderr)  # noqa: T201

        if options:
            for i, opt in enumerate(options, 1):
                print(f"  {i}. {opt}", file=sys.stderr)  # noqa: T201

        raw = ""
        for _attempt in range(2):
            try:
                raw = input("> ")
            except EOFError as e:
                msg = "no input available (non-interactive environment)"
                raise RuntimeError(msg) from e
            answer = raw.strip()

            if options:
                # Accept numeric selection or exact option string match.
                try:
                    idx = int(answer) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                except ValueError:
                    pass
                if answer in options:
                    return answer
                print(f"Please select 1-{len(options)} or type an option.", file=sys.stderr)  # noqa: T201
            else:
                if answer:
                    return answer
                print("Please provide a non-empty answer.", file=sys.stderr)  # noqa: T201

        msg = f"invalid selection '{raw}' after retries"
        raise ValueError(msg)

    # Run I/O on a thread so the event loop is not blocked.
    return await asyncio.to_thread(_io)


@tool(
    capability="user.confirm",
    description="Ask the user for yes/no confirmation.",
)
async def confirm_user(
    *,
    message: str,
    ctx: ToolContext,  # noqa: ARG001
) -> bool:
    if not message.strip():
        msg = "message must be non-empty"
        raise ValueError(msg)

    import sys  # noqa: PLC0415

    def _io() -> bool:
        print(f"{message} [y/n]", file=sys.stderr)  # noqa: T201

        raw = ""
        for _attempt in range(2):
            try:
                raw = input("> ")
            except EOFError as e:
                msg = "no input available (non-interactive environment)"
                raise RuntimeError(msg) from e
            answer = raw.strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            print("Please answer y/yes or n/no.", file=sys.stderr)  # noqa: T201

        # After retries, default to False for safety.
        return False

    return await asyncio.to_thread(_io)
