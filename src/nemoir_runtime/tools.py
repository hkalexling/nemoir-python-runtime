from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemoir_runtime.capabilities import CAPABILITY_CATALOG, CapabilityParamType
from nemoir_runtime.errors import (
    MissingCapabilityError,
    NemoIRRuntimeError,
    ToolInvocationError,
    ToolValidationError,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


@dataclass(frozen=True)
class ToolContext:
    workflow_id: str
    stage_id: str
    inputs: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


CATALOG_TYPE_MAP: dict[CapabilityParamType, type] = {
    CapabilityParamType.STRING: str,
    CapabilityParamType.PATH: Path,
    CapabilityParamType.BOOL: bool,
}


@dataclass(frozen=True)
class Tool:
    name: str
    capability: str
    description: str
    input_schema: Mapping[str, type]
    handler: Callable[..., Any]


def tool(
    *,
    capability: str,
    description: str,
) -> Callable[[Callable[..., Any]], Tool]:
    def decorator(handler: Callable[..., Any]) -> Tool:
        return Tool(
            name=handler.__name__,
            capability=capability,
            description=description,
            input_schema=_derive_input_schema(handler),
            handler=handler,
        )

    return decorator


def _derive_input_schema(handler: Callable[..., Any]) -> Mapping[str, type]:
    sig = inspect.signature(handler)
    hints = typing.get_type_hints(handler)
    schema: dict[str, type] = {}
    for name in sig.parameters:
        if name == "ctx":
            continue
        if name in hints:
            schema[name] = hints[name]
    return schema


_SUPPORTED_TOOL_PARAM_TYPES: set[Any] = {
    str,
    bool,
    int,
    float,
    Path,
}


def _is_supported_tool_param_type(typ: Any) -> bool:
    """Check whether a type annotation is supported for tool parameters.

    Supported: str, bool, int, float, Path, list[str], list[str] | None,
    Optional[list[str]].
    """
    if typ in _SUPPORTED_TOOL_PARAM_TYPES:
        return True
    if _is_list_str(typ):
        return True
    return bool(_is_optional_list_str(typ))


def _is_list_str(typ: Any) -> bool:
    """Detect bare ``list[str]``."""
    origin = typing.get_origin(typ)
    if origin is not list:
        return False
    args = typing.get_args(typ)
    return args == (str,)


def _is_optional_list_str(typ: Any) -> bool:
    """Detect ``list[str] | None`` or ``Optional[list[str]]``."""
    origin = typing.get_origin(typ)
    if origin is None:
        return False
    # Both Union and UnionType (py 3.10+ `| None`) share the same pattern.
    if origin not in (typing.Union, type(int | None)):  # UnionType
        return False
    args = typing.get_args(typ)
    # Exactly two args: one must be NoneType, the other must be list[str].
    if len(args) != 2:  # noqa: PLR2004
        return False
    type_none = type(None)
    found_none = False
    found_list_str = False
    for arg in args:
        if arg is type_none:
            found_none = True
        elif _is_list_str(arg):
            found_list_str = True
    return found_none and found_list_str


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools_by_capability: dict[str, list[Tool]] = {}
        self._tools_by_name: dict[str, Tool] = {}
        tool_list = list(tools)
        for t in tool_list:
            if t.name in self._tools_by_name:
                msg = (
                    f"Duplicate tool name: '{t.name}' used by capability "
                    f"'{self._tools_by_name[t.name].capability}' "
                    f"and capability '{t.capability}'"
                )
                raise ToolValidationError(msg)
            self._tools_by_capability.setdefault(t.capability, []).append(t)
            self._tools_by_name[t.name] = t
        # Validate after registration so catalog lookups inside validation
        # can find tools we just stored.
        # NOTE: validation is already done in _validate_tools which uses
        # the catalog, not the registry's own storage.  We run it on the
        # original list to preserve the caller's insertion order for
        # deterministic error messages.
        self._validate_tools(tool_list)

    def get(self, capability: str) -> Tool | None:
        tools = self._tools_by_capability.get(capability)
        if tools:
            return tools[0]
        return None

    def get_by_name(self, name: str) -> Tool | None:
        return self._tools_by_name.get(name)

    def tools_for_capabilities(self, capabilities: Iterable[str]) -> tuple[Tool, ...]:
        cap_set = set(capabilities)
        return tuple(t for t in self._tools_by_name.values() if t.capability in cap_set)

    def require_capabilities(self, capabilities: Iterable[str]) -> None:
        for cap in capabilities:
            if cap not in self._tools_by_capability or not self._tools_by_capability[cap]:
                msg = f"Required capability '{cap}' has no registered tool"
                raise MissingCapabilityError(msg)

    async def call(
        self,
        capability: str,
        args: Mapping[str, Any],
        ctx: ToolContext,
        *,
        tool_name: str | None = None,
    ) -> Any:
        if tool_name is not None:
            tool = self._tools_by_name.get(tool_name)
            if tool is None:
                msg = f"No tool named '{tool_name}' registered"
                raise MissingCapabilityError(msg)
            if tool.capability != capability:
                msg = (
                    f"Tool '{tool_name}' has capability '{tool.capability}', "
                    f"but was called as '{capability}'"
                )
                raise MissingCapabilityError(msg)
        else:
            tool = self.get(capability)
            if tool is None:
                msg = f"No tool registered for capability '{capability}'"
                raise MissingCapabilityError(msg)
        try:
            return await tool.handler(**args, ctx=ctx)
        except NemoIRRuntimeError:
            raise
        except Exception as e:
            msg = f"Tool '{tool.name}' (capability '{tool.capability}') failed: {e}"
            raise ToolInvocationError(msg) from e

    @staticmethod
    def _validate_tools(tools: list[Tool]) -> None:
        for t in tools:
            spec = CAPABILITY_CATALOG.get(t.capability)
            if spec is None:
                msg = f"Tool '{t.name}' declares unknown capability '{t.capability}'"
                raise ToolValidationError(msg)
            ToolRegistry._validate_tool_params(t, spec)

    @staticmethod
    def _validate_tool_params(t: Tool, spec: Any) -> None:
        sig = inspect.signature(t.handler)
        hints = typing.get_type_hints(t.handler)

        if not inspect.iscoroutinefunction(t.handler):
            msg = f"Tool '{t.name}' must be async"
            raise ToolValidationError(msg)

        if "ctx" not in sig.parameters:
            msg = f"Tool '{t.name}' must have a 'ctx' parameter"
            raise ToolValidationError(msg)

        for param_spec in spec.required_params:
            if param_spec.name not in sig.parameters:
                expected_type_name = CATALOG_TYPE_MAP[param_spec.type].__name__
                msg = (
                    f"Tool '{t.name}' declares capability '{t.capability}' "
                    f"but is missing required parameter "
                    f"'{param_spec.name}: {expected_type_name}'"
                )
                raise ToolValidationError(msg)

        ToolRegistry._validate_param_types(t, spec, sig, hints)

    @staticmethod
    def _validate_param_types(
        t: Tool,
        spec: Any,
        sig: inspect.Signature,
        hints: dict[str, Any],
    ) -> None:
        for name in sig.parameters:
            if name == "ctx":
                continue
            is_catalog_required = any(p.name == name for p in spec.required_params)
            if is_catalog_required:
                expected_type = CATALOG_TYPE_MAP[
                    next(p.type for p in spec.required_params if p.name == name)
                ]
                actual_type = hints.get(name)
                if actual_type is None:
                    msg = (
                        f"Tool '{t.name}' parameter '{name}' is missing type annotation, "
                        f"expected '{expected_type.__name__}' for capability '{t.capability}'"
                    )
                    raise ToolValidationError(msg)
                if actual_type is not expected_type:
                    actual_name = getattr(actual_type, "__name__", str(actual_type))
                    msg = (
                        f"Tool '{t.name}' parameter '{name}' has type "
                        f"'{actual_name}', expected "
                        f"'{expected_type.__name__}' for capability "
                        f"'{t.capability}'"
                    )
                    raise ToolValidationError(msg)
            else:
                # Tool-specific parameter: required or optional is fine,
                # but the annotation must be a supported type (and must exist).
                actual_type = hints.get(name)
                if actual_type is None:
                    msg = f"Tool '{t.name}' parameter '{name}' is missing a type annotation"
                    raise ToolValidationError(msg)
                if not _is_supported_tool_param_type(actual_type):
                    actual_name = getattr(actual_type, "__name__", str(actual_type))
                    msg = (
                        f"Tool '{t.name}' parameter '{name}' has unsupported "
                        f"type annotation '{actual_name}'"
                    )
                    raise ToolValidationError(msg)
