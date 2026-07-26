from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


class CapabilityParamType(Enum):
    STRING = "string"
    PATH = "path"
    BOOL = "bool"
    JSON = "json"


@dataclass(frozen=True)
class CapabilityParam:
    name: str
    type: CapabilityParamType
    required: bool = True


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    required_params: tuple[CapabilityParam, ...]

    def has_required_param(self, name: str) -> bool:
        return any(param.name == name and param.required for param in self.required_params)


CAPABILITY_CATALOG: Mapping[str, CapabilitySpec] = MappingProxyType(
    {
        "fs.read": CapabilitySpec(
            name="fs.read",
            required_params=(CapabilityParam("path", CapabilityParamType.PATH),),
        ),
        "fs.write": CapabilitySpec(
            name="fs.write",
            required_params=(
                CapabilityParam("path", CapabilityParamType.PATH),
                CapabilityParam("content", CapabilityParamType.STRING),
            ),
        ),
        "os.shell": CapabilitySpec(
            name="os.shell",
            required_params=(CapabilityParam("command", CapabilityParamType.STRING),),
        ),
        "user.elicit": CapabilitySpec(
            name="user.elicit",
            required_params=(CapabilityParam("question", CapabilityParamType.STRING),),
        ),
        "user.confirm": CapabilitySpec(
            name="user.confirm",
            required_params=(CapabilityParam("message", CapabilityParamType.STRING),),
        ),
        "http.fetch": CapabilitySpec(
            name="http.fetch",
            required_params=(
                CapabilityParam("url", CapabilityParamType.STRING),
                CapabilityParam("method", CapabilityParamType.STRING),
                CapabilityParam("headers", CapabilityParamType.JSON, required=False),
                CapabilityParam("body", CapabilityParamType.JSON, required=False),
            ),
        ),
        "browser.storage.read": CapabilitySpec(
            name="browser.storage.read",
            required_params=(CapabilityParam("key", CapabilityParamType.STRING),),
        ),
        "browser.storage.write": CapabilitySpec(
            name="browser.storage.write",
            required_params=(
                CapabilityParam("key", CapabilityParamType.STRING),
                CapabilityParam("value", CapabilityParamType.JSON),
            ),
        ),
        "browser.js.run": CapabilitySpec(
            name="browser.js.run",
            required_params=(
                CapabilityParam("code", CapabilityParamType.STRING),
                CapabilityParam("input", CapabilityParamType.JSON),
            ),
        ),
    },
)


def get_capability(name: str) -> CapabilitySpec | None:
    return CAPABILITY_CATALOG.get(name)


def required_param_names(name: str) -> frozenset[str]:
    spec = get_capability(name)
    if spec is None:
        return frozenset()
    return frozenset(param.name for param in spec.required_params if param.required)
