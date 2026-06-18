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


@dataclass(frozen=True)
class CapabilityParam:
    name: str
    type: CapabilityParamType


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    required_params: tuple[CapabilityParam, ...]

    def has_required_param(self, name: str) -> bool:
        return any(param.name == name for param in self.required_params)


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
    },
)


def get_capability(name: str) -> CapabilitySpec | None:
    return CAPABILITY_CATALOG.get(name)


def required_param_names(name: str) -> frozenset[str]:
    spec = get_capability(name)
    if spec is None:
        return frozenset()
    return frozenset(param.name for param in spec.required_params)
