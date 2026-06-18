from __future__ import annotations

from nemoir_runtime.capabilities import (
    CAPABILITY_CATALOG,
    CapabilityParamType,
    get_capability,
    required_param_names,
)


def test_catalog_has_exactly_five_capabilities() -> None:
    assert len(CAPABILITY_CATALOG) == 5


def test_catalog_contains_fs_read() -> None:
    spec = get_capability("fs.read")
    assert spec is not None
    assert spec.name == "fs.read"
    assert len(spec.required_params) == 1
    assert spec.required_params[0].name == "path"
    assert spec.required_params[0].type == CapabilityParamType.PATH


def test_catalog_contains_fs_write() -> None:
    spec = get_capability("fs.write")
    assert spec is not None
    assert spec.name == "fs.write"
    assert len(spec.required_params) == 2

    param_names = [p.name for p in spec.required_params]
    assert "path" in param_names
    assert "content" in param_names

    path_param = next(p for p in spec.required_params if p.name == "path")
    assert path_param.type == CapabilityParamType.PATH

    content_param = next(p for p in spec.required_params if p.name == "content")
    assert content_param.type == CapabilityParamType.STRING


def test_catalog_contains_os_shell() -> None:
    spec = get_capability("os.shell")
    assert spec is not None
    assert spec.name == "os.shell"
    assert len(spec.required_params) == 1
    assert spec.required_params[0].name == "command"
    assert spec.required_params[0].type == CapabilityParamType.STRING


def test_catalog_contains_user_elicit() -> None:
    spec = get_capability("user.elicit")
    assert spec is not None
    assert spec.name == "user.elicit"
    assert len(spec.required_params) == 1
    assert spec.required_params[0].name == "question"
    assert spec.required_params[0].type == CapabilityParamType.STRING


def test_catalog_contains_user_confirm() -> None:
    spec = get_capability("user.confirm")
    assert spec is not None
    assert spec.name == "user.confirm"
    assert len(spec.required_params) == 1
    assert spec.required_params[0].name == "message"
    assert spec.required_params[0].type == CapabilityParamType.STRING


def test_get_capability_returns_none_for_unknown() -> None:
    assert get_capability("made.up") is None


def test_required_param_names() -> None:
    assert required_param_names("fs.read") == frozenset({"path"})
    assert required_param_names("fs.write") == frozenset({"path", "content"})
    assert required_param_names("made.up") == frozenset()


def test_catalog_specs_have_no_description() -> None:
    spec = get_capability("fs.read")
    assert spec is not None
    assert not hasattr(spec, "description")


def test_catalog_specs_have_no_return_type() -> None:
    spec = get_capability("fs.read")
    assert spec is not None
    assert not hasattr(spec, "return_type")
