from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-untyped]

from nemoir_runtime.errors import ModelOutputValidationError, ModelProviderError
from nemoir_runtime.models import (
    LiteLLMModelAdapter,
    ModelRequest,
    ModelResponse,
    ModelRouter,
    ModelSpec,
    ModelToolCall,
    model_for_stage,
    normalize_model,
    normalize_stage_output,
    normalize_tool_args,
    output_schema_for_stage,
    tool_result_to_model_content,
    tool_schema,
)
from nemoir_runtime.runtime import StageSpec, WriteSpec
from nemoir_runtime.tools import Tool


class FakeAdapter:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return ModelResponse(content="{}")


# ------------------------------------------------------------------
# normalize_model
# ------------------------------------------------------------------


def test_normalize_string_creates_litellm_adapter() -> None:
    result = normalize_model("openai/gpt-4.1-mini")
    assert isinstance(result, LiteLLMModelAdapter)
    assert result.name == "openai/gpt-4.1-mini"
    assert result.temperature is None
    assert result.max_tokens is None


def test_normalize_mapping_requires_name() -> None:
    with pytest.raises(TypeError, match="string 'name' key"):  # type: ignore[reportUnknownMemberType]
        normalize_model({"temperature": 0.5})


def test_normalize_mapping_preserves_options() -> None:
    result = normalize_model(
        {"name": "openai/gpt-4.1-mini", "temperature": 0.2, "max_tokens": 4096}
    )
    assert isinstance(result, LiteLLMModelAdapter)
    assert result.name == "openai/gpt-4.1-mini"
    assert result.temperature == 0.2
    assert result.max_tokens == 4096


def test_normalize_mapping_stores_extra_in_extra() -> None:
    result = normalize_model({"name": "anthropic/claude-sonnet-4-5", "top_p": 0.9})
    assert isinstance(result, LiteLLMModelAdapter)
    assert result.extra == {"top_p": 0.9}
    assert "top_p" not in (result.temperature, result.max_tokens)


def test_normalize_custom_adapter_passthrough() -> None:
    adapter = FakeAdapter()
    result = normalize_model(adapter)
    assert result is adapter


def test_normalize_model_router_passthrough() -> None:
    router = ModelRouter(default="openai/gpt-4.1-mini")
    result = normalize_model(router)
    assert result is router


def test_normalize_model_spec() -> None:
    spec = ModelSpec(name="openai/gpt-4.1-mini", temperature=0.5, max_tokens=2048)
    result = normalize_model(spec)
    assert isinstance(result, LiteLLMModelAdapter)
    assert result.name == "openai/gpt-4.1-mini"
    assert result.temperature == 0.5
    assert result.max_tokens == 2048


def test_normalize_invalid_type_raises() -> None:
    with pytest.raises(TypeError, match="Invalid model config"):  # type: ignore[reportUnknownMemberType]
        normalize_model(42)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# model_for_stage
# ------------------------------------------------------------------


def test_router_selects_stage_specific() -> None:
    router = ModelRouter(
        default="openai/gpt-4.1-mini",
        stages={"Apply": "anthropic/claude-sonnet-4-5"},
    )
    adapter = model_for_stage(router, "Apply")
    assert isinstance(adapter, LiteLLMModelAdapter)
    assert adapter.name == "anthropic/claude-sonnet-4-5"


def test_router_falls_back_to_default() -> None:
    router = ModelRouter(
        default="openai/gpt-4.1-mini",
        stages={"Apply": "anthropic/claude-sonnet-4-5"},
    )
    adapter = model_for_stage(router, "Triage")
    assert isinstance(adapter, LiteLLMModelAdapter)
    assert adapter.name == "openai/gpt-4.1-mini"


def test_router_default_can_be_adapter() -> None:
    fake = FakeAdapter("custom")
    router = ModelRouter(default=fake)
    adapter = model_for_stage(router, "AnyStage")
    assert adapter is fake


def test_router_stage_can_be_adapter() -> None:
    fake = FakeAdapter("stage-adapter")
    router = ModelRouter(default="openai/gpt-4.1-mini", stages={"Verify": fake})
    adapter = model_for_stage(router, "Verify")
    assert adapter is fake


def test_router_stage_can_be_mapping() -> None:
    router = ModelRouter(
        default="openai/gpt-4.1-mini",
        stages={"Plan": {"name": "openai/gpt-4o", "temperature": 0.1}},
    )
    adapter = model_for_stage(router, "Plan")
    assert isinstance(adapter, LiteLLMModelAdapter)
    assert adapter.name == "openai/gpt-4o"
    assert adapter.temperature == 0.1


def test_model_for_stage_passthrough_non_router() -> None:
    fake = FakeAdapter()
    result = model_for_stage(fake, "Triage")  # type: ignore[arg-type]
    assert result is fake


# ------------------------------------------------------------------
# LiteLLMModelAdapter stub
# ------------------------------------------------------------------


async def test_litellm_adapter_complete_content_only() -> None:
    """LiteLLM adapter with injected fake acompletion returns content."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    fake_response = type(  # type: ignore[reportUnknownVariableType]
        "FakeResponse",
        (),
        {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": '{"x":1}'})()})()]},  # type: ignore[reportUnknownMemberType]
    )
    mock_acompletion = AsyncMock(return_value=fake_response)

    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    response = await adapter.complete(request)
    assert response.content == '{"x":1}'
    assert response.tool_calls == ()
    mock_acompletion.assert_called_once()


async def test_litellm_adapter_structured_outputs_sends_json_schema() -> None:
    """When structured_outputs=True, the adapter sends json_schema response format."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    fake_response = type(  # type: ignore[reportUnknownVariableType]
        "FakeResponse",
        (),
        {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": '{"x":1}'})()})()]},  # type: ignore[reportUnknownMemberType]
    )
    mock_acompletion = AsyncMock(return_value=fake_response)
    output_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    }

    adapter = LiteLLMModelAdapter(
        ModelSpec(name="openai/gpt-4.1-mini", structured_outputs=True),
        _acompletion=mock_acompletion,
    )
    request = ModelRequest(stage_id="Test", messages=(), tools=(), output_schema=output_schema)

    response = await adapter.complete(request)
    assert response.content == '{"x":1}'
    mock_acompletion.assert_called_once()
    call_kwargs = mock_acompletion.call_args.kwargs
    rf = call_kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == output_schema
    assert rf["json_schema"]["strict"] is False


async def test_litellm_adapter_structured_outputs_false_uses_json_object() -> None:
    """When structured_outputs=False (default), the adapter sends json_object."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    fake_response = type(  # type: ignore[reportUnknownVariableType]
        "FakeResponse",
        (),
        {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": '{"x":1}'})()})()]},  # type: ignore[reportUnknownMemberType]
    )
    mock_acompletion = AsyncMock(return_value=fake_response)
    output_schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "additionalProperties": False,
    }

    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema=output_schema)

    response = await adapter.complete(request)
    assert response.content == '{"x":1}'
    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


async def test_litellm_adapter_extra_response_format_overrides_structured_outputs() -> None:
    """extra['response_format'] wins over structured_outputs."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    fake_response = type(  # type: ignore[reportUnknownVariableType]
        "FakeResponse",
        (),
        {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": '{"x":1}'})()})()]},  # type: ignore[reportUnknownMemberType]
    )
    mock_acompletion = AsyncMock(return_value=fake_response)
    output_schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "additionalProperties": False,
    }
    custom_rf = {"type": "json_object"}

    adapter = LiteLLMModelAdapter(
        ModelSpec(
            name="openai/gpt-4.1-mini",
            structured_outputs=True,
            extra={"response_format": custom_rf},
        ),
        _acompletion=mock_acompletion,
    )
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema=output_schema)

    response = await adapter.complete(request)
    assert response.content == '{"x":1}'
    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["response_format"] == custom_rf


async def test_litellm_adapter_tool_call_response_normalizes_to_model_tool_call() -> None:
    """Tool-call response normalizes to ModelToolCall with parsed JSON args.

    Covers two plan items: "Fake LiteLLM tool-call response normalizes to
    ModelToolCall" and "Tool-call JSON args are parsed".
    """
    from unittest.mock import AsyncMock  # noqa: PLC0415

    tool_call_obj = type(  # type: ignore[reportUnknownVariableType]
        "TC",
        (),
        {
            "id": "call_1",
            "type": "function",
            "function": type(
                "Func",
                (),
                {  # type: ignore[reportUnknownMemberType]
                    "name": "read",
                    "arguments": '{"path": "/tmp/x", "limit": 5}',
                },
            )(),
        },
    )()
    message_obj = type(
        "Msg",
        (),
        {  # type: ignore[reportUnknownMemberType]
            "content": None,
            "tool_calls": [tool_call_obj],
        },
    )()
    fake_response = type(
        "FakeResponse",
        (),
        {  # type: ignore[reportUnknownMemberType]
            "choices": [type("Choice", (), {"message": message_obj})()],
        },
    )()

    mock_acompletion = AsyncMock(return_value=fake_response)
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    response = await adapter.complete(request)
    assert response.content is None
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert isinstance(tc, ModelToolCall)
    assert tc.id == "call_1"
    assert tc.name == "read"
    assert tc.arguments == {"path": "/tmp/x", "limit": 5}
    assert isinstance(tc.arguments["limit"], int)


async def test_litellm_adapter_malformed_tool_call_json_raises() -> None:
    """Malformed tool-call arguments JSON raises ModelOutputValidationError."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    tool_call_obj = type(  # type: ignore[reportUnknownVariableType]
        "TC",
        (),
        {
            "id": "c1",
            "type": "function",
            "function": type(
                "Func",
                (),
                {  # type: ignore[reportUnknownMemberType]
                    "name": "t",
                    "arguments": "this is not valid json",
                },
            )(),
        },
    )()
    message_obj = type(
        "Msg",
        (),
        {  # type: ignore[reportUnknownMemberType]
            "content": None,
            "tool_calls": [tool_call_obj],
        },
    )()
    fake_response = type(
        "FakeResponse",
        (),
        {  # type: ignore[reportUnknownMemberType]
            "choices": [type("Choice", (), {"message": message_obj})()],
        },
    )()

    mock_acompletion = AsyncMock(return_value=fake_response)
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    with pytest.raises(ModelOutputValidationError, match="malformed tool-call"):  # type: ignore[reportUnknownMemberType]
        await adapter.complete(request)


async def test_litellm_adapter_provider_exception_raises_model_provider_error() -> None:
    """Provider exception raises ModelProviderError."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    mock_acompletion = AsyncMock(side_effect=RuntimeError("provider connection failed"))
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    with pytest.raises(ModelProviderError, match="LiteLLM provider error"):  # type: ignore[reportUnknownMemberType]
        await adapter.complete(request)


async def test_litellm_adapter_non_dict_tool_call_args_raises() -> None:
    """Tool-call args that parse to a non-dict raise ModelOutputValidationError."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    tool_call_obj = type(  # type: ignore[reportUnknownVariableType]
        "TC",
        (),
        {
            "id": "c1",
            "type": "function",
            "function": type(
                "Func",
                (),
                {  # type: ignore[reportUnknownMemberType]
                    "name": "t",
                    "arguments": "[1, 2, 3]",
                },
            )(),
        },
    )()
    message_obj = type(
        "Msg",
        (),
        {  # type: ignore[reportUnknownMemberType]
            "content": None,
            "tool_calls": [tool_call_obj],
        },
    )()
    fake_response = type(
        "FakeResponse",
        (),
        {  # type: ignore[reportUnknownMemberType]
            "choices": [type("Choice", (), {"message": message_obj})()],
        },
    )()

    mock_acompletion = AsyncMock(return_value=fake_response)
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    with pytest.raises(ModelOutputValidationError, match="must be an object"):  # type: ignore[reportUnknownMemberType]
        await adapter.complete(request)


async def test_litellm_adapter_no_choices_raises_model_provider_error() -> None:
    """Response with no choices raises ModelProviderError."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    fake_response = type("FakeResponse", (), {})()  # type: ignore[reportUnknownMemberType]
    mock_acompletion = AsyncMock(return_value=fake_response)
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    with pytest.raises(ModelProviderError, match="no choices"):  # type: ignore[reportUnknownMemberType]
        await adapter.complete(request)


# ------------------------------------------------------------------
# output_schema_for_stage
# ------------------------------------------------------------------


def _make_stage(*, writes: tuple[WriteSpec, ...]) -> StageSpec:
    return StageSpec(
        id="Test",
        prompt="test",
        reads=(),
        writes=writes,
        requires=frozenset(),
        transitions=(),
    )


def test_output_schema_string() -> None:
    stage = _make_stage(writes=(WriteSpec(name="summary", type="string", optional=False),))
    schema = output_schema_for_stage(stage)
    assert schema["type"] == "object"
    assert schema["properties"]["summary"] == {"type": "string"}
    assert "summary" in schema["required"]
    assert schema["additionalProperties"] is False


def test_output_schema_bool() -> None:
    stage = _make_stage(writes=(WriteSpec(name="ok", type="bool", optional=False),))
    schema = output_schema_for_stage(stage)
    assert schema["properties"]["ok"] == {"type": "boolean"}


def test_output_schema_path() -> None:
    stage = _make_stage(writes=(WriteSpec(name="file", type="path", optional=False),))
    schema = output_schema_for_stage(stage)
    assert schema["properties"]["file"] == {"type": "string"}


def test_output_schema_string_array() -> None:
    stage = _make_stage(writes=(WriteSpec(name="items", type="string[]", optional=False),))
    schema = output_schema_for_stage(stage)
    assert schema["properties"]["items"] == {"type": "array", "items": {"type": "string"}}


def test_output_schema_optional_not_required() -> None:
    stage = _make_stage(
        writes=(
            WriteSpec(name="required_f", type="string", optional=False),
            WriteSpec(name="optional_f", type="string", optional=True),
        )
    )
    schema = output_schema_for_stage(stage)
    assert "required_f" in schema["required"]
    assert "optional_f" not in schema["required"]


def test_output_schema_unsupported_type_raises() -> None:
    stage = _make_stage(writes=(WriteSpec(name="bad", type="unknown", optional=False),))
    with pytest.raises(ModelOutputValidationError, match="unsupported"):  # type: ignore[reportUnknownMemberType]
        output_schema_for_stage(stage)


def test_output_schema_no_writes() -> None:
    stage = _make_stage(writes=())
    schema = output_schema_for_stage(stage)
    assert schema == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


# ------------------------------------------------------------------
# normalize_stage_output
# ------------------------------------------------------------------


def test_normalize_stage_output_string() -> None:
    stage = _make_stage(writes=(WriteSpec(name="summary", type="string", optional=False),))
    result = normalize_stage_output(stage, {"summary": "hello"})
    assert result == {"summary": "hello"}


def test_normalize_stage_output_bool() -> None:
    stage = _make_stage(writes=(WriteSpec(name="ok", type="bool", optional=False),))
    result = normalize_stage_output(stage, {"ok": True})
    assert result == {"ok": True}


def test_normalize_stage_output_path_from_str() -> None:
    stage = _make_stage(writes=(WriteSpec(name="file", type="path", optional=False),))
    result = normalize_stage_output(stage, {"file": "/tmp/foo"})
    assert result["file"] == Path("/tmp/foo")


def test_normalize_stage_output_path_wrong_type_raises() -> None:
    stage = _make_stage(writes=(WriteSpec(name="file", type="path", optional=False),))
    with pytest.raises(ModelOutputValidationError, match="expected str"):  # type: ignore[reportUnknownMemberType]
        normalize_stage_output(stage, {"file": 42})  # type: ignore[arg-type]


def test_normalize_stage_output_string_array() -> None:
    stage = _make_stage(writes=(WriteSpec(name="items", type="string[]", optional=False),))
    result = normalize_stage_output(stage, {"items": ["a", "b"]})
    assert result == {"items": ["a", "b"]}


def test_normalize_stage_output_string_array_non_string_raises() -> None:
    stage = _make_stage(writes=(WriteSpec(name="items", type="string[]", optional=False),))
    with pytest.raises(ModelOutputValidationError, match="expected list"):  # type: ignore[reportUnknownMemberType]
        normalize_stage_output(stage, {"items": [1, 2]})  # type: ignore[arg-type]


def test_normalize_stage_output_unknown_field_raises() -> None:
    stage = _make_stage(writes=(WriteSpec(name="summary", type="string", optional=False),))
    with pytest.raises(ModelOutputValidationError, match="unknown"):  # type: ignore[reportUnknownMemberType]
        normalize_stage_output(stage, {"summary": "x", "extra": "y"})


def test_normalize_stage_output_missing_required_field_raises() -> None:
    stage = _make_stage(writes=(WriteSpec(name="summary", type="string", optional=False),))
    with pytest.raises(ModelOutputValidationError, match="missing"):  # type: ignore[reportUnknownMemberType]
        normalize_stage_output(stage, {})


def test_normalize_stage_output_optional_field_absent_is_ok() -> None:
    stage = _make_stage(
        writes=(
            WriteSpec(name="summary", type="string", optional=False),
            WriteSpec(name="extra", type="string", optional=True),
        )
    )
    result = normalize_stage_output(stage, {"summary": "s"})
    assert result == {"summary": "s", "extra": None}


def test_normalize_stage_output_optional_field_none_is_ok() -> None:
    stage = _make_stage(
        writes=(
            WriteSpec(name="summary", type="string", optional=False),
            WriteSpec(name="extra", type="string", optional=True),
        )
    )
    result = normalize_stage_output(stage, {"summary": "s", "extra": None})
    assert result == {"summary": "s", "extra": None}


def test_normalize_stage_output_string_wrong_type_raises() -> None:
    stage = _make_stage(writes=(WriteSpec(name="summary", type="string", optional=False),))
    with pytest.raises(ModelOutputValidationError, match="expected str"):  # type: ignore[reportUnknownMemberType]
        normalize_stage_output(stage, {"summary": 42})  # type: ignore[arg-type]


def test_normalize_stage_output_bool_wrong_type_raises() -> None:
    stage = _make_stage(writes=(WriteSpec(name="ok", type="bool", optional=False),))
    with pytest.raises(ModelOutputValidationError, match="expected bool"):  # type: ignore[reportUnknownMemberType]
        normalize_stage_output(stage, {"ok": "true"})  # type: ignore[arg-type]


# ------------------------------------------------------------------
# tool_schema
# ------------------------------------------------------------------


def _make_tool(
    *,
    name: str = "test",
    capability: str = "fs.read",
    description: str = "test tool",
    input_schema: dict[str, type] | None = None,
) -> Tool:
    async def handler(*, path: Path, ctx: Any) -> str:
        return "ok"

    return Tool(
        name=name,
        capability=capability,
        description=description,
        input_schema=input_schema or {"path": Path},
        handler=handler,
    )


def test_tool_schema_exposes_name_description() -> None:
    t = _make_tool(name="reader", description="Reads files")
    schema = tool_schema(t)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "reader"
    assert schema["function"]["description"] == "Reads files"


def test_tool_schema_path_param_is_string() -> None:
    t = _make_tool(input_schema={"path": Path})
    schema = tool_schema(t)
    props = schema["function"]["parameters"]["properties"]
    assert props["path"] == {"type": "string"}
    assert "ctx" not in props


def test_tool_schema_catalog_required_params_are_required() -> None:
    t = _make_tool(capability="fs.read")
    schema = tool_schema(t)
    required = schema["function"]["parameters"]["required"]
    assert "path" in required


def test_tool_schema_int_param() -> None:
    t = _make_tool(capability="fs.read", input_schema={"path": Path, "limit": int})
    schema = tool_schema(t)
    props = schema["function"]["parameters"]["properties"]
    assert props["path"] == {"type": "string"}
    assert props["limit"] == {"type": "integer"}
    assert "limit" not in schema["function"]["parameters"]["required"]


def test_tool_schema_bool_param() -> None:
    t = _make_tool(capability="user.confirm", input_schema={"message": str, "verbose": bool})
    schema = tool_schema(t)
    props = schema["function"]["parameters"]["properties"]
    assert props["verbose"] == {"type": "boolean"}


def test_tool_schema_list_str_param() -> None:
    t = _make_tool(capability="fs.read", input_schema={"paths": list[str]})
    schema = tool_schema(t)
    props = schema["function"]["parameters"]["properties"]
    assert props["paths"] == {"type": "array", "items": {"type": "string"}}


def test_tool_schema_unknown_param_type_raises() -> None:
    t = _make_tool(capability="fs.read", input_schema={"data": bytes})  # type: ignore[arg-type]
    with pytest.raises(ModelOutputValidationError, match="unsupported tool parameter"):  # type: ignore[reportUnknownMemberType]
        tool_schema(t)


# ------------------------------------------------------------------
# normalize_tool_args
# ------------------------------------------------------------------


def test_normalize_tool_args_path_from_str() -> None:
    t = _make_tool(input_schema={"path": Path})
    result = normalize_tool_args(t, {"path": "/tmp/test"})
    assert result["path"] == Path("/tmp/test")


def test_normalize_tool_args_str_passthrough() -> None:
    t = _make_tool(capability="user.confirm", input_schema={"message": str})
    result = normalize_tool_args(t, {"message": "hello"})
    assert result["message"] == "hello"


def test_normalize_tool_args_bool_rejects_str() -> None:
    t = _make_tool(capability="user.confirm", input_schema={"message": str, "verbose": bool})
    with pytest.raises(ModelOutputValidationError, match="expected bool"):  # type: ignore[reportUnknownMemberType]
        normalize_tool_args(t, {"message": "test", "verbose": "true"})


def test_normalize_tool_args_int_rejects_bool() -> None:
    t = _make_tool(capability="fs.read", input_schema={"path": Path, "limit": int})
    with pytest.raises(ModelOutputValidationError, match="expected int"):  # type: ignore[reportUnknownMemberType]
        normalize_tool_args(t, {"path": "/tmp", "limit": True})


def test_normalize_tool_args_float_from_int() -> None:
    t = _make_tool(capability="fs.read", input_schema={"path": Path, "score": float})
    result = normalize_tool_args(t, {"path": "/tmp", "score": 3})
    assert result["score"] == 3.0
    assert isinstance(result["score"], float)


def test_normalize_tool_args_unknown_arg_rejected() -> None:
    t = _make_tool(input_schema={"path": Path})
    with pytest.raises(ModelOutputValidationError, match="unknown argument"):  # type: ignore[reportUnknownMemberType]
        normalize_tool_args(t, {"path": "/tmp", "extra": "val"})


def test_normalize_tool_args_optional_param_omitted() -> None:
    t = _make_tool(capability="fs.read", input_schema={"path": Path, "limit": int})
    result = normalize_tool_args(t, {"path": "/tmp"})
    assert "path" in result
    assert "limit" not in result


def test_normalize_tool_args_missing_required_path_raises() -> None:
    t = _make_tool(capability="fs.read", input_schema={"path": Path})
    with pytest.raises(ModelOutputValidationError, match="missing required"):  # type: ignore[reportUnknownMemberType]
        normalize_tool_args(t, {})


def test_normalize_tool_args_missing_required_content_raises() -> None:
    t = _make_tool(capability="fs.write", input_schema={"path": Path, "content": str})
    with pytest.raises(ModelOutputValidationError, match="missing required"):  # type: ignore[reportUnknownMemberType]
        normalize_tool_args(t, {"path": "/tmp"})


# ------------------------------------------------------------------
# tool_result_to_model_content
# ------------------------------------------------------------------


def test_tool_result_none() -> None:
    assert tool_result_to_model_content(None) == "null"


def test_tool_result_str() -> None:
    assert tool_result_to_model_content("hello") == "hello"


def test_tool_result_bool_false() -> None:
    assert tool_result_to_model_content(False) == "false"  # noqa: FBT003


def test_tool_result_int() -> None:
    assert tool_result_to_model_content(42) == "42"


def test_tool_result_float() -> None:
    assert tool_result_to_model_content(3.14) == "3.14"


def test_tool_result_path() -> None:
    assert tool_result_to_model_content(Path("/tmp/foo")) == "/tmp/foo"


def test_tool_result_list() -> None:
    assert tool_result_to_model_content(["a", "b"]) == '["a", "b"]'


def test_tool_result_dict() -> None:
    assert tool_result_to_model_content({"key": "val"}) == '{"key": "val"}'


def test_tool_result_dataclass() -> None:
    @dataclass
    class Result:
        x: int
        y: str

    content = tool_result_to_model_content(Result(x=1, y="ok"))
    parsed = json.loads(content)
    assert parsed == {"x": 1, "y": "ok"}


def test_tool_result_fallback_str() -> None:
    assert tool_result_to_model_content([1, 2]) == "[1, 2]"


# ------------------------------------------------------------------
# LiteLLM stream normalization (Phase 5 review Medium-2 + Medium-C3)
# ------------------------------------------------------------------


def _make_attr_chunk(content_text: str | None = None, tool_calls: Any = None) -> Any:
    """Create an attribute-shaped chunk resembling a LiteLLM streaming delta."""
    delta_kwargs: dict[str, Any] = {}
    if content_text is not None:
        delta_kwargs["content"] = content_text
    if tool_calls is not None:
        delta_kwargs["tool_calls"] = tool_calls
    delta = type("Delta", (), delta_kwargs)()
    choice = type("Choice", (), {"delta": delta})()
    return type("Chunk", (), {"choices": [choice]})()


def _make_dict_chunk(content_text: str | None = None, tool_calls: Any = None) -> dict[str, Any]:
    """Create a dict-shaped chunk with the same logical shape."""
    delta: dict[str, Any] = {}
    if content_text is not None:
        delta["content"] = content_text
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"delta": delta}]}


def _make_tc_delta(*, index: int, id_: str, name: str, arguments: str) -> Any:
    """Create an attribute-shaped tool-call delta."""
    func = type("Func", (), {"name": name, "arguments": arguments})()
    return [type("TCDelta", (), {"index": index, "id": id_, "function": func})()]


def _make_tc_delta_dict(*, index: int, id_: str, name: str, arguments: str) -> list[dict[str, Any]]:
    """Create a dict-shaped tool-call delta."""
    return [{"index": index, "id": id_, "function": {"name": name, "arguments": arguments}}]


async def _aiter(items: list[Any]) -> Any:
    """Yield items from a list as an async iterable."""
    for item in items:
        yield item


async def test_litellm_normalize_stream_content_deltas() -> None:
    """Content deltas from attribute-shaped chunks produce ModelStreamChunk values."""
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini")
    response = _aiter(
        [
            _make_attr_chunk(content_text="Hello "),
            _make_attr_chunk(content_text="world"),
        ]
    )

    chunks: list[Any] = []
    async for chunk in adapter._normalize_stream(response, "test"):  # type: ignore[reportPrivateUsage]  # noqa: SLF001
        chunks.append(chunk)

    deltas = [c for c in chunks if c.kind == "delta"]
    assert len(deltas) == 2
    assert deltas[0].text == "Hello "
    assert deltas[0].channel == "assistant"
    assert deltas[1].text == "world"

    completed = [c for c in chunks if c.kind == "completed"]
    assert len(completed) == 1
    assert completed[0].response is not None
    assert completed[0].response.content == "Hello world"


async def test_litellm_normalize_stream_dict_shaped_chunks() -> None:
    """Dict-shaped chunks work through _chunk_field helper."""
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini")
    response = _aiter(
        [
            _make_dict_chunk(content_text="Hi "),
            _make_dict_chunk(content_text="there"),
        ]
    )

    chunks: list[Any] = []
    async for chunk in adapter._normalize_stream(response, "test"):  # type: ignore[reportPrivateUsage]  # noqa: SLF001
        chunks.append(chunk)

    deltas = [c for c in chunks if c.kind == "delta"]
    assert len(deltas) == 2
    assert deltas[0].text == "Hi "
    assert deltas[1].text == "there"

    completed = [c for c in chunks if c.kind == "completed"]
    assert len(completed) == 1
    assert completed[0].response.content == "Hi there"


async def test_litellm_normalize_stream_assembles_tool_calls() -> None:
    """Streamed tool-call deltas assemble into ModelToolCall objects."""
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini")
    tc_delta = _make_tc_delta(index=0, id_="call_1", name="read", arguments='{"path": "/tmp/x"}')
    response = _aiter(
        [
            _make_attr_chunk(tool_calls=tc_delta),
        ]
    )

    chunks: list[Any] = []
    async for chunk in adapter._normalize_stream(response, "test"):  # type: ignore[reportPrivateUsage]  # noqa: SLF001
        chunks.append(chunk)

    completed = [c for c in chunks if c.kind == "completed"]
    assert len(completed) == 1
    tc = completed[0].response.tool_calls
    assert len(tc) == 1
    assert tc[0].name == "read"
    assert tc[0].id == "call_1"
    assert tc[0].arguments == {"path": "/tmp/x"}


async def test_litellm_normalize_stream_tool_calls_dict_shaped() -> None:
    """Dict-shaped tool-call deltas also assemble correctly."""
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini")
    tc_delta = _make_tc_delta_dict(
        index=0, id_="call_2", name="write", arguments='{"path": "/tmp/y", "content": "hi"}'
    )
    response = _aiter(
        [
            _make_dict_chunk(tool_calls=tc_delta),
        ]
    )

    chunks: list[Any] = []
    async for chunk in adapter._normalize_stream(response, "test"):  # type: ignore[reportPrivateUsage]  # noqa: SLF001
        chunks.append(chunk)

    completed = [c for c in chunks if c.kind == "completed"]
    assert len(completed) == 1
    tc = completed[0].response.tool_calls
    assert len(tc) == 1
    assert tc[0].name == "write"
    assert tc[0].arguments == {"path": "/tmp/y", "content": "hi"}


async def test_litellm_normalize_stream_malformed_tool_args_raises() -> None:
    """Malformed JSON in streamed tool-call arguments raises ModelOutputValidationError."""
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini")
    tc_delta = _make_tc_delta(index=0, id_="call_1", name="read", arguments="not valid json!!!")
    response = _aiter(
        [
            _make_attr_chunk(tool_calls=tc_delta),
        ]
    )

    with pytest.raises(ModelOutputValidationError, match="malformed streamed tool-call arguments"):  # type: ignore[reportUnknownMemberType]
        async for _ in adapter._normalize_stream(response, "test"):  # type: ignore[reportPrivateUsage]  # noqa: SLF001
            pass


async def test_litellm_normalize_stream_non_dict_tool_args_raises() -> None:
    """Streamed tool-call arguments that are not a dict (e.g. a list) raise."""
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini")
    tc_delta = _make_tc_delta(index=0, id_="call_1", name="read", arguments="[1, 2, 3]")
    response = _aiter(
        [
            _make_attr_chunk(tool_calls=tc_delta),
        ]
    )

    with pytest.raises(ModelOutputValidationError, match="must be an object"):  # type: ignore[reportUnknownMemberType]
        async for _ in adapter._normalize_stream(response, "test"):  # type: ignore[reportPrivateUsage]  # noqa: SLF001
            pass


async def test_litellm_stream_provider_exception_raises_model_provider_error() -> None:
    """Provider exception during streaming raises ModelProviderError."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    mock_acompletion = AsyncMock(side_effect=RuntimeError("connection dropped"))
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    with pytest.raises(ModelProviderError, match="LiteLLM provider error"):  # type: ignore[reportUnknownMemberType]
        async for _ in adapter.stream(request):
            pass


async def test_litellm_complete_unchanged_after_stream_refactor() -> None:
    """complete() still works correctly after stream refactoring."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    fake_response = type(  # type: ignore[reportUnknownVariableType]
        "FakeResponse",
        (),
        {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": '{"x":1}'})()})()]},  # type: ignore[reportUnknownMemberType]
    )
    mock_acompletion = AsyncMock(return_value=fake_response)
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    response = await adapter.complete(request)
    assert response.content == '{"x":1}'
    assert response.tool_calls == ()


async def test_litellm_stream_success_yields_model_stream_chunks() -> None:
    """Successful stream() with injected _acompletion yields ModelStreamChunk values."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    # Create an async iterable that mimics a LiteLLM streaming response.
    async def _fake_stream() -> Any:
        for chunk in [
            _make_attr_chunk(content_text="Hello "),
            _make_attr_chunk(content_text="world"),
        ]:
            yield chunk

    mock_acompletion = AsyncMock(return_value=_fake_stream())
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    chunks: list[Any] = []
    async for chunk in adapter.stream(request):
        chunks.append(chunk)

    # Verify content deltas.
    deltas = [c for c in chunks if c.kind == "delta"]
    assert len(deltas) == 2
    assert deltas[0].text == "Hello "
    assert deltas[0].channel == "assistant"
    assert deltas[1].text == "world"

    # Verify final completed chunk.
    completed = [c for c in chunks if c.kind == "completed"]
    assert len(completed) == 1
    assert completed[0].response is not None
    assert completed[0].response.content == "Hello world"

    # Verify _acompletion was called with stream=True.
    mock_acompletion.assert_called_once()
    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs.get("stream") is True


async def test_litellm_stream_mid_stream_exception_raises_model_provider_error() -> None:
    """Provider exception during chunk delivery raises ModelProviderError."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    # Fake _acompletion returning an async iterable that raises after one chunk.
    async def _failing_stream() -> Any:
        yield _make_attr_chunk(content_text="Hello ")
        msg = "mid-stream network drop"
        raise RuntimeError(msg)

    mock_acompletion = AsyncMock(return_value=_failing_stream())
    adapter = LiteLLMModelAdapter("openai/gpt-4.1-mini", _acompletion=mock_acompletion)
    request = ModelRequest(stage_id="test", messages=(), tools=(), output_schema={})

    # The connection phase succeeds; iteration fails.
    with pytest.raises(ModelProviderError, match="LiteLLM provider error"):  # type: ignore[reportUnknownMemberType]
        async for _ in adapter.stream(request):
            pass

    # Verify _acompletion was still called (connection succeeded).
    mock_acompletion.assert_called_once()
