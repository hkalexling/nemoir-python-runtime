from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from nemoir_runtime.capabilities import CAPABILITY_CATALOG
from nemoir_runtime.errors import (
    ModelOutputValidationError,
    ModelProviderError,
    ToolInvocationError,
)
from nemoir_runtime.tools import (
    _is_list_str,  # type: ignore[reportPrivateUsage]
    _is_optional_list_str,  # type: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from nemoir_runtime.events import WorkflowEventChannel
    from nemoir_runtime.runtime import StageContext, StageSpec
    from nemoir_runtime.tools import Tool, ToolRegistry


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    reasoning: str | None = None


@dataclass(frozen=True)
class ModelRequest:
    stage_id: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    output_schema: Mapping[str, Any]
    options: Mapping[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


class ModelAdapter(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True)
class ModelStreamChunk:
    kind: Literal["delta", "completed"]
    channel: WorkflowEventChannel | None = None
    text: str | None = None
    response: ModelResponse | None = None


class ModelStreamingAdapter(Protocol):
    """Optional streaming capability on top of ModelAdapter.

    Adapters must implement `ModelAdapter.complete()` to be accepted by
    `normalize_model` / `ModelStageExecutor`.  `stream()` is additive —
    when present and a consumer is attached, the executor uses it to emit
    `model_delta` workflow events live.
    """

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]: ...


def supports_streaming(adapter: object) -> bool:
    return callable(getattr(adapter, "stream", None))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    temperature: float | None = None
    max_tokens: int | None = None
    structured_outputs: bool = False
    reasoning: Literal["none", "raw"] = "none"
    extra: Mapping[str, Any] = field(default_factory=dict)  # type: ignore[reportUnknownVariableType]


@dataclass(frozen=True)
class ModelRouter:
    default: str | Mapping[str, Any] | ModelAdapter
    stages: Mapping[str, str | Mapping[str, Any] | ModelAdapter] = field(  # type: ignore[reportUnknownVariableType]
        default_factory=dict
    )


class LiteLLMModelAdapter:
    name: str
    temperature: float | None
    max_tokens: int | None
    structured_outputs: bool
    reasoning: Literal["none", "raw"]
    extra: Mapping[str, Any]

    def __init__(
        self,
        spec: str | Mapping[str, Any] | ModelSpec,
        *,
        _acompletion: Any = None,
    ) -> None:
        self._spec = _resolve_spec(spec)
        self.name = self._spec.name
        self.temperature = self._spec.temperature
        self.max_tokens = self._spec.max_tokens
        self.structured_outputs = self._spec.structured_outputs
        self.reasoning = self._spec.reasoning
        self.extra = self._spec.extra
        self._acompletion = _acompletion

    async def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs = self._completion_kwargs(request, stream=False)
        acompletion = self._acompletion or _get_litellm_acompletion()
        try:
            response = await acompletion(**kwargs)
        except Exception as e:
            msg = f"LiteLLM provider error for model '{self.name}': {e}"
            raise ModelProviderError(msg) from e
        reasoning_mode = _resolve_reasoning_mode(
            self.reasoning, request.options.get("reasoning", "none")
        )
        return _normalize_litellm_response(response, request.stage_id, reasoning=reasoning_mode)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        kwargs = self._completion_kwargs(request, stream=True)
        if "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}
        acompletion = self._acompletion or _get_litellm_acompletion()
        try:
            response = await acompletion(**kwargs)
        except Exception as e:
            msg = f"LiteLLM provider error for model '{self.name}': {e}"
            raise ModelProviderError(msg) from e
        try:
            async for chunk in self._normalize_stream(
                response,
                request.stage_id,
                reasoning=_resolve_reasoning_mode(
                    self.reasoning,
                    request.options.get("reasoning", "none"),
                ),
            ):
                yield chunk
        except ModelOutputValidationError:  # validation errors are not provider errors
            raise
        except Exception as e:
            msg = f"LiteLLM provider error for model '{self.name}': {e}"
            raise ModelProviderError(msg) from e

    def _completion_kwargs(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": list(request.messages),
            "stream": stream,
        }
        if request.tools:
            kwargs["tools"] = list(request.tools)
            kwargs["tool_choice"] = "auto"
        if request.output_schema and "response_format" in self.extra:
            kwargs["response_format"] = self.extra["response_format"]
        elif request.output_schema and self.structured_outputs:
            kwargs["response_format"] = _json_schema_response_format(
                request.stage_id, request.output_schema
            )
        elif request.output_schema:
            kwargs["response_format"] = {
                "type": "json_object",
            }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        kwargs.update(self.extra)
        return kwargs

    @staticmethod
    def _chunk_field(obj: Any, key: str, default: Any = None) -> Any:
        """Read a field from an attribute- or mapping-shaped chunk."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
        return getattr(obj, key, default)

    async def _normalize_stream(
        self, response: Any, stage_id: str, *, reasoning: str = "none"
    ) -> AsyncIterator[ModelStreamChunk]:
        """Normalize a LiteLLM streaming response into ModelStreamChunk values."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Accumulate tool-call deltas indexed by position.
        tool_call_builders: dict[int, dict[str, Any]] = {}

        async for chunk in response:  # type: ignore[reportUnknownVariableType]
            try:
                choices: Any = self._chunk_field(chunk, "choices") or []
            except (AttributeError, TypeError):
                continue
            if not choices:
                continue

            choice: Any = choices[0]  # type: ignore[reportUnknownVariableType]
            delta = self._chunk_field(choice, "delta")
            if delta is None:
                continue

            # Raw provider reasoning (e.g. delta.reasoning_content in
            # DeepSeek/Qwen).  LiteLLM also normalises the Cerebras/Groq
            # ``reasoning`` alias onto ``reasoning_content`` upstream.
            # Forwarded on the new ``reasoning`` channel only when the
            # effective mode is ``"raw"`` (opt-in).  Reasoning text is
            # kept in a separate buffer so it never pollutes the final
            # structured-output ``ModelResponse.content``.
            if reasoning == "raw":
                delta_reasoning = self._chunk_field(delta, "reasoning_content")
                if delta_reasoning:
                    reasoning_str = str(delta_reasoning)
                    reasoning_parts.append(reasoning_str)
                    yield ModelStreamChunk(kind="delta", channel="reasoning", text=reasoning_str)

            # Text delta.
            delta_content = self._chunk_field(delta, "content")
            if delta_content:
                content_parts.append(str(delta_content))
                yield ModelStreamChunk(kind="delta", channel="assistant", text=str(delta_content))

            # Tool-call deltas.
            raw_tool_calls = self._chunk_field(delta, "tool_calls") or []  # type: ignore[reportUnknownVariableType]
            for tc_delta in raw_tool_calls:  # type: ignore[reportUnknownVariableType]
                idx = self._chunk_field(tc_delta, "index", 0)
                builder = tool_call_builders.setdefault(
                    idx, {"id": "", "name": "", "arguments": ""}
                )
                tc_id = self._chunk_field(tc_delta, "id")
                if tc_id:
                    builder["id"] = str(tc_id)
                func = self._chunk_field(tc_delta, "function")
                if func is not None:
                    fn_name = self._chunk_field(func, "name")
                    if fn_name:
                        builder["name"] = str(fn_name)
                    fn_args = self._chunk_field(func, "arguments")
                    if fn_args:
                        builder["arguments"] += str(fn_args)

        # Assemble final response.
        final_content = "".join(content_parts) if content_parts else None

        tool_calls_list: list[ModelToolCall] = []
        for idx in sorted(tool_call_builders.keys()):
            builder = tool_call_builders[idx]
            tc_name = builder["name"]
            tc_args_str = builder["arguments"]
            tc_id = builder["id"] or f"call_{idx}"
            if tc_name:
                try:
                    args: Any = json.loads(tc_args_str) if tc_args_str else {}
                except json.JSONDecodeError as e:
                    msg = (
                        f"model returned malformed streamed tool-call arguments "
                        f"in stage '{stage_id}': {e}"
                    )
                    raise ModelOutputValidationError(msg) from e
                if not isinstance(args, dict):
                    msg = (
                        f"streamed tool-call arguments must be an object, got {type(args).__name__}"
                    )
                    raise ModelOutputValidationError(msg)
                tool_calls_list.append(
                    ModelToolCall(id=tc_id, name=tc_name, arguments=args)  # type: ignore[reportUnknownArgumentType]
                )

        final_response = ModelResponse(
            content=final_content,
            tool_calls=tuple(tool_calls_list),
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
        )
        yield ModelStreamChunk(kind="completed", response=final_response)


def _resolve_reasoning_mode(
    adapter_reasoning: str,
    override: str,
) -> str:
    """Resolve the effective reasoning mode for a LiteLLM request.

    *adapter_reasoning* comes from ``ModelSpec.reasoning`` (stored on the
    adapter).  *override* comes from ``RunOptions.reasoning`` via
    ``request.options["reasoning"]``.  The override wins when it is not
    ``"none"``; otherwise the adapter default is used.

    Returns ``"raw"`` or ``"none"``.
    """
    if override and override != "none":
        return override
    return adapter_reasoning


def _resolve_spec(config: str | Mapping[str, Any] | ModelSpec) -> ModelSpec:
    if isinstance(config, ModelSpec):
        return config
    if isinstance(config, str):
        return ModelSpec(name=config)
    if isinstance(config, dict):
        name = config.get("name")
        if not name or not isinstance(name, str):
            msg = "model config mapping must have a string 'name' key"
            raise TypeError(msg)
        reserved = {"name", "temperature", "max_tokens", "structured_outputs", "reasoning"}
        extra = {k: v for k, v in config.items() if k not in reserved}

        reasoning_raw = config.get("reasoning", "none")
        if reasoning_raw is True or reasoning_raw == "raw":
            reasoning: Literal["none", "raw"] = "raw"
        elif reasoning_raw is False or reasoning_raw == "none":
            reasoning = "none"
        elif isinstance(reasoning_raw, str):
            reasoning = reasoning_raw  # type: ignore[assignment]
        else:
            reasoning = "none"

        return ModelSpec(
            name=name,
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens"),
            structured_outputs=config.get("structured_outputs", False),
            reasoning=reasoning,
            extra=extra,
        )
    msg = f"unsupported model spec type: {type(config).__name__}"
    raise TypeError(msg)


def _is_adapter(obj: object) -> bool:
    return callable(getattr(obj, "complete", None))  # type: ignore[arg-type]


def normalize_model(model: object) -> ModelAdapter | ModelRouter:
    if isinstance(model, ModelRouter):
        return model
    if _is_adapter(model):
        return model  # type: ignore[return-value]
    if isinstance(model, (str, dict, ModelSpec)):
        return LiteLLMModelAdapter(model)  # type: ignore[reportUnknownArgumentType]
    msg = (
        "Invalid model config: expected str, mapping, ModelSpec, "
        f"ModelRouter, or ModelAdapter, got {type(model).__name__}"
    )
    raise TypeError(msg)


def model_for_stage(model: ModelAdapter | ModelRouter, stage_id: str) -> ModelAdapter:
    if isinstance(model, ModelRouter):
        resolved = model.stages.get(stage_id, model.default)  # type: ignore[arg-type]
        result = normalize_model(resolved)
        if isinstance(result, ModelRouter):
            msg = f"ModelRouter.default for stage '{stage_id}' resolved to another ModelRouter"
            raise TypeError(msg)
        return result
    return model


# ---------------------------------------------------------------------------
# Stage output schema helpers
# ---------------------------------------------------------------------------

_WRITE_TYPE_TO_JSON: dict[str, dict[str, str | dict[str, str]]] = {
    "string": {"type": "string"},
    "bool": {"type": "boolean"},
    "path": {"type": "string"},
    "number": {"type": "number"},
    "string[]": {"type": "array", "items": {"type": "string"}},
}


def output_schema_for_stage(stage: StageSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for write in stage.writes:
        json_type = _WRITE_TYPE_TO_JSON.get(write.type)
        if json_type is None:
            msg = f"unsupported output write type '{write.type}' in stage '{stage.id}'"
            raise ModelOutputValidationError(msg)
        properties[write.name] = json_type
        if not write.optional:
            required.append(write.name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def normalize_stage_output(stage: StageSpec, raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {w.name for w in stage.writes}
    for key in raw:
        if key not in allowed:
            msg = f"unknown output field '{key}' in stage '{stage.id}'"
            raise ModelOutputValidationError(msg)
    result: dict[str, Any] = {}
    for write in stage.writes:
        val = raw.get(write.name)
        if write.optional and isinstance(val, list) and len(val) == 0 and write.type.endswith("[]"):  # type: ignore[reportUnknownArgumentType]
            val = None
        if val is None:
            if not write.optional:
                msg = f"missing required output field '{write.name}' in stage '{stage.id}'"
                raise ModelOutputValidationError(msg)
            result[write.name] = None
            continue
        result[write.name] = _normalize_write_value(write, val, stage.id)
    return result


def _normalize_write_value(write: Any, val: Any, stage_id: str) -> Any:
    if write.type == "string":
        if not isinstance(val, str):
            msg = f"expected str for '{write.name}' in stage '{stage_id}', got {type(val).__name__}"
            raise ModelOutputValidationError(msg)
        return val
    if write.type == "bool":
        if not isinstance(val, bool):
            msg = (
                f"expected bool for '{write.name}' in stage '{stage_id}', got {type(val).__name__}"
            )
            raise ModelOutputValidationError(msg)
        return val
    if write.type == "path":
        if isinstance(val, str):
            return Path(val)
        msg = (
            f"expected str or Path for '{write.name}' "
            f"in stage '{stage_id}', got {type(val).__name__}"
        )
        raise ModelOutputValidationError(msg)
    if write.type == "string[]":
        if not isinstance(val, list) or not all(
            isinstance(v, str)
            for v in val  # type: ignore[reportUnknownVariableType]
        ):
            msg = f"expected list[str] for '{write.name}' in stage '{stage_id}'"
            raise ModelOutputValidationError(msg)
        return list(val)  # type: ignore[reportUnknownArgumentType]
    if write.type == "number":
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            msg = (
                f"expected int or float (number) for '{write.name}'"
                f" in stage '{stage_id}', got {type(val).__name__}"
            )
            raise ModelOutputValidationError(msg)
        return val
    msg = f"unsupported write type '{write.type}' in stage '{stage_id}'"
    raise ModelOutputValidationError(msg)


# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------

_TOOL_ARG_TYPE_TO_JSON: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    bool: {"type": "boolean"},
    int: {"type": "integer"},
    float: {"type": "number"},
    Path: {"type": "string"},
}


def _required_tool_params(tool: Tool) -> frozenset[str]:
    spec = CAPABILITY_CATALOG.get(tool.capability)
    if spec is None:
        return frozenset()
    return frozenset(p.name for p in spec.required_params)


def _non_defaulted_params(tool: Tool) -> frozenset[str]:
    """Return the set of input_schema parameters that have no default value.

    Always computed from the handler signature intersected with
    ``tool.input_schema``, plus the catalog-required set.  This makes the
    behavior identical for ``@tool``-decorated and ad-hoc ``Tool(...)``
    instances.

    Logic: catalog-required params are always required (per plan guarantee
    they never have defaults).  Additionally, every ``input_schema`` param
    that appears in the handler signature without a default is required.
    """
    required: set[str] = set(_required_tool_params(tool))
    sig = inspect.signature(tool.handler)
    for name in tool.input_schema:
        param = sig.parameters.get(name)
        if param is not None and param.default is inspect.Parameter.empty:
            required.add(name)
    return frozenset(required)


def _param_type_to_json_schema(param_type: Any, param_name: str, tool_name: str) -> dict[str, Any]:
    """Convert a tool parameter type annotation into a JSON Schema property."""
    json_type = _TOOL_ARG_TYPE_TO_JSON.get(param_type)
    if json_type is not None:
        return json_type
    if _is_optional_list_str(param_type):
        return {"type": "array", "items": {"type": "string"}}
    if _is_list_str(param_type):
        return {"type": "array", "items": {"type": "string"}}
    msg = (
        f"unsupported tool parameter type '{getattr(param_type, '__name__', str(param_type))}' "
        f"for parameter '{param_name}' in tool '{tool_name}'"
    )
    raise ModelOutputValidationError(msg)


def tool_schema(tool: Tool) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    non_defaulted = _non_defaulted_params(tool)

    for param_name, param_type in tool.input_schema.items():
        properties[param_name] = _param_type_to_json_schema(param_type, param_name, tool.name)
        if param_name in non_defaulted:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def normalize_tool_args(tool: Tool, raw_args: Mapping[str, Any]) -> dict[str, Any]:
    known = set(tool.input_schema.keys())
    for key in raw_args:
        if key not in known:
            msg = (
                f"unknown argument '{key}' for tool '{tool.name}' (capability '{tool.capability}')"
            )
            raise ModelOutputValidationError(msg)

    # All non-defaulted params (catalog-required + tool-specific required) must
    # be present.
    non_defaulted = _non_defaulted_params(tool)
    for name in non_defaulted:
        if name not in raw_args:
            msg = (
                f"missing required argument '{name}' for tool "
                f"'{tool.name}' (capability '{tool.capability}')"
            )
            raise ModelOutputValidationError(msg)

    result: dict[str, Any] = {}
    for param_name, param_type in tool.input_schema.items():
        if param_name not in raw_args:
            continue
        val = raw_args[param_name]
        if param_type is Path:
            if isinstance(val, str):
                result[param_name] = Path(val)
            else:
                msg = (
                    f"expected str for 'Path' parameter '{param_name}' "
                    f"in tool '{tool.name}', got {type(val).__name__}"
                )
                raise ModelOutputValidationError(msg)
        elif param_type is str:
            if not isinstance(val, str):
                msg = (
                    f"expected str for parameter '{param_name}' "
                    f"in tool '{tool.name}', got {type(val).__name__}"
                )
                raise ModelOutputValidationError(msg)
            result[param_name] = val
        elif param_type is bool:
            if not isinstance(val, bool):
                msg = (
                    f"expected bool for parameter '{param_name}' "
                    f"in tool '{tool.name}', got {type(val).__name__}"
                )
                raise ModelOutputValidationError(msg)
            result[param_name] = val
        elif param_type is int:
            if not isinstance(val, int) or isinstance(val, bool):
                msg = (
                    f"expected int for parameter '{param_name}' "
                    f"in tool '{tool.name}', got {type(val).__name__}"
                )
                raise ModelOutputValidationError(msg)
            result[param_name] = val
        elif param_type is float:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                msg = (
                    f"expected float for parameter '{param_name}' "
                    f"in tool '{tool.name}', got {type(val).__name__}"
                )
                raise ModelOutputValidationError(msg)
            result[param_name] = float(val)
        elif _is_list_str(param_type):
            if not isinstance(val, list) or not all(
                isinstance(v, str)
                for v in val  # type: ignore[reportUnknownVariableType]
            ):
                msg = f"expected list[str] for parameter '{param_name}' in tool '{tool.name}'"
                raise ModelOutputValidationError(msg)
            result[param_name] = list(val)  # type: ignore[reportUnknownArgumentType]
        elif _is_optional_list_str(param_type):
            if val is None:
                result[param_name] = None
            elif isinstance(val, list) and all(
                isinstance(v, str)
                for v in val  # type: ignore[reportUnknownVariableType]
            ):
                result[param_name] = list(val)  # type: ignore[reportUnknownArgumentType]
            else:
                val_type_name: str = getattr(type(val), "__name__", "unknown")  # type: ignore[reportUnknownArgumentType]
                msg = (
                    f"expected list[str] | None for parameter '{param_name}' "
                    f"in tool '{tool.name}', got {val_type_name}"
                )
                raise ModelOutputValidationError(msg)
        else:
            msg = (
                f"unsupported parameter type '{getattr(param_type, '__name__', str(param_type))}' "
                f"for parameter '{param_name}' in tool '{tool.name}'"
            )
            raise ModelOutputValidationError(msg)
    return result


def tool_result_to_model_content(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json.dumps(asdict(value))
    return str(value)


# ---------------------------------------------------------------------------
# LiteLLM helper
# ---------------------------------------------------------------------------


def _get_litellm_acompletion() -> Any:
    import litellm  # noqa: PLC0415

    return litellm.acompletion  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _json_schema_response_format(stage_id: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in stage_id) or "stage"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": safe_name[:64],
            "schema": dict(schema),
            "strict": False,
        },
    }


# ---------------------------------------------------------------------------
# ModelStageExecutor
# ---------------------------------------------------------------------------


_INVALID_CONTENT_PREVIEW_MAX = 500


class ModelStageExecutor:
    def __init__(
        self,
        *,
        model: object,
        tools: ToolRegistry,
        max_tool_rounds: int | None = 32,  # None = unlimited
    ) -> None:
        normalized = normalize_model(model)
        self._model: ModelAdapter | ModelRouter = normalized
        self._tools = tools
        self._max_tool_rounds = max_tool_rounds

    # ----------------------------------------------------------------
    # Retry / error helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _max_model_retries(opts: object) -> int:
        if isinstance(opts, dict):
            return opts.get("max_model_retries", 3)  # type: ignore[reportUnknownMemberType]
        return getattr(opts, "max_model_retries", 3)

    @staticmethod
    async def _emit_model_retry(
        emitter: object | None,
        *,
        stage_id: str,
        error_msg: str,
        category: str,
        attempt: int,
        max_retries: int,
    ) -> None:
        if emitter is None:
            return
        await emitter.emit(  # type: ignore[union-attr]
            "model_retry",
            stage_id=stage_id,
            error=error_msg,
            metadata={
                "attempt": attempt,
                "max_retries": max_retries,
                "category": category,
            },
        )

    @staticmethod
    def _stage_retry_message(
        *,
        stage: Any,
        error_msg: str,
        output_schema: Mapping[str, Any],
        invalid_content: str | None = None,
    ) -> dict[str, Any]:
        content = (
            f"The previous response for stage '{stage.id}' was invalid. "
            f"Correct the errors and retry.\n\nError:\n{error_msg}\n\n"
            f"Return only a JSON object matching this schema:\n"
            f"{json.dumps(output_schema, indent=2)}"
        )
        if invalid_content:
            preview = (
                invalid_content[:_INVALID_CONTENT_PREVIEW_MAX]
                if len(invalid_content) > _INVALID_CONTENT_PREVIEW_MAX
                else invalid_content
            )
            content += f"\n\nYour previous output was:\n{preview}"
        return {"role": "user", "content": content}

    @staticmethod
    def _tool_error_content(error_msg: str) -> str:
        return json.dumps({"ok": False, "error": error_msg, "retryable": True})

    @staticmethod
    def _canonical_tool_call_id(tc: Any, index: int) -> str:
        return tc.id or f"call_{index}"

    # ----------------------------------------------------------------
    # Main execution loop
    # ----------------------------------------------------------------

    async def execute(self, ctx: StageContext) -> dict[str, Any]:
        adapter = model_for_stage(self._model, ctx.stage.id)

        # Resolve the effective reasoning mode: RunOptions overrides adapter.
        opts = ctx.options
        if isinstance(opts, dict):
            effective_reasoning: str = opts.get("reasoning", "none")  # type: ignore[union-attr]
        else:
            effective_reasoning = opts.reasoning
        if effective_reasoning == "none":
            effective_reasoning = getattr(adapter, "reasoning", "none")

        max_retries = self._max_model_retries(opts)
        retry_count = 0

        output_schema = output_schema_for_stage(ctx.stage)
        stage_tools = self._tools.tools_for_capabilities(ctx.allowed_capabilities)
        tool_schemas = tuple(tool_schema(t) for t in stage_tools)

        messages: list[dict[str, Any]] = self._build_initial_messages(ctx, output_schema)

        emitter = ctx.event_emitter
        use_streaming = emitter is not None and emitter.has_sink and supports_streaming(adapter)

        tool_rounds = 0
        while True:
            request = ModelRequest(
                stage_id=ctx.stage.id,
                messages=tuple(messages),
                tools=tool_schemas,
                output_schema=output_schema,
                options={"reasoning": effective_reasoning},
            )

            # Acquire model response; catch provider-level parse errors
            # (e.g. malformed streamed tool-call JSON) so they are retryable.
            try:
                if use_streaming:
                    response = await self._stream_adapter_response(adapter, request, ctx, emitter)
                else:
                    response = await adapter.complete(request)
                    if emitter is not None:
                        await emitter.emit(
                            "model_completed",
                            stage_id=ctx.stage.id,
                        )
            except ModelOutputValidationError as e:
                if retry_count >= max_retries:
                    raise
                retry_count += 1
                await self._emit_model_retry(
                    emitter,
                    stage_id=ctx.stage.id,
                    error_msg=str(e),
                    category="model_tool_call_parse",
                    attempt=retry_count,
                    max_retries=max_retries,
                )
                messages.append(
                    self._stage_retry_message(
                        stage=ctx.stage,
                        error_msg=str(e),
                        output_schema=output_schema,
                    )
                )
                continue

            if response.tool_calls:
                if self._max_tool_rounds is not None and tool_rounds >= self._max_tool_rounds:
                    msg = f"stage '{ctx.stage.id}' exceeded max_tool_rounds={self._max_tool_rounds}"
                    raise ModelOutputValidationError(msg)
                tool_rounds += 1
                messages.append(self._assistant_tool_call_message(response))

                # Execute every tool call; report errors individually and
                # retry at the response level when any call fails.
                had_error = False
                first_error_msg = ""
                for i, tc in enumerate(response.tool_calls):
                    tc_id = self._canonical_tool_call_id(tc, i)
                    try:
                        tool = self._tools.get_by_name(tc.name)
                        if tool is None:
                            msg = (
                                f"model requested unknown tool "
                                f"'{tc.name}' in stage '{ctx.stage.id}'"
                            )
                            raise ModelOutputValidationError(msg)  # noqa: TRY301
                        if tool.capability not in ctx.allowed_capabilities:
                            msg = (
                                f"tool '{tc.name}' has capability "
                                f"'{tool.capability}' which is not allowed "
                                f"in stage '{ctx.stage.id}'"
                            )
                            raise ModelOutputValidationError(msg)  # noqa: TRY301
                        normalized_args = normalize_tool_args(tool, tc.arguments)
                        result = await ctx.call_tool(
                            tool.capability, normalized_args, tool_name=tool.name
                        )
                        result_content = tool_result_to_model_content(result)
                        messages.append(self._tool_result_message(tc_id, result_content))
                    except ModelOutputValidationError as e:
                        if not had_error:
                            first_error_msg = str(e)
                        had_error = True
                        messages.append(
                            self._tool_result_message(tc_id, self._tool_error_content(str(e)))
                        )
                    except ToolInvocationError as e:
                        if not had_error:
                            first_error_msg = str(e)
                        had_error = True
                        messages.append(
                            self._tool_result_message(tc_id, self._tool_error_content(str(e)))
                        )

                if had_error:
                    if retry_count >= max_retries:
                        raise ModelOutputValidationError(first_error_msg)
                    retry_count += 1
                    await self._emit_model_retry(
                        emitter,
                        stage_id=ctx.stage.id,
                        error_msg="One or more tool calls failed validation or execution",
                        category="tool_args",
                        attempt=retry_count,
                        max_retries=max_retries,
                    )
                    continue
            else:
                # Parse and validate final stage output.
                try:
                    if not response.content:
                        msg = f"model returned empty content in stage '{ctx.stage.id}'"
                        raise ModelOutputValidationError(msg)  # noqa: TRY301
                    try:
                        parsed = json.loads(response.content)  # type: ignore[reportUnknownArgumentType]
                    except json.JSONDecodeError as e:
                        msg = f"model returned invalid JSON in stage '{ctx.stage.id}': {e}"
                        raise ModelOutputValidationError(msg) from e
                    if not isinstance(parsed, dict):
                        msg = (
                            f"model returned {type(parsed).__name__} "
                            f"instead of object in stage '{ctx.stage.id}'"
                        )
                        raise ModelOutputValidationError(msg)  # noqa: TRY301
                    return normalize_stage_output(ctx.stage, parsed)  # type: ignore[reportUnknownArgumentType]
                except ModelOutputValidationError as e:
                    if retry_count >= max_retries:
                        raise
                    retry_count += 1
                    await self._emit_model_retry(
                        emitter,
                        stage_id=ctx.stage.id,
                        error_msg=str(e),
                        category="stage_output",
                        attempt=retry_count,
                        max_retries=max_retries,
                    )
                    messages.append(
                        self._stage_retry_message(
                            stage=ctx.stage,
                            error_msg=str(e),
                            output_schema=output_schema,
                            invalid_content=response.content or None,
                        )
                    )
                    continue

    @staticmethod
    async def _stream_adapter_response(
        adapter: Any,
        request: ModelRequest,
        ctx: StageContext,
        emitter: Any,
    ) -> ModelResponse:
        """Consume a streaming adapter and emit workflow events for deltas."""
        final_response: ModelResponse | None = None
        async for chunk in adapter.stream(request):
            if chunk.kind == "delta":
                await emitter.emit(
                    "model_delta",
                    stage_id=ctx.stage.id,
                    channel=chunk.channel,
                    text=chunk.text,
                )
            elif chunk.kind == "completed":
                final_response = chunk.response
                await emitter.emit(
                    "model_completed",
                    stage_id=ctx.stage.id,
                )
        if final_response is None:
            msg = f"streaming adapter returned no completed chunk for stage '{ctx.stage.id}'"
            raise ModelOutputValidationError(msg)
        return final_response

    @staticmethod
    def _build_initial_messages(
        ctx: StageContext,
        output_schema: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        system_msg: dict[str, Any] = {
            "role": "system",
            "content": (
                "You are executing one NemoIR workflow stage. "
                "Follow the stage prompt. "
                "Use only the supplied tools when needed. "
                "Return only JSON matching the required output schema "
                "when the stage is complete. "
                "Do not expose hidden chain-of-thought."
            ),
        }
        user_content = (
            f"Workflow: {ctx.workflow_id}\n"
            f"Stage: {ctx.stage.id}\n\n"
            f"Stage prompt:\n{ctx.stage.prompt}\n\n"
            f"Readable context:\n"
            f"{json.dumps(ctx.readable_context, indent=2, default=str)}\n\n"
            f"Allowed capabilities:\n"
            f"{json.dumps(list(ctx.allowed_capabilities))}\n\n"
            f"When complete, respond with a JSON object matching this schema:\n"
            f"{json.dumps(output_schema, indent=2)}"
        )
        user_msg: dict[str, Any] = {"role": "user", "content": user_content}
        return [system_msg, user_msg]

    @staticmethod
    def _assistant_tool_call_message(
        response: ModelResponse,
    ) -> dict[str, Any]:
        tool_calls_list: list[dict[str, Any]] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in response.tool_calls
        ]
        return {"role": "assistant", "tool_calls": tool_calls_list}  # type: ignore[reportUnknownVariableType]

    @staticmethod
    def _tool_result_message(
        tool_call_id: str,
        content: str,
    ) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ---------------------------------------------------------------------------
# LiteLLM response normalization
# ---------------------------------------------------------------------------


def _normalize_litellm_response(
    response: Any, stage_id: str, *, reasoning: str = "none"
) -> ModelResponse:
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError) as e:
        msg = "LiteLLM response has no choices"
        raise ModelProviderError(msg) from e

    message = getattr(choice, "message", None)
    if message is None:
        msg = "LiteLLM response choice has no message"
        raise ModelProviderError(msg)

    content: str | None = getattr(message, "content", None)

    # Raw provider reasoning on the non-streaming path.
    reasoning_text: str | None = None
    if reasoning == "raw":
        rc = getattr(message, "reasoning_content", None)
        if rc and isinstance(rc, str):
            reasoning_text = rc

    tool_calls_list: list[ModelToolCall] = []
    raw_tool_calls = getattr(message, "tool_calls", None) or []  # type: ignore[reportUnknownVariableType]

    for raw_tc in raw_tool_calls:  # type: ignore[reportUnknownVariableType]
        tc_type = getattr(raw_tc, "type", None) or "function"  # type: ignore[reportUnknownArgumentType]
        if tc_type != "function":
            continue
        func = getattr(raw_tc, "function", None)  # type: ignore[reportUnknownArgumentType]
        if func is None:
            continue
        try:
            args_str = getattr(func, "arguments", "{}")
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            if not isinstance(args, dict):
                msg = f"tool-call arguments must be an object, got {type(args).__name__}"
                raise ModelOutputValidationError(msg)
        except json.JSONDecodeError as e:
            msg = f"model returned malformed tool-call arguments in stage '{stage_id}': {e}"
            raise ModelOutputValidationError(msg) from e
        except TypeError as e:
            msg = f"model returned non-dict tool-call arguments in stage '{stage_id}': {e}"
            raise ModelOutputValidationError(msg) from e

        tool_calls_list.append(
            ModelToolCall(
                id=getattr(raw_tc, "id", ""),  # type: ignore[reportUnknownArgumentType]
                name=getattr(func, "name", ""),
                arguments=args,  # type: ignore[reportUnknownArgumentType]
            )
        )

    return ModelResponse(
        content=content,
        tool_calls=tuple(tool_calls_list),
        reasoning=reasoning_text,
    )
