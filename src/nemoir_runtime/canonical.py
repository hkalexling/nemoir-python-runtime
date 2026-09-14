"""RFC 8785 JSON Canonicalization Scheme (JCS) + NemoTrace content hashing.

Pure-stdlib Python side of the shared cross-language contract defined in
``docs/trace/schema/README.md`` §3. Mirrors
``compiler/crates/nemoir-ir/src/canonical.rs`` and the forthcoming
TypeScript ``canonical.ts``; all three must pass the vectors in
``docs/trace/schema/test-vectors/jcs/``.

Key subtlety: Python's ``json`` number formatting is NOT ECMAScript
``Number::toString`` (e.g. ``json.dumps(0.000001)`` is ``"1e-06"`` while JS
gives ``"0.000001"``). Floats therefore go through :func:`format_js_number`,
which re-applies the ECMA-262 placement rules to the shortest round-trip
digits from ``repr``.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

# ECMA-262 Number::toString placement thresholds (see format_js_number).
_PLAIN_MAX_EXPONENT = 21
_SMALL_NEGATIVE_BOUND = -6
_CONTROL_CHAR_MAX = 0x20


def sha256_tag(data: bytes) -> str:
    """Render bytes as ``sha256:<64 lowercase hex>``."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def format_js_number(value: float) -> str:
    """Format a finite float per ECMAScript ``Number::toString``.

    Implements ECMA-262 §6.1.6.1.20 placement from the shortest round-trip
    decimal digits (taken from ``repr``)::

    - ``k <= n <= 21``: digits plus ``n - k`` zeros
    - ``0 < n <= 21``: decimal point inserted at ``n``
    - ``-6 < n <= 0``: ``0.`` plus ``-n`` zeros plus digits
    - otherwise: single digit, optional fraction, ``e±exp``

    Raises :class:`ValueError` for non-finite values (JCS rejects them) and
    returns ``"0"`` for both ``0.0`` and ``-0.0``.
    """
    if value == 0:
        return "0"
    if value != value or value in (float("inf"), float("-inf")):  # noqa: PLR0124
        msg = "JCS rejects non-finite numbers"
        raise ValueError(msg)
    negative = value < 0
    text = repr(abs(value))
    mantissa, _, exp_text = text.partition("e")
    exponent = int(exp_text) if exp_text else 0
    int_part, _, frac_part = mantissa.partition(".")
    frac_len = len(frac_part)
    digits = int(int_part + frac_part)  # strips leading zeros
    if digits == 0:
        return "0"
    significant = str(digits).rstrip("0")
    trailing = len(str(digits)) - len(significant)
    k = len(significant)
    n = k + exponent - frac_len + trailing
    if k <= n <= _PLAIN_MAX_EXPONENT:
        out = significant + "0" * (n - k)
    elif 0 < n <= _PLAIN_MAX_EXPONENT:
        out = significant[:n] + "." + significant[n:]
    elif _SMALL_NEGATIVE_BOUND < n <= 0:
        out = "0." + "0" * (-n) + significant
    else:
        exp = n - 1
        sign = "+" if exp >= 0 else "-"
        out = significant[0]
        if k > 1:
            out += "." + significant[1:]
        out += f"e{sign}{abs(exp)}"
    return "-" + out if negative else out


def _check_string(value: str) -> None:
    """Reject lone surrogates (not valid JCS / UTF-8)."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        msg = f"JCS rejects lone surrogates: {exc}"
        raise ValueError(msg) from exc


def _escape_string(value: str) -> str:
    """JSON-escape a string per JCS (UTF-8 output, minimal escapes)."""
    _check_string(value)
    parts: list[str] = ['"']
    for char in value:
        code = ord(char)
        if char == '"':
            parts.append('\\"')
        elif char == "\\":
            parts.append("\\\\")
        elif char == "\b":
            parts.append("\\b")
        elif char == "\f":
            parts.append("\\f")
        elif char == "\n":
            parts.append("\\n")
        elif char == "\r":
            parts.append("\\r")
        elif char == "\t":
            parts.append("\\t")
        elif code < _CONTROL_CHAR_MAX:
            parts.append(f"\\u{code:04x}")
        else:
            parts.append(char)
    parts.append('"')
    return "".join(parts)


def _canonicalize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_js_number(value)
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, (list, tuple)):
        sequence = cast("Sequence[Any]", value)
        return "[" + ",".join(_canonicalize(item) for item in sequence) + "]"
    if isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        for key in mapping:
            if not isinstance(key, str):
                msg = f"JCS object keys must be strings, got {type(key).__name__}"
                raise TypeError(msg)
            _check_string(key)
        # RFC 8785 sorts by UTF-16 code units (ECMAScript order), which
        # differs from code-point order when a high-BMP key (e.g. U+FB33)
        # meets an astral key (lead surrogate U+D800-U+DBFF sorts first).
        # UTF-16-BE bytes compare unit-by-unit, so byte order == unit order.
        items = sorted(mapping.items(), key=_utf16_sort_key)
        return "{" + ",".join(_escape_string(k) + ":" + _canonicalize(v) for k, v in items) + "}"
    msg = f"unsupported JSON value: {type(value).__name__}"
    raise TypeError(msg)


def _utf16_sort_key(item: tuple[Any, Any]) -> bytes:
    """Sort key for RFC 8785 (UTF-16 code-unit order)."""
    return item[0].encode("utf-16-be")


def to_canonical_bytes(value: Any) -> bytes:
    """Canonicalize a JSON-like value to RFC 8785 bytes (no trailing LF)."""
    return _canonicalize(value).encode("utf-8")


def to_canonical_str(value: Any) -> str:
    """Canonicalize a JSON-like value to an RFC 8785 string (no trailing LF)."""
    return _canonicalize(value)


def parse_json_strict(text: str) -> Any:
    """Parse JSON, rejecting duplicate object keys (reader-side strictness)."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, val in pairs:
            if key in result:
                msg = f"duplicate JSON object key: {key}"
                raise ValueError(msg)
            result[key] = val
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates)
