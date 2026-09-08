"""Tests for RFC 8785 canonicalization against shared NemoTrace vectors."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from nemoir_runtime.canonical import (
    format_js_number,
    parse_json_strict,
    sha256_tag,
    to_canonical_bytes,
)

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "docs" / "trace" / "schema" / "test-vectors"


def test_primitives_vector_matches_checked_in_canonical() -> None:
    d = VECTORS / "jcs"
    raw = json.loads((d / "rfc8785-primitives.input.json").read_text(encoding="utf-8"))
    assert to_canonical_bytes(raw) == (d / "rfc8785-primitives.canonical.json").read_bytes()


def test_property_order_vector_matches_checked_in_canonical() -> None:
    d = VECTORS / "jcs"
    raw = json.loads((d / "rfc8785-property-order.input.json").read_text(encoding="utf-8"))
    assert to_canonical_bytes(raw) == (d / "rfc8785-property-order.canonical.json").read_bytes()


def test_number_samples_match_ecmascript_serialization() -> None:
    d = VECTORS / "jcs"
    samples = json.loads((d / "rfc8785-number-samples.json").read_text(encoding="utf-8"))
    for sample in samples:
        number = struct.unpack(">d", bytes.fromhex(sample["ieee754"]))[0]
        expected = sample["json"]
        if expected is None:
            with pytest.raises(ValueError, match="non-finite"):
                format_js_number(number)
        else:
            assert format_js_number(number) == expected, sample["ieee754"]


def test_js_number_plain_vs_exponent_thresholds() -> None:
    # ECMAScript uses plain decimals for 1e-6 <= |n| < 1e21, unlike Python repr.
    assert format_js_number(0.000001) == "0.000001"
    assert format_js_number(0.00001) == "0.00001"
    assert format_js_number(0.0001) == "0.0001"
    assert format_js_number(1000000.0) == "1000000"
    assert format_js_number(1e21) == "1e+21"
    assert format_js_number(-0.0) == "0"
    assert format_js_number(0.0) == "0"
    assert format_js_number(333333333.33333329) == "333333333.3333333"
    assert format_js_number(4.5) == "4.5"
    assert format_js_number(0.002) == "0.002"
    assert format_js_number(-0.0000033333333333333333) == "-0.0000033333333333333333"


def test_non_finite_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        to_canonical_bytes(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        to_canonical_bytes(float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        to_canonical_bytes([1.0, float("-inf")])


def test_lone_surrogates_rejected() -> None:
    with pytest.raises(ValueError, match="lone surrogates"):
        to_canonical_bytes("\ud800")
    with pytest.raises(ValueError, match="lone surrogates"):
        to_canonical_bytes("\udc00")
    with pytest.raises(ValueError, match="lone surrogates"):
        to_canonical_bytes({"key\ud800": 1})


def test_duplicate_keys_rejected_on_parse() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_json_strict('{"a":1,"a":2}')


def test_minimal_ir_fingerprint_matches_vector() -> None:
    d = VECTORS / "ir-fingerprint"
    raw = json.loads((d / "minimal-ir.input.json").read_text(encoding="utf-8"))
    canonical = to_canonical_bytes(raw)
    assert canonical == (d / "minimal-ir.canonical.json").read_bytes()
    assert sha256_tag(canonical) == (d / "minimal-ir.sha256").read_text().strip()


def test_no_trailing_lf_and_sorted_keys() -> None:
    out = to_canonical_bytes({"b": 1, "a": [3, 2]})
    assert out == b'{"a":[3,2],"b":1}'
    assert not out.endswith(b"\n")
