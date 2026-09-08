"""Cross-language parity: the shared fixture must project byte-identical ledgers.

``docs/trace/schema/test-vectors/parity/fake-run.json`` drives the recorder
through a fixed op sequence (hooks + observations, fixed clock + trace id).
Both the Python and TypeScript recorders must emit byte-identical
``public/events.ndjson`` and ``public/workflow.graph.json`` matching the
frozen ``expected-*`` files. This is the Phase 1 exit gate's core proof:
one logical run, two recorders, identical canonical bytes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nemoir_runtime.events import WorkflowEvent
from nemoir_runtime.runtime import (
    GuardSpec,
    PolicySpec,
    StageExecutionSpec,
    StageSpec,
    TransitionSpec,
    TriggerSpec,
    WorkflowManifest,
    WriteSpec,
)
from nemoir_runtime.trace import (
    HostProvenance,
    TraceRecorder,
    read_archive_entries,
    verify_archive,
)

ROOT = Path(__file__).resolve().parents[3]
PARITY = ROOT / "docs" / "trace" / "schema" / "test-vectors" / "parity"


def _load_fixture() -> dict[str, Any]:
    return json.loads((PARITY / "fake-run.json").read_text(encoding="utf-8"))  # type: ignore[reportUnknownVariableType]


def _build_manifest(spec: dict[str, Any]) -> WorkflowManifest:
    policies = tuple(
        PolicySpec(
            id=p["id"],
            kind=p["kind"],
            trigger=TriggerSpec(capability=p["trigger_capability"], bind={}),
            requires=(),
        )
        for p in spec["policies"]
    )
    stages = tuple(
        StageSpec(
            id=s["id"],
            prompt="",
            reads=(),
            writes=tuple(
                WriteSpec(name=w["name"], type=w["type"], optional=w["optional"])
                for w in s["writes"]
            ),
            requires=frozenset(s["requires"]),
            transitions=tuple(
                TransitionSpec(
                    to=t["to"],
                    priority=t["priority"],
                    reason=t["reason"],
                    guard=GuardSpec(kind=t["guard_kind"]),
                )
                for t in s["transitions"]
            ),
            execution=StageExecutionSpec(kind=s["execution"]),
        )
        for s in spec["stages"]
    )
    capabilities: set[str] = set()
    for stage in stages:
        capabilities.update(stage.requires)
    for policy in policies:
        capabilities.add(policy.trigger.capability)
    return WorkflowManifest(
        workflow_id=spec["workflow_id"],
        entry_stage_id=spec["entry"],
        exit_stage_ids=frozenset(spec["exits"]),
        inputs=(),
        capabilities=frozenset(capabilities),
        policies=policies,
        stages=stages,
    )


def _drive_fixture(tmp_path: Path) -> Path:
    fixture = _load_fixture()
    config = fixture["config"]
    clock_at = datetime.fromisoformat(config["clock"])
    recorder = TraceRecorder.create(
        tmp_path / "parity.nemotrace",
        profile="audit",
        provenance=HostProvenance(
            frontend="parity",
            target="parity",
            compiler_version="parity",
            ir_version="0.1",
            ir_sha256="sha256:" + "ef" * 32,
        ),
        path_aliases={key: Path(value) for key, value in config["path_aliases"].items()},
        safe_path_aliases=frozenset(config["safe_path_aliases"]),
        approved_metrics=frozenset(config["approved_metrics"]),
        secrets=tuple(config["secrets"]),
        trace_id=config["trace_id"],
        clock=lambda: clock_at,
    )
    last_mid = ""
    last_tid = ""
    dummy_ts = datetime(2020, 1, 1, tzinfo=UTC)
    for op in fixture["ops"]:
        kind = op["op"]
        if kind == "begin_run":
            recorder.begin_run(_build_manifest(fixture["manifest"]))
        elif kind == "begin_stage_visit":
            recorder.begin_stage_visit(op["stage_id"])
        elif kind == "begin_model_call":
            last_mid = recorder.begin_model_call()
        elif kind == "record_model_response":
            recorder.record_model_response(
                last_mid,
                response_bytes=op["response_bytes"],
                tool_call_count=op["tool_call_count"],
            )
        elif kind == "begin_tool_call":
            last_tid = recorder.begin_tool_call()
        elif kind == "record_tool_result":
            recorder.record_tool_result(last_tid, op["result"])
        elif kind == "record_tool_error":
            error_type = type(op["error_type"], (Exception,), {})
            recorder.record_tool_error(last_tid, error_type())
        elif kind == "observe":
            spec = op["event"]
            event = WorkflowEvent(
                kind=spec["kind"],
                run_id="0" * 32,
                sequence=spec["sequence"],
                timestamp=dummy_ts,
                stage_id=spec.get("stage_id"),
                channel=spec.get("channel"),
                text=spec.get("text"),
                capability=spec.get("capability"),
                tool_name=spec.get("tool_name"),
                args=spec.get("args"),
                output=spec.get("output"),
                result=spec.get("result"),
                error=spec.get("error"),
                transition_to=spec.get("transition_to"),
                metadata=spec.get("metadata", {}),
            )
            recorder.observe_workflow_event(event)
        else:  # pragma: no cover - fixture bug
            msg = f"unknown parity op {kind!r}"
            raise AssertionError(msg)
    return recorder.finish_run("complete")


def test_parity_fixture_matches_frozen_vectors(tmp_path: Path) -> None:
    path = _drive_fixture(tmp_path)
    entries = read_archive_entries(path)
    expected_ledger = (PARITY / "expected-ledger.ndjson").read_bytes()
    expected_graph = (PARITY / "expected-graph.json").read_bytes()
    assert entries["public/events.ndjson"] == expected_ledger
    assert entries["public/workflow.graph.json"] == expected_graph
    report = verify_archive(path)
    assert report.ok, report.errors
