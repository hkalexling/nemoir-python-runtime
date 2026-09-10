"""Tests for the NemoTrace Phase 1 audit recorder and archive.

Covers the Phase 1 exit gate: canonical archives from `run()` and
`stream()`, IR/provenance binding, per-event redaction, seeded-secret
absence, terminal statuses, and unchanged live-event behavior.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from referencing import Registry, Resource  # type: ignore[import-untyped]

from nemoir_runtime import ToolContext, ToolRegistry, tool
from nemoir_runtime.errors import MaxStepsExceededError, PolicyDeniedError, ToolInvocationError
from nemoir_runtime.events import WorkflowEvent
from nemoir_runtime.models import ModelResponse, ModelStageExecutor
from nemoir_runtime.runtime import (
    ExprSpec,
    GuardSpec,
    InputSpec,
    PolicySpec,
    ReadSpec,
    RefSpec,
    RunOptions,
    StageExecutionSpec,
    StageExecutor,
    StageSpec,
    TransitionSpec,
    TriggerSpec,
    WorkflowManifest,
    WorkflowRuntime,
    WriteSpec,
)
from nemoir_runtime.trace import (
    HostProvenance,
    ModelDescriptor,
    TraceConfig,
    TraceError,
    TraceRecorder,
    read_archive_entries,
    resolve_trace_recorder,
    verify_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "docs" / "trace" / "schema"
VECTORS = SCHEMA_DIR / "test-vectors"

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXED_TRACE_ID = "0123456789abcdef0123456789abcdef"


def _fixed_clock() -> datetime:
    return FIXED_TIME


def _validators() -> dict[str, Any]:
    documents = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(SCHEMA_DIR.glob("*.schema.json"))
    }
    raw_registry: Any = Registry()  # type: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    registry: Any = raw_registry.with_resources(
        [(doc["$id"], Resource.from_contents(doc)) for doc in documents.values()]
    )
    return {
        name: Draft202012Validator(doc, registry=registry, format_checker=FormatChecker())
        for name, doc in documents.items()
    }


def _assert_valid_archive(
    validators: dict[str, Any], path: Path, *, trace_id: str
) -> dict[str, bytes]:
    entries = read_archive_entries(path)
    assert set(entries) == {
        "manifest.json",
        "public/workflow.graph.json",
        "public/events.ndjson",
        "public/summary.json",
        "integrity.json",
    }
    manifest = json.loads(entries["manifest.json"])
    validators["manifest.schema.json"].validate(manifest)
    assert manifest["trace_id"] == trace_id
    assert manifest["capture"]["profile"] == "audit"
    assert manifest["capture"]["vault_present"] is False
    graph = json.loads(entries["public/workflow.graph.json"])
    validators["workflow-graph.schema.json"].validate(graph)
    integrity = json.loads(entries["integrity.json"])
    validators["integrity.schema.json"].validate(integrity)
    summary = json.loads(entries["public/summary.json"])
    validators["summary.schema.json"].validate(summary)
    event_validator = validators["public-event.schema.json"]
    lines = [line for line in entries["public/events.ndjson"].split(b"\n") if line.strip()]
    assert lines, "ledger must not be empty"
    for line in lines:
        event = json.loads(line)
        event_validator.validate(event)
        assert event["run_id"] == trace_id
        assert event["redacted_fields"] == sorted(set(event["redacted_fields"]))
    report = verify_archive(path)
    assert report.ok, report.errors
    assert not report.errors
    return entries


def _writer_tool(calls: list[tuple[str, dict[str, Any]]]) -> Any:
    @tool(capability="fs.write", description="w", returns={"note": str, "summary": str})
    async def writer(*, path: Path, content: str, ctx: ToolContext) -> dict[str, str]:
        calls.append(
            ("fs.write", {"path": str(path), "content": content, "stage": ctx.stage_id})
        )
        return {"note": f"wrote {content}", "summary": f"wrote {content}"}

    return writer


def _trace_manifest() -> WorkflowManifest:
    return WorkflowManifest(
        workflow_id="TraceTest",
        entry_stage_id="Start",
        exit_stage_ids=frozenset({"Done"}),
        inputs=(InputSpec(name="p", type="string"),),
        capabilities=frozenset({"fs.write"}),
        policies=(),
        stages=(
            StageSpec(
                id="Start",
                prompt="start",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="p"), optional=False),),
                writes=(WriteSpec(name="note", type="string", optional=False),),
                requires=frozenset({"fs.write"}),
                transitions=(
                    TransitionSpec(
                        to="Done",
                        priority=0,
                        reason="explicit_transition",
                        guard=GuardSpec(kind="always"),
                    ),
                ),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.write",
                    args={
                        "path": ExprSpec(kind="ref", ref=RefSpec(kind="input", name="p")),
                        "content": ExprSpec(kind="literal", type="string", value="hello"),
                    },
                ),
            ),
            StageSpec(
                id="Done",
                prompt="done",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="p"), optional=False),),
                writes=(WriteSpec(name="summary", type="string", optional=False),),
                requires=frozenset({"fs.write"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.write",
                    args={
                        "path": ExprSpec(kind="ref", ref=RefSpec(kind="input", name="p")),
                        "content": ExprSpec(kind="literal", type="string", value="bye"),
                    },
                ),
            ),
        ),
    )


def _make_recorder(tmp_path: Path, name: str = "run.nemotrace", **kwargs: Any) -> TraceRecorder:
    params: dict[str, Any] = {
        "profile": "audit",
        "provenance": HostProvenance(
            frontend="nemo_dsl",
            target="python",
            compiler_version="0.1.9",
            ir_version="0.1",
            ir_sha256="sha256:" + "ab" * 32,
        ),
        "path_aliases": {"$workspace": tmp_path},
        "safe_path_aliases": frozenset({"$workspace"}),
        "trace_id": FIXED_TRACE_ID,
        "clock": _fixed_clock,
    }
    params.update(kwargs)
    return TraceRecorder.create(tmp_path / name, **params)


async def test_audit_archive_end_to_end(tmp_path: Path) -> None:
    validators = _validators()
    calls: list[tuple[str, dict[str, Any]]] = []
    manifest = _trace_manifest()
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([_writer_tool(calls)]),
        stage_executor=_failing_executor(),
    )
    recorder = _make_recorder(tmp_path)
    target = tmp_path / "f.txt"
    result = await runtime.run({"p": str(target)}, trace_recorder=recorder)
    assert result.output["summary"] == "wrote bye"
    entries = _assert_valid_archive(validators, tmp_path / "run.nemotrace", trace_id=FIXED_TRACE_ID)
    kinds = [
        json.loads(line)["kind"]
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip()
    ]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_completed"
    assert "stage_started" in kinds
    assert "transition_selected" in kinds
    # Tool args redacted: content marker, aliased path.
    started = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip() and json.loads(line)["kind"] == "tool_call_started"
    ]
    assert len(started) == 2
    for event in started:
        assert event["args"]["content"]["$redacted"]["reason"] == "private_content"
        assert event["args"]["path"] == "$workspace/f.txt"
        assert "/args/content" in event["redacted_fields"]
    # Stage outputs redacted (strings are never metrics).
    completed = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip() and json.loads(line)["kind"] == "stage_completed"
    ]
    assert len(completed) == 2
    assert completed[0]["output"]["note"]["$redacted"]["reason"] == "private_content"
    # Manifest binds the run.
    manifest_obj = json.loads(entries["manifest.json"])
    assert manifest_obj["status"] == "complete"
    assert manifest_obj["provenance"]["complete"] is True
    assert manifest_obj["workflow"]["id"] == "TraceTest"


def _failing_executor() -> Any:
    class NeverModel(StageExecutor):
        async def execute(self, ctx: Any) -> Any:
            msg = f"no model stages in this fixture (stage {ctx.stage.id})"
            raise AssertionError(msg)

    return NeverModel()


async def test_run_and_stream_produce_identical_ledgers(tmp_path: Path) -> None:
    first = tmp_path / "first.nemotrace"
    second = tmp_path / "second.nemotrace"
    target = tmp_path / "f.txt"
    calls: list[tuple[str, dict[str, Any]]] = []
    runtime = WorkflowRuntime(
        manifest=_trace_manifest(),
        tools=ToolRegistry([_writer_tool(calls)]),
        stage_executor=_failing_executor(),
    )
    await runtime.run(
        {"p": str(target)}, trace_recorder=_make_recorder(tmp_path, "first.nemotrace")
    )
    async for _ in runtime.stream(
        {"p": str(target)}, trace_recorder=_make_recorder(tmp_path, "second.nemotrace")
    ):
        pass
    first_entries = read_archive_entries(first)
    second_entries = read_archive_entries(second)
    assert first_entries == second_entries
    assert verify_archive(first).content_identity == verify_archive(second).content_identity


async def test_live_events_unchanged_by_recorder(tmp_path: Path) -> None:
    async def collect(*, use_trace: bool) -> list[tuple[Any, ...]]:
        calls: list[tuple[str, dict[str, Any]]] = []
        runtime = WorkflowRuntime(
            manifest=_trace_manifest(),
            tools=ToolRegistry([_writer_tool(calls)]),
            stage_executor=_failing_executor(),
        )
        seen: list[WorkflowEvent] = []

        async def sink(event: WorkflowEvent) -> None:
            seen.append(event)

        recorder = _make_recorder(tmp_path) if use_trace else None
        await runtime.run(
            {"p": str(tmp_path / "f.txt")},
            event_sink=sink,
            trace_recorder=recorder,
        )
        return [(e.kind, e.sequence, e.stage_id, e.capability) for e in seen]

    plain = await collect(use_trace=False)
    traced = await collect(use_trace=True)
    assert traced == plain


async def test_seeded_secrets_absent_from_cleartext(tmp_path: Path) -> None:
    seeds = (VECTORS / "redaction" / "seeded-values.txt").read_text(encoding="utf-8").splitlines()
    assert seeds
    unsafe = json.loads((VECTORS / "redaction" / "unsafe-input.json").read_text(encoding="utf-8"))
    api_key = unsafe["run_inputs"]["api_key_echo"]
    calls: list[tuple[str, dict[str, Any]]] = []

    @tool(capability="fs.write", description="w", returns={"note": str})
    async def writer(*, path: Path, content: str, ctx: ToolContext) -> dict[str, str]:
        calls.append(
            ("fs.write", {"path": str(path), "content": content, "stage": ctx.stage_id})
        )
        return {"note": "ok"}

    manifest = WorkflowManifest(
        workflow_id="LeakTest",
        entry_stage_id="Only",
        exit_stage_ids=frozenset({"Only"}),
        inputs=(InputSpec(name="p", type="string"),),
        capabilities=frozenset({"fs.write"}),
        policies=(),
        stages=(
            StageSpec(
                id="Only",
                prompt="",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="p"), optional=False),),
                writes=(WriteSpec(name="note", type="string", optional=False),),
                requires=frozenset({"fs.write"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.write",
                    args={
                        "path": ExprSpec(kind="ref", ref=RefSpec(kind="input", name="p")),
                        "content": ExprSpec(kind="literal", type="string", value=api_key),
                    },
                ),
            ),
        ),
    )
    runtime = WorkflowRuntime(
        manifest=manifest, tools=ToolRegistry([writer]), stage_executor=_failing_executor()
    )
    recorder = TraceRecorder.create(
        tmp_path / "run.nemotrace",
        profile="audit",
        provenance=HostProvenance(frontend="nemo_dsl", target="python", compiler_version="0.1.9"),
        path_aliases={"$workspace": tmp_path},
        secrets=tuple(seeds),
        trace_id=FIXED_TRACE_ID,
        clock=_fixed_clock,
    )
    await runtime.run({"p": str(tmp_path / "secret.txt")}, trace_recorder=recorder)
    entries = read_archive_entries(tmp_path / "run.nemotrace")
    blob = b"\n".join(
        name.encode() + b"\n" + data for name, data in sorted(entries.items())
    )
    for seed in seeds:
        assert seed.encode() not in blob, seed
    # Alias not marked safe: paths degrade to opaque refs, never segments.
    started = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip() and json.loads(line)["kind"] == "tool_call_started"
    ]
    assert started[0]["args"]["path_ref"] == "path-1"
    assert "path" not in started[0]["args"]


async def test_tool_failure_finalizes_failed_without_raw_message(tmp_path: Path) -> None:
    @tool(capability="fs.read", description="r", returns={"content": str})
    async def reader(*, path: Path, ctx: ToolContext) -> str:
        msg = f"boom under /home/secret-user: {path}"
        raise ToolInvocationError(msg)

    manifest = WorkflowManifest(
        workflow_id="FailTest",
        entry_stage_id="Only",
        exit_stage_ids=frozenset({"Only"}),
        inputs=(InputSpec(name="p", type="string"),),
        capabilities=frozenset({"fs.read"}),
        policies=(),
        stages=(
            StageSpec(
                id="Only",
                prompt="",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="p"), optional=False),),
                writes=(WriteSpec(name="content", type="string", optional=False),),
                requires=frozenset({"fs.read"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.read",
                    args={"path": ExprSpec(kind="ref", ref=RefSpec(kind="input", name="p"))},
                ),
            ),
        ),
    )
    runtime = WorkflowRuntime(
        manifest=manifest, tools=ToolRegistry([reader]), stage_executor=_failing_executor()
    )
    recorder = _make_recorder(tmp_path)
    with pytest.raises(ToolInvocationError):
        await runtime.run({"p": str(tmp_path / "f.txt")}, trace_recorder=recorder)
    entries = read_archive_entries(tmp_path / "run.nemotrace")
    manifest_obj = json.loads(entries["manifest.json"])
    assert manifest_obj["status"] == "failed"
    failed = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip() and json.loads(line)["kind"] == "tool_call_failed"
    ]
    assert len(failed) == 1
    assert failed[0]["error"] == "tool_failed"
    assert failed[0]["metadata"]["error_type"] == "ToolInvocationError"
    blob = b"\n".join(entries.values())
    assert b"boom under" not in blob
    assert b"/home/secret-user" not in blob


async def test_policy_denial_uses_opaque_ref(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    manifest = WorkflowManifest(
        workflow_id="PolicyTest",
        entry_stage_id="Only",
        exit_stage_ids=frozenset({"Only"}),
        inputs=(InputSpec(name="p", type="string"),),
        capabilities=frozenset({"fs.write"}),
        policies=(
            PolicySpec(
                id='deny fs.write(path) if command.eq("PRIVATE-COMMAND")',
                kind="deny",
                trigger=TriggerSpec(capability="fs.write", bind={"path": "path"}),
                requires=(),
                condition=ExprSpec(kind="literal", type="bool", value=True),
            ),
        ),
        stages=(
            StageSpec(
                id="Only",
                prompt="",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="p"), optional=False),),
                writes=(WriteSpec(name="note", type="string", optional=False),),
                requires=frozenset({"fs.write"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.write",
                    args={
                        "path": ExprSpec(kind="ref", ref=RefSpec(kind="input", name="p")),
                        "content": ExprSpec(kind="literal", type="string", value="x"),
                    },
                ),
            ),
        ),
    )
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=ToolRegistry([_writer_tool(calls)]),
        stage_executor=_failing_executor(),
    )
    recorder = _make_recorder(tmp_path)
    with pytest.raises(PolicyDeniedError):
        await runtime.run({"p": str(tmp_path / "f.txt")}, trace_recorder=recorder)
    entries = read_archive_entries(tmp_path / "run.nemotrace")
    denied = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip() and json.loads(line)["kind"] == "policy_denied"
    ]
    assert len(denied) == 1
    assert denied[0]["metadata"]["policy_ref"] == "p-1"
    blob = b"\n".join(entries.values())
    assert b"PRIVATE-COMMAND" not in blob


async def test_model_stage_records_response_bytes_and_metrics(tmp_path: Path) -> None:
    class Adapter:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        async def complete(self, request: Any) -> ModelResponse:
            self.calls.append(request)
            return ModelResponse(content='{"score": 0.9}')

    manifest = WorkflowManifest(
        workflow_id="ModelTest",
        entry_stage_id="Judge",
        exit_stage_ids=frozenset({"Judge"}),
        inputs=(),
        capabilities=frozenset(),
        policies=(),
        stages=(
            StageSpec(
                id="Judge",
                prompt="judge",
                reads=(),
                writes=(WriteSpec(name="score", type="number", optional=False),),
                requires=frozenset(),
                transitions=(),
            ),
        ),
    )
    tools = ToolRegistry([])
    runtime = WorkflowRuntime(
        manifest=manifest,
        tools=tools,
        stage_executor=ModelStageExecutor(model=Adapter(), tools=tools),  # type: ignore[arg-type]
    )
    recorder = _make_recorder(tmp_path, approved_metrics=frozenset({"Judge.score"}))
    result = await runtime.run({}, trace_recorder=recorder)
    assert result.output["score"] == 0.9
    entries = read_archive_entries(tmp_path / "run.nemotrace")
    events = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip()
    ]
    completed = [e for e in events if e["kind"] == "model_completed"]
    assert len(completed) == 1
    assert completed[0]["metadata"]["response_bytes"] > 0
    stage_done = next(e for e in events if e["kind"] == "stage_completed")
    assert stage_done["output"]["score"] == 0.9


async def test_stream_cancel_finalizes_interrupted(tmp_path: Path) -> None:
    @tool(capability="fs.read", description="r", returns={"content": str})
    async def blocker(*, path: Path, ctx: ToolContext) -> str:
        await asyncio.sleep(30)
        return "never"

    manifest = WorkflowManifest(
        workflow_id="CancelTest",
        entry_stage_id="Only",
        exit_stage_ids=frozenset({"Only"}),
        inputs=(InputSpec(name="p", type="string"),),
        capabilities=frozenset({"fs.read"}),
        policies=(),
        stages=(
            StageSpec(
                id="Only",
                prompt="",
                reads=(ReadSpec(ref=RefSpec(kind="input", name="p"), optional=False),),
                writes=(WriteSpec(name="content", type="string", optional=False),),
                requires=frozenset({"fs.read"}),
                transitions=(),
                execution=StageExecutionSpec(
                    kind="tool",
                    capability="fs.read",
                    args={"path": ExprSpec(kind="ref", ref=RefSpec(kind="input", name="p"))},
                ),
            ),
        ),
    )
    runtime = WorkflowRuntime(
        manifest=manifest, tools=ToolRegistry([blocker]), stage_executor=_failing_executor()
    )
    recorder = _make_recorder(tmp_path)

    async def consume() -> None:
        async for event in runtime.stream({"p": str(tmp_path / "f.txt")}, trace_recorder=recorder):
            if event.kind == "stage_started":
                break

    await asyncio.wait_for(consume(), timeout=10)
    # Finalization is synchronous inside cancellation handling, but poll
    # briefly to stay robust against scheduling skew.
    for _ in range(100):
        trace_path = tmp_path / "run.nemotrace"
        partial_path = tmp_path / "run.nemotrace.partial"
        if trace_path.exists() and not partial_path.exists():
            break
        await asyncio.sleep(0.05)
    entries = read_archive_entries(tmp_path / "run.nemotrace")
    assert json.loads(entries["manifest.json"])["status"] == "interrupted"


def test_resolve_trace_recorder(tmp_path: Path) -> None:
    generated = HostProvenance(
        frontend="nemo_dsl",
        target="python",
        compiler_version="0.1.9",
        ir_sha256="sha256:" + "cd" * 32,
    )
    model = ModelDescriptor(name="fake", temperature=0.2)
    assert resolve_trace_recorder(None) is None
    with pytest.raises(TraceError, match="unsupported trace value"):
        resolve_trace_recorder("audit")  # type: ignore[arg-type]
    with pytest.raises(TraceError, match="factory must return"):
        resolve_trace_recorder(lambda: "nope")  # type: ignore[return-value]

    bare = TraceConfig(path=tmp_path / "a.nemotrace")
    recorder = resolve_trace_recorder(
        bare, default_provenance=generated, default_model=model
    )
    assert recorder is not None
    assert recorder.config.provenance == generated
    assert recorder.config.model == model

    explicit = HostProvenance(frontend="x", target="manual", compiler_version="y")
    manual = TraceConfig(path=tmp_path / "b.nemotrace", provenance=explicit)
    recorder = resolve_trace_recorder(
        manual, default_provenance=generated, default_model=model
    )
    assert recorder is not None
    # Provenance is atomic: without a host fingerprint the generated
    # descriptor (whose hash was verified against its own resources) wins
    # wholesale rather than merging into a false binding.
    assert recorder.config.provenance == generated
    assert recorder.config.model == model

    made: list[TraceRecorder] = []

    def factory() -> TraceRecorder | None:
        rec = TraceRecorder(TraceConfig(path=tmp_path / f"{len(made)}.nemotrace"))
        made.append(rec)
        return rec

    first = resolve_trace_recorder(factory)
    second = resolve_trace_recorder(factory)
    assert first is not None
    assert second is not None
    assert first is not second


def test_profiles_and_annotations_refused(tmp_path: Path) -> None:
    with pytest.raises(TraceError, match="only 'audit'"):
        TraceRecorder.create(tmp_path / "x.nemotrace", profile="replay")
    with pytest.raises(TraceError, match="only 'audit'"):
        TraceRecorder.create(tmp_path / "x.nemotrace", profile="publication")


def _trial_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trial_id": 1,
        "candidate_ref": "candidate-1",
        "verdict": "rejected",
        "reason_code": "no_improvement",
        "selection_metrics": {"candidate_median_ns": 100.0, "valid": True},
        "artifact_refs": [],
    }
    payload.update(overrides)
    return payload


def test_annotation_accepted_and_shaped(tmp_path: Path) -> None:
    validators = _validators()
    recorder = _make_recorder(tmp_path)
    recorder.begin_run(_trace_manifest())
    visit = recorder.begin_stage_visit("RecordTrial")
    record = recorder.record_annotation(
        "nemoir.autoresearch/v1", "trial_finished", _trial_payload(), anchor_sequence=7
    )
    assert record is not None
    assert record["kind"] == "annotation"
    assert "sequence" not in record
    assert record["stage_id"] == "RecordTrial"
    assert record["stage_visit_id"] == visit
    assert record["anchor_sequence"] == 7
    assert record["run_id"] == FIXED_TRACE_ID
    validators["public-event.schema.json"].validate(record)
    validators["autoresearch-annotation.schema.json"].validate(
        record["annotation"]["payload"]
    )


def test_verifier_rejects_malformed_known_annotation(tmp_path: Path) -> None:
    """H4: forged archive with invalid trial_id must not verify as ok."""
    import zipfile  # noqa: PLC0415

    from nemoir_runtime.canonical import sha256_tag, to_canonical_bytes  # noqa: PLC0415

    recorder = _make_recorder(tmp_path, name="good.nemotrace")
    recorder.begin_run(_trace_manifest())
    recorder.begin_stage_visit("RecordTrial")
    evt = WorkflowEvent(
        kind="stage_completed", run_id="x", sequence=1, timestamp=FIXED_TIME,
        stage_id="RecordTrial", output={"report": "x"},
    )
    recorder.observe_workflow_event(evt)
    recorder.record_annotation(
        "nemoir.autoresearch/v1", "trial_finished", _trial_payload(), anchor_sequence=1,
    )
    good_path = recorder.finish_run("complete")
    entries = read_archive_entries(good_path)
    # Forge: trial_id 0 violates schema; recompute hashes + identity deterministically.
    lines = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip()
    ]
    for line in lines:
        if line["kind"] == "annotation":
            line["annotation"]["payload"]["trial_id"] = 0
            break
    else:
        msg = "no annotation to forge"
        raise AssertionError(msg)
    forged_events = b"".join(to_canonical_bytes(e) + b"\n" for e in lines)
    forged_entries = dict(entries)
    forged_entries["public/events.ndjson"] = forged_events
    # Recompute summary + integrity for a self-consistent forgery.
    summary = json.loads(forged_entries["public/summary.json"])
    summary["events_sha256"] = sha256_tag(forged_events)
    kinds: dict[str, int] = {}
    for event in lines:
        kinds[event["kind"]] = kinds.get(event["kind"], 0) + 1
    summary["counts_by_kind"] = kinds
    forged_entries["public/summary.json"] = to_canonical_bytes(summary)
    integrity_entries = [
        {
            "path": path,
            "media_type": (
                "application/x-ndjson"
                if path.endswith(".ndjson")
                else "application/json"
            ),
            "uncompressed_bytes": len(data),
            "sha256": sha256_tag(data),
        }
        for path, data in sorted(forged_entries.items())
        if path != "integrity.json"
    ]
    identity = {
        "format": "nemoir.trace.content-identity/0.1",
        "entries": sorted(
            (
                {
                    "path": e["path"],
                    "sha256": e["sha256"],
                    "uncompressed_bytes": e["uncompressed_bytes"],
                }
                for e in integrity_entries
            ),
            key=lambda e: e["path"],
        ),
    }
    forged_entries["integrity.json"] = to_canonical_bytes({
        "format": "nemoir.trace.integrity/0.1", "algorithm": "sha256",
        "entries": integrity_entries, "content_identity": sha256_tag(to_canonical_bytes(identity)),
    })
    forged_path = tmp_path / "forged.nemotrace"
    with zipfile.ZipFile(forged_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(forged_entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, forged_entries[name])
    report = verify_archive(forged_path)
    assert not report.ok
    assert any("trial_finished" in e for e in report.errors)


def test_hook_anchor_forced_and_malformed_counted(tmp_path: Path) -> None:
    """M1: hook anchor ignored (triggering seq wins); malformed hook counted."""

    def bad_hook(_info: Any) -> Any:
        return {
            "namespace": "nemoir.autoresearch/v1",
            "kind": "trial_finished",
            "payload": _trial_payload(reason_code="bogus"),
            "anchor_sequence": 999,  # must be ignored; payload is malformed anyway
        }

    recorder = _make_recorder(tmp_path, name="hook.nemotrace", on_stage_completed=bad_hook)
    recorder.begin_run(_trace_manifest())
    recorder.begin_stage_visit("RecordTrial")
    evt = WorkflowEvent(
        kind="stage_completed", run_id="x", sequence=5, timestamp=FIXED_TIME,
        stage_id="RecordTrial", output={"report": "x"},
    )
    recorder.observe_workflow_event(evt)
    assert recorder.annotations_dropped == 1
    assert recorder.annotation_warnings
    # Warnings are surfaced via summary.json (review item 1 completeness).
    hook_path = recorder.finish_run("complete")
    hook_entries = read_archive_entries(hook_path)
    hook_summary = json.loads(hook_entries["public/summary.json"])
    assert hook_summary["annotations_dropped"] == 1
    assert hook_summary["annotation_warnings"] == list(recorder.annotation_warnings)
    assert len(hook_summary["annotation_warnings"]) == 1
    # Direct cross-visit anchor raises loudly.
    recorder2 = _make_recorder(tmp_path, name="direct.nemotrace")
    recorder2.begin_run(_trace_manifest())
    recorder2.begin_stage_visit("RecordTrial")
    evt2 = WorkflowEvent(
        kind="stage_completed", run_id="x", sequence=3, timestamp=FIXED_TIME,
        stage_id="RecordTrial", output={"report": "x"},
    )
    recorder2.observe_workflow_event(evt2)
    with pytest.raises(TraceError, match="does not belong to visit"):
        recorder2.record_annotation(
            "nemoir.autoresearch/v1", "trial_finished", _trial_payload(), anchor_sequence=999,
        )


def test_model_retry_without_pending_call_keeps_valid_id(tmp_path: Path) -> None:
    """Regression: a tool-error retry after model_completed consumed the call
    id must still carry a valid model_call_id (schema requires it)."""

    recorder = _make_recorder(tmp_path)
    recorder.begin_run(_trace_manifest())
    recorder.begin_stage_visit("Start")
    seq = 0

    def emit(kind: str, **kwargs: Any) -> Any:
        nonlocal seq
        seq += 1
        return recorder.observe_workflow_event(
            WorkflowEvent(
                kind=kind,  # type: ignore[arg-type]
                run_id="x",
                sequence=seq,
                timestamp=FIXED_TIME,
                stage_id="Start",
                **kwargs,  # type: ignore[arg-type]
            )
        )

    recorder.begin_model_call("Start")
    emit("model_completed")
    retry = emit(
        "model_retry",
        error="x",
        metadata={"attempt": 1, "max_retries": 3, "category": "tool_call"},
    )
    assert retry is not None
    assert re.fullmatch(r"m-[1-9][0-9]*", retry["model_call_id"])
    path = recorder.finish_run("complete")
    report = verify_archive(path)
    assert report.ok, report.errors
    assert not report.errors


def test_annotation_unknown_namespace_markered(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path)
    recorder.begin_run(_trace_manifest())
    recorder.begin_stage_visit("RecordTrial")
    record = recorder.record_annotation("example.com/v1", "custom", {"x": 1})
    assert record is not None
    assert record["annotation"]["payload"] == {
        "$redacted": record["annotation"]["payload"]["$redacted"]
    }
    assert "/annotation/payload" in record["redacted_fields"]


def test_annotation_malformed_rejected(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path)
    recorder.begin_run(_trace_manifest())
    recorder.begin_stage_visit("RecordTrial")
    with pytest.raises(TraceError, match="trial_finished"):
        recorder.record_annotation("nemoir.autoresearch/v1", "trial_finished", {})
    with pytest.raises(TraceError, match="reason_code"):
        recorder.record_annotation(
            "nemoir.autoresearch/v1", "trial_finished", _trial_payload(reason_code="bogus")
        )
    with pytest.raises(TraceError, match="unknown fields"):
        recorder.record_annotation(
            "nemoir.autoresearch/v1",
            "trial_finished",
            _trial_payload(detail="free prose leaks"),
        )
    with pytest.raises(TraceError, match="anchor_sequence"):
        recorder.record_annotation(
            "nemoir.autoresearch/v1", "trial_finished", _trial_payload(), anchor_sequence=0
        )
    fresh = _make_recorder(tmp_path, name="b.nemotrace")
    fresh.begin_run(_trace_manifest())
    with pytest.raises(TraceError, match="enclosing stage visit"):
        fresh.record_annotation(
            "nemoir.autoresearch/v1", "trial_finished", _trial_payload()
        )


def test_annotation_secret_scan_omits_or_masks(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path, secrets=("sk-cvxpygen-TEST-secret-0001",))
    recorder.begin_run(_trace_manifest())
    recorder.begin_stage_visit("RecordTrial")
    # Audit profile rejects raw mechanism_id/digests outright (redaction-policy
    # §11: they require explicit publication review). Opaque refs only.
    with pytest.raises(TraceError, match="mechanism_id"):
        recorder.record_annotation(
            "nemoir.autoresearch/v1",
            "trial_finished",
            _trial_payload(mechanism_id="sk-cvxpygen-TEST-secret-0001"),
        )
    with pytest.raises(TraceError, match="candidate_digest"):
        recorder.record_annotation(
            "nemoir.autoresearch/v1",
            "trial_finished",
            _trial_payload(candidate_digest="sha256:" + "ab" * 32),
        )
    # Opaque-only payload still records and stays secret-free.
    record = recorder.record_annotation(
        "nemoir.autoresearch/v1",
        "trial_finished",
        _trial_payload(),
    )
    assert record is not None
    blob = json.dumps(record).encode()
    assert b"sk-cvxpygen-TEST-secret-0001" not in blob


async def test_blocked_finalization_is_loud_without_double_terminal(tmp_path: Path) -> None:
    """A scanner hit on a successful run raises and never emits run_failed."""
    calls: list[tuple[str, dict[str, Any]]] = []
    runtime = WorkflowRuntime(
        manifest=_trace_manifest(),
        tools=ToolRegistry([_writer_tool(calls)]),
        stage_executor=_failing_executor(),
    )
    seen: list[Any] = []

    async def sink(event: Any) -> None:
        seen.append(event)

    recorder = _make_recorder(
        tmp_path, model=ModelDescriptor(name="sk-blocked-0123456789abcdef")
    )
    with pytest.raises(TraceError, match="blocked finalization"):
        await runtime.run(
            {"p": str(tmp_path / "f.txt")}, event_sink=sink, trace_recorder=recorder
        )
    kinds = [event.kind for event in seen]
    assert "run_completed" in kinds
    assert "run_failed" not in kinds
    assert not (tmp_path / "run.nemotrace").exists()
    assert not (tmp_path / "run.nemotrace.partial").exists()


async def test_max_steps_finalizes_failed_with_taxonomy(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    runtime = WorkflowRuntime(
        manifest=_trace_manifest(),
        tools=ToolRegistry([_writer_tool(calls)]),
        stage_executor=_failing_executor(),
    )
    recorder = _make_recorder(tmp_path)
    with pytest.raises(MaxStepsExceededError):
        await runtime.run(
            {"p": str(tmp_path / "f.txt")},
            options=RunOptions(max_steps=1),
            trace_recorder=recorder,
        )
    entries = read_archive_entries(tmp_path / "run.nemotrace")
    assert json.loads(entries["manifest.json"])["status"] == "failed"
    failed = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip() and json.loads(line)["kind"] == "run_failed"
    ]
    assert len(failed) == 1
    assert failed[0]["error"] == "run_failed"
    assert failed[0]["metadata"]["error_type"] == "MaxStepsExceededError"


def test_double_begin_and_finish_raise(tmp_path: Path) -> None:
    recorder = _make_recorder(tmp_path)
    recorder.begin_run(_trace_manifest())
    with pytest.raises(TraceError, match="twice"):
        recorder.begin_run(_trace_manifest())
    with pytest.raises(TraceError, match="unknown trace status"):
        recorder.finish_run("bogus")
