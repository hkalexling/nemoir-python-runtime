"""Phase 4 replay vault: capture, encryption, unlock, and tamper resistance.

``docs/trace/schema/test-vectors/vault/vault-fake-run.json`` drives the
recorder through a fixed op sequence (fixed clock + trace id + passphrase).
The decrypted vault plaintext must match the frozen
``expected-vault-records.ndjson`` byte-for-byte; the web recorder asserts
the same frozen file (cross-language vault parity).
"""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

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
    TraceError,
    TraceRecorder,
    VaultCapture,
    read_archive_entries,
    to_canonical_bytes,
    unlock_archive,
    verify_archive,
)

ROOT = Path(__file__).resolve().parents[3]
VAULT_VECTORS = ROOT / "docs" / "trace" / "schema" / "test-vectors" / "vault"
FAKE_PASSPHRASE = "phase4-vault-fake-passphrase-01"  # noqa: S105


def _load_fixture() -> dict[str, Any]:
    return json.loads((VAULT_VECTORS / "vault-fake-run.json").read_text(encoding="utf-8"))  # type: ignore[reportUnknownVariableType]


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


def _drive_fixture(
    tmp_path: Path,
    *,
    profile: str = "replay",
    passphrase: str | None = FAKE_PASSPHRASE,
    capture: VaultCapture | None = None,
) -> Path:
    fixture = _load_fixture()
    config = fixture["config"]
    clock_at = datetime.fromisoformat(config["clock"])
    recorder = TraceRecorder.create(
        tmp_path / "vault.nemotrace",
        profile=profile,
        provenance=HostProvenance(
            frontend="vault-fixture",
            target="python",
            compiler_version="vault-fixture",
            ir_version="0.1",
            ir_sha256="sha256:" + "ab" * 32,
        ),
        path_aliases={key: Path(value) for key, value in config["path_aliases"].items()},
        safe_path_aliases=frozenset(config["safe_path_aliases"]),
        approved_metrics=frozenset(config["approved_metrics"]),
        secrets=tuple(config["secrets"]),
        trace_id=config["trace_id"],
        clock=lambda: clock_at,
        vault_passphrase=passphrase,
        vault_capture=capture or VaultCapture(),
    )
    last_visit = ""
    last_mid = ""
    last_tid = ""
    dummy_ts = datetime(2020, 1, 1, tzinfo=UTC)
    for op in fixture["ops"]:
        kind = op["op"]
        if kind == "begin_run":
            recorder.begin_run(_build_manifest(fixture["manifest"]))
        elif kind == "begin_stage_visit":
            last_visit = recorder.begin_stage_visit(op["stage_id"])
        elif kind == "record_run_inputs":
            recorder.record_run_inputs(op["inputs"])
        elif kind == "begin_model_call":
            last_mid = recorder.begin_model_call()
        elif kind == "record_model_request":
            recorder.record_model_request(last_mid, op["request"])
        elif kind == "record_model_response_full":
            recorder.record_model_response(
                last_mid,
                response_bytes=op["response_bytes"],
                tool_call_count=op["tool_call_count"],
                response=op["response"],
            )
        elif kind == "begin_tool_call":
            last_tid = recorder.begin_tool_call()
        elif kind == "record_tool_result":
            recorder.record_tool_result(last_tid, op["result"], args=op.get("args"))
        elif kind == "record_transition_evaluation":
            recorder.record_transition_evaluation(last_visit, op["candidates"])
        elif kind == "record_policy_evaluation":
            recorder.record_policy_evaluation(
                last_visit, op["policy_id"], op["bound"], op["outcome"]
            )
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
            msg = f"unknown vault op {kind!r}"
            raise AssertionError(msg)
    return recorder.finish_run("complete")


def _decrypted_records(path: Path) -> list[dict[str, Any]]:
    records, report = unlock_archive(path, FAKE_PASSPHRASE)
    assert report.semantic == "passed", report.errors
    return records


def test_replay_fixture_matches_frozen_vault(tmp_path: Path) -> None:
    path = _drive_fixture(tmp_path)
    report = verify_archive(path)
    assert report.ok, report.errors
    assert report.replayability == "taped-replay"
    assert report.integrity == "passed"
    assert report.structural == "passed", report.warnings
    assert report.semantic == "not-evaluated"
    entries = read_archive_entries(path)
    assert set(entries) == {
        "manifest.json",
        "public/workflow.graph.json",
        "public/events.ndjson",
        "public/summary.json",
        "private/vault.enc",
        "private/vault.meta.json",
        "integrity.json",
    }
    # Ciphertext entry is STORE; everything else is DEFLATE.
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.filename == "private/vault.enc":
                assert info.compress_type == zipfile.ZIP_STORED
            else:
                assert info.compress_type == zipfile.ZIP_DEFLATED
    manifest = json.loads(entries["manifest.json"])
    assert manifest["capture"]["profile"] == "replay"
    assert manifest["capture"]["vault_present"] is True
    assert manifest["capture"]["publication_eligible"] is False
    records = _decrypted_records(path)
    frozen = (VAULT_VECTORS / "expected-vault-records.ndjson").read_bytes()
    actual = b"".join(to_canonical_bytes(r) + b"\n" for r in records)
    assert actual == frozen


def test_vault_record_types_cover_replay_evidence(tmp_path: Path) -> None:
    records = _decrypted_records(_drive_fixture(tmp_path))
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(record["record_type"], []).append(record)
    assert set(by_type) >= {
        "run_inputs",
        "stage_snapshot",
        "model_request",
        "model_response",
        "tool_result",
        "transition_evaluation",
        "policy_evaluation",
        "full_workflow_ir",
    }
    assert [r["record_id"] for r in records] == [f"v-{n}" for n in range(1, len(records) + 1)]
    transitions = {r["stage_visit_id"]: r for r in by_type["transition_evaluation"]}
    assert transitions["s-1"]["event_sequence"] == 6
    assert transitions["s-1"]["payload"]["candidates"][0]["matched"] is True
    assert transitions["s-2"]["event_sequence"] == 10


def test_hostile_values_never_survive(tmp_path: Path) -> None:
    path = _drive_fixture(tmp_path)
    entries = read_archive_entries(path)
    cleartext = b"\n".join(
        name.encode() + b"\n" + data
        for name, data in entries.items()
        if name != "private/vault.enc"
    )
    for forbidden in (
        b"sk-vault-TEST-secret-9999",
        b"SHOULD-NEVER-APPEAR",
        b"/home/vault-user",
        b"private-chain-of-thought",
        b"Bearer ",
    ):
        assert forbidden not in cleartext, forbidden
    records = _decrypted_records(path)
    vault_text = "\n".join(json.dumps(r, sort_keys=True) for r in records).encode()
    # Credentials never enter the vault either; echoes become markers.
    for forbidden in (
        b"sk-vault-TEST-secret-9999",
        b"SHOULD-NEVER-APPEAR",
        b"/home/vault-user",
        b"Bearer ",
    ):
        assert forbidden not in vault_text, forbidden
    # Reasoning is excluded by default (opt-in capture only).
    assert b"private-chain-of-thought" not in vault_text
    # ...but structure survives: paths aliased, public values intact.
    assert b"$workspace/a.txt" in vault_text
    assert b"synthetic-vault" in vault_text


def test_reasoning_opt_in(tmp_path: Path) -> None:
    path = _drive_fixture(tmp_path, capture=VaultCapture(include_reasoning=True))
    records, report = unlock_archive(path, FAKE_PASSPHRASE)
    assert report.semantic == "passed", report.errors
    responses = [r for r in records if r["record_type"] == "model_response"]
    assert responses
    assert responses[0]["payload"]["reasoning"] == "private-chain-of-thought"


def test_wrong_passphrase_fails_closed(tmp_path: Path) -> None:
    path = _drive_fixture(tmp_path)
    records, report = unlock_archive(path, "wrong-passphrase")
    assert records == []
    assert report.semantic == "failed"
    # Generic failure only: must not distinguish wrong passphrase from tampering.
    assert report.errors == ("vault unlock failed",)


def test_modified_ciphertext_fails_closed(tmp_path: Path) -> None:
    path = _drive_fixture(tmp_path)
    entries = read_archive_entries(path)
    sealed = bytearray(entries["private/vault.enc"])
    sealed[0] ^= 1
    entries["private/vault.enc"] = bytes(sealed)
    tampered = tmp_path / "tampered.nemotrace"
    with zipfile.ZipFile(tampered, "w") as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = (
                zipfile.ZIP_STORED if name == "private/vault.enc" else zipfile.ZIP_DEFLATED
            )
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entries[name])
    records, report = unlock_archive(tampered, FAKE_PASSPHRASE)
    assert records == []
    assert report.semantic == "failed"


def test_replay_requires_passphrase(tmp_path: Path) -> None:
    with pytest.raises(TraceError, match="requires vault_passphrase"):
        TraceRecorder.create(tmp_path / "x.nemotrace", profile="replay")
    with pytest.raises(TraceError, match="requires trace profile 'replay'"):
        TraceRecorder.create(
            tmp_path / "x.nemotrace", profile="audit", vault_passphrase="pw"  # noqa: S106
        )
    with pytest.raises(TraceError, match="unsupported trace profile"):
        TraceRecorder.create(tmp_path / "x.nemotrace", profile="publication")


def test_audit_profile_has_no_vault(tmp_path: Path) -> None:
    fixture = _load_fixture()
    clock_at = datetime.fromisoformat(fixture["config"]["clock"])
    recorder = TraceRecorder.create(tmp_path / "audit.nemotrace", clock=lambda: clock_at)
    assert recorder.vault_enabled is False
    recorder.begin_run(_build_manifest(fixture["manifest"]))
    path = recorder.finish_run("complete")
    report = verify_archive(path)
    assert report.ok, report.errors
    assert report.replayability == "playback-only"
    records, unlock_report = unlock_archive(path, FAKE_PASSPHRASE)
    assert records == []
    assert unlock_report.semantic == "failed"


def test_retry_after_completed_reuses_attempt_id(tmp_path: Path) -> None:
    """Regression: a retry emitted after model_completed consumed the call id.

    Real runs emit ``model_completed`` then a tool-error ``model_retry``
    before the next attempt begins (see ``models.py``). The retry must reuse
    the consumed attempt id — which already has vault request/response
    evidence — instead of synthesizing a dangling id that fails semantic
    verification with ``vault missing model evidence`` (found in a real
    2-trial CVXPYgen replay trace: 17 retry-only ids, zero vault records).
    """
    clock_at = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    recorder = TraceRecorder.create(
        tmp_path / "retry.nemotrace",
        profile="replay",
        provenance=HostProvenance(
            frontend="retry-regression",
            target="python",
            compiler_version="retry-regression",
            ir_version="0.1",
            ir_sha256="sha256:" + "ab" * 32,
        ),
        trace_id="ab" * 16,
        clock=lambda: clock_at,
        vault_passphrase=FAKE_PASSPHRASE,
    )
    manifest = WorkflowManifest(
        workflow_id="MiniRetry",
        entry_stage_id="A",
        exit_stage_ids=frozenset({"A"}),
        inputs=(),
        capabilities=frozenset(),
        policies=(),
        stages=(
            StageSpec(
                id="A",
                prompt="",
                reads=(),
                writes=(WriteSpec(name="score", type="number", optional=False),),
                requires=frozenset(),
                transitions=(),
                execution=StageExecutionSpec(kind="model"),
            ),
        ),
    )
    dummy_ts = datetime(2020, 1, 1, tzinfo=UTC)

    def observe(kind: str, sequence: int, **fields: Any) -> None:
        recorder.observe_workflow_event(
            WorkflowEvent(
                kind=kind,  # type: ignore[arg-type]
                run_id="r" * 32,
                sequence=sequence,
                timestamp=dummy_ts,
                stage_id="A",
                metadata=fields.pop("metadata", {}),
                **fields,  # type: ignore[arg-type]
            )
        )

    def attempt(call_id: str, score: float) -> None:
        begun = recorder.begin_model_call()
        assert begun == call_id
        recorder.record_model_request(
            begun,
            {"messages": [{"role": "user", "content": "score"}], "tools": [], "output_schema": {}},
        )
        recorder.record_model_response(
            begun,
            response_bytes=16,
            tool_call_count=0,
            response={"content": json.dumps({"score": score}), "tool_calls": [], "usage": {}},
        )

    recorder.begin_run(manifest)
    observe("run_started", 1)
    recorder.begin_stage_visit("A")
    observe("stage_started", 2)
    attempt("m-1", 0.1)
    observe("model_completed", 3)
    # Tool-error retry arrives with an empty pending queue (real runtime
    # order: completed consumed m-1 before the retry was emitted).
    observe(
        "model_retry",
        4,
        metadata={"attempt": 1, "max_retries": 3, "category": "tool_call"},
    )
    attempt("m-2", 0.9)
    observe("model_completed", 5)
    observe("stage_completed", 6, output={"score": 0.9})
    observe("run_completed", 7)
    path = recorder.finish_run("complete")

    entries = read_archive_entries(path)
    ledger = [
        json.loads(line)
        for line in entries["public/events.ndjson"].split(b"\n")
        if line.strip()
    ]
    by_seq = {event["sequence"]: event for event in ledger}
    assert by_seq[3]["model_call_id"] == "m-1"
    assert by_seq[4]["model_call_id"] == "m-1", (
        "retry must reuse the consumed attempt id, not synthesize a dangling one"
    )
    assert by_seq[5]["model_call_id"] == "m-2"

    report = verify_archive(path)
    assert report.ok, report.errors
    records, unlocked = unlock_archive(path, FAKE_PASSPHRASE)
    assert unlocked.ok, unlocked.errors
    assert unlocked.semantic == "passed", unlocked.errors
    requested = {
        r.get("model_call_id") for r in records if r.get("record_type") == "model_request"
    }
    assert requested == {"m-1", "m-2"}
