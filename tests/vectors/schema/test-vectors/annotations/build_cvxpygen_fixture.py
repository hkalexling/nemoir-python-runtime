#!/usr/bin/env python3
"""Build a deterministic synthetic CVXPYgen annotation fixture (Phase 3).

Produces a small audit archive with three ``trial_finished`` annotations
(accepted, no_improvement, preflight_build) anchored to RecordTrial visits.
All values are synthetic and secret-free. The archive is audit-profile
(``publication_eligible: false``) for local viewer testing only -- not a
publication precedent (Phase 5 owns the publication transform).

Usage:
    python docs/trace/schema/test-vectors/annotations/build_cvxpygen_fixture.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "python" / "nemoir-runtime" / "src"))

from nemoir_runtime.runtime import (  # noqa: E402
    GuardSpec,
    InputSpec,
    StageExecutionSpec,
    StageSpec,
    TransitionSpec,
    WorkflowManifest,
    WriteSpec,
)
from nemoir_runtime.trace import HostProvenance, TraceRecorder  # noqa: E402

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
FIXED_TRACE_ID = "0123456789abcdef0123456789abcdef"

TRIALS = [
    {
        "trial_id": 1,
        "candidate_ref": "candidate-1",
        "parent_ref": None,
        "verdict": "rejected",
        "reason_code": "preflight_build",
        "source_reason_code": "build",
        "selection_metrics": {"candidate_median_ns": 0.0, "valid": False},
        "confirmation_metrics": None,
        "mechanism_ref": "mechanism-1",
        "artifact_refs": [],
    },
    {
        "trial_id": 2,
        "candidate_ref": "candidate-2",
        "parent_ref": "candidate-1",
        "verdict": "rejected",
        "reason_code": "no_improvement",
        "source_reason_code": "no-improvement",
        "selection_metrics": {
            "candidate_median_ns": 348940.0,
            "incumbent_median_ns": 352354.5,
            "delta_ns": 3414.5,
            "effect_ns": 3414.5,
            "speedup_pct": 0.97,
            "valid": True,
            "noise_ok": True,
            "regressions_ok": True,
        },
        "confirmation_metrics": None,
        "mechanism_ref": "mechanism-2",
        "artifact_refs": [],
    },
    {
        "trial_id": 3,
        "candidate_ref": "candidate-3",
        "parent_ref": "candidate-2",
        "verdict": "accepted",
        "reason_code": "accepted",
        "source_reason_code": "accepted",
        "selection_metrics": {
            "candidate_median_ns": 240000.0,
            "incumbent_median_ns": 352354.5,
            "delta_ns": 112354.5,
            "effect_ns": 112354.5,
            "speedup_pct": 31.9,
            "valid": True,
            "noise_ok": True,
            "regressions_ok": True,
        },
        "confirmation_metrics": {
            "candidate_median_ns": 241000.0,
            "incumbent_median_ns": 352000.0,
            "valid": True,
        },
        "mechanism_ref": "mechanism-3",
        "artifact_refs": ["artifact-1"],
    },
]


def _manifest() -> WorkflowManifest:
    def stage(stage_id: str, *targets: str) -> StageSpec:
        transitions = tuple(
            TransitionSpec(
                to=to, priority=0, reason="explicit_transition", guard=GuardSpec(kind="always")
            )
            for to in targets
        )
        return StageSpec(
            id=stage_id,
            prompt=stage_id,
            reads=(),
            writes=(WriteSpec(name="report", type="string", optional=False),),
            requires=frozenset(),
            transitions=transitions,
            execution=StageExecutionSpec(kind="tool"),
        )

    stages = (
        stage("Init", "RecordTrial"),
        # RecordTrial loops for trials 1-2 then exits to FinalEval; the graph
        # must declare the self-loop the ledger emits (review item 7).
        stage("RecordTrial", "RecordTrial", "FinalEval"),
        stage("FinalEval"),
    )
    return WorkflowManifest(
        workflow_id="CvxpygenH50Autoresearch",
        entry_stage_id="Init",
        exit_stage_ids=frozenset({"FinalEval"}),
        inputs=(InputSpec(name="eps_ns", type="number"),),
        capabilities=frozenset(),
        policies=(),
        stages=stages,
    )


def main() -> int:
    from nemoir_runtime.events import WorkflowEvent  # noqa: PLC0415

    out = REPO_ROOT / "docs" / "trace" / "schema" / "test-vectors" / "cvxpygen-public.nemotrace"
    viewer_copy = (
        REPO_ROOT / "web" / "nemoir-trace-viewer" / "public" / "fixtures" / "cvxpygen-public.nemotrace"
    )
    recorder = TraceRecorder.create(
        out,
        provenance=HostProvenance(
            frontend="nemo_dsl",
            target="python",
            compiler_version="0.1.9",
            ir_version="0.1",
            ir_sha256=None,  # synthetic fixture: provenance incomplete, playback only
        ),
        trace_id=FIXED_TRACE_ID,
        clock=lambda: FIXED_TIME,
    )
    manifest = _manifest()
    recorder.begin_run(manifest)
    seq = 0

    def emit(kind: str, **kwargs) -> int:
        nonlocal seq
        seq += 1
        event = WorkflowEvent(
            kind=kind,  # type: ignore[arg-type]
            run_id="x",
            sequence=seq,
            timestamp=FIXED_TIME,
            **kwargs,  # type: ignore[arg-type]
        )
        recorder.observe_workflow_event(event)
        return seq

    emit("run_started", metadata={"workflow_id": "CvxpygenH50Autoresearch", "entry": "Init"})
    # Init visit (no annotation).
    recorder.begin_stage_visit("Init")
    emit("stage_started", stage_id="Init")
    emit("stage_completed", stage_id="Init", output={"report": "x"})
    emit("transition_selected", stage_id="Init", transition_to="RecordTrial", metadata={"reason": "explicit_transition", "priority": 0})
    # Three RecordTrial visits, each with one anchored annotation.
    for trial in TRIALS:
        recorder.begin_stage_visit("RecordTrial")
        emit("stage_started", stage_id="RecordTrial")
        anchor = emit("stage_completed", stage_id="RecordTrial", output={"report": "x"})
        recorder.record_annotation(
            "nemoir.autoresearch/v1",
            "trial_finished",
            dict(trial),
            anchor_sequence=anchor,
        )
        emit(
            "transition_selected",
            stage_id="RecordTrial",
            transition_to="RecordTrial" if trial["trial_id"] < 3 else "FinalEval",
            metadata={"reason": "explicit_transition", "priority": 0},
        )
    recorder.begin_stage_visit("FinalEval")
    emit("stage_started", stage_id="FinalEval")
    emit("stage_completed", stage_id="FinalEval", output={"report": "x"})
    emit("run_completed", result={"output": {"summary": "synthetic"}})
    recorder.finish_run("complete")
    # Copy to viewer fixtures for bundled tests/manual loading.
    viewer_copy.parent.mkdir(parents=True, exist_ok=True)
    viewer_copy.write_bytes(out.read_bytes())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"wrote {viewer_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
