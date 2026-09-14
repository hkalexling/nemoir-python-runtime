#!/usr/bin/env python3
"""Build the reviewed CVXPYgen *publication* fixture (Phase 5, WS-2b).

This is the Phase 3 §10.2 item 5 fixture: a small, secret-free,
`publication`-profile archive produced by the real toolchain — the
`autoresearch.nemo` workflow compiled by the `nemo` binary, run against the
fake model/harness with an audit recorder — and then transformed by the
publication transform with a recorded human attestation.

It is not byte-reproducible: the campaign uses the wall clock and the compiled
package embeds its own provenance. It *is* verifiable, schema-valid, small,
and free of secrets, absolute paths, raw reasoning, and private source.

```bash
python/nemoir-runtime/.venv/bin/python \
  docs/trace/schema/test-vectors/publication/build_cvxpygen_publication_fixture.py
```

The compiled package needs `compiler/target/release/nemo`
(`cargo build --release` in `compiler/`), and the demo's own venv is used for
the harness imports.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
DEMO_ROOT = REPO_ROOT / "demos" / "cvxpygen-autoresearch"
RUNTIME_SRC = REPO_ROOT / "python" / "nemoir-runtime" / "src"
NEMO = REPO_ROOT / "compiler" / "target" / "release" / "nemo"
FIXTURE = Path(__file__).resolve().parent / "cvxpygen-publication.nemotrace"
VIEWER_FIXTURE = (
    REPO_ROOT / "web" / "nemoir-trace-viewer" / "public" / "fixtures" / "cvxpygen-publication.nemotrace"
)
REVIEWED_AT = "2026-09-12T00:00:00.000Z"
REVIEWER = "NemoIR maintainers"
LICENSE = "Apache-2.0"
CONSENT = (
    "Fake-model campaign fixture reviewed field by field: no credentials, no "
    "absolute paths, no raw reasoning, no candidate source. Safe to publish."
)

sys.path.insert(0, str(RUNTIME_SRC))
sys.path.insert(0, str(DEMO_ROOT))

from nemoir_runtime.publication import (  # noqa: E402
    attestation_from_report,
    prepare_publication,
    scan_publication,
    write_attestation,
    write_publication_report,
)
from nemoir_runtime.trace import TraceConfig  # noqa: E402


def _load_demo_helper() -> tuple[object, object]:
    """Import the demo's fake-model test helpers without importing pytest."""
    spec = importlib.util.spec_from_file_location(
        "demo_helpers", DEMO_ROOT / "tests" / "test_compiled_workflow.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module


TRIAL_STORY: list[dict[str, object]] = [
    {
        "verdict": "rejected",
        "reason": "preflight_build",
        "source": "build",
        "candidate_ns": 0.0,
        "incumbent_ns": 360000.0,
        "valid": False,
        "confirmation": False,
    },
    {
        "verdict": "rejected",
        "reason": "no_improvement",
        "source": "no-improvement",
        "candidate_ns": 352354.5,
        "incumbent_ns": 360000.0,
        "valid": True,
        "confirmation": False,
    },
    {
        "verdict": "accepted",
        "reason": "accepted",
        "source": "accepted",
        "candidate_ns": 240000.0,
        "incumbent_ns": 352354.5,
        "valid": True,
        "confirmation": True,
    },
]


def _trial_payload(trial_id: int) -> dict[str, object]:
    story = TRIAL_STORY[trial_id - 1]
    selection: dict[str, object] = {
        "candidate_median_ns": story["candidate_ns"],
        "incumbent_median_ns": story["incumbent_ns"],
        "valid": story["valid"],
        "noise_ok": bool(story["valid"]),
        "regressions_ok": bool(story["valid"]),
    }
    confirmation = None
    if story["confirmation"]:
        confirmation = {
            "candidate_median_ns": story["candidate_ns"],
            "incumbent_median_ns": story["incumbent_ns"],
            "valid": True,
        }
    payload: dict[str, object] = {
        "trial_id": trial_id,
        "candidate_ref": f"candidate-{trial_id}",
        "parent_ref": None if trial_id == 1 else f"candidate-{trial_id - 1}",
        "verdict": story["verdict"],
        "reason_code": story["reason"],
        "source_reason_code": story["source"],
        "selection_metrics": selection,
        "confirmation_metrics": confirmation,
        "mechanism_ref": f"mechanism-{trial_id}",
        "artifact_refs": [],
        # Digests stay null even in publication: the config source is private
        # and has not been classified public (redaction-policy §11).
        "candidate_digest": None,
        "parent_digest": None,
    }
    return payload


def _compile(out_dir: Path) -> None:
    if not NEMO.exists():
        msg = f"nemo binary not found at {NEMO}; run `cargo build --release` in compiler/"
        raise SystemExit(msg)
    result = subprocess.run(  # noqa: S603
        [str(NEMO), "compile", str(DEMO_ROOT / "autoresearch.nemo"), "--target", "python", "-o", str(out_dir)],
        capture_output=True,
        cwd=str(DEMO_ROOT),
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode() if result.stderr else "(no stderr)\n")
        raise SystemExit("compile failed")


def main() -> int:
    helpers, _unused = _load_demo_helper()
    compiled_agent = helpers  # module holding the fake harness + tool builders
    build_inputs = compiled_agent._inputs_for  # type: ignore[attr-defined]
    make_tools = compiled_agent._make_basic_tools  # type: ignore[attr-defined]
    import_compiled = compiled_agent._import_compiled  # type: ignore[attr-defined]
    fake_state = compiled_agent.FakeHarnessState  # type: ignore[attr-defined]
    fake_model = compiled_agent.FakeModelAdapter  # type: ignore[attr-defined]

    with tempfile.TemporaryDirectory(prefix="cvxpygen-publication-") as raw:
        work = Path(raw)
        out_dir = work / "compiled"
        out_dir.mkdir()
        _compile(out_dir)
        pkg = import_compiled(out_dir)

        run_dir = work / "run"
        state = fake_state(max_trials=len(TRIAL_STORY))
        emitted: list[dict[str, object]] = []

        async def harness_handler(*, command: str, ctx: object) -> object:
            return state.handle(command)

        def hook(info: object) -> dict[str, object] | None:
            if not isinstance(info, dict) or info.get("stage_id") != "RecordTrial":
                return None
            trial_id = len(emitted) + 1
            payload = _trial_payload(trial_id)
            emitted.append(payload)
            spec: dict[str, object] = {
                "namespace": "nemoir.autoresearch/v1",
                "kind": "trial_finished",
                "payload": payload,
            }
            sequence = info.get("sequence")
            if isinstance(sequence, int) and sequence >= 1:
                spec["anchor_sequence"] = sequence
            return spec

        registry = make_tools(harness_handler)
        agent = pkg.Agent(
            model=fake_model(),
            tools=registry,
            trace=TraceConfig(
                path=run_dir / "run.nemotrace",
                path_aliases={
                    "$candidate_root": run_dir / "candidate",
                    "$agent_view": run_dir / "agent_view",
                    "$research_root": run_dir / "agent_view" / "research",
                    "$profile": DEMO_ROOT / "profiles" / "h50_mpc",
                },
                safe_path_aliases=frozenset({"$agent_view", "$research_root"}),
                approved_metrics=frozenset(
                    {
                        "RecordTrial.candidate_median_ns",
                        "RecordTrial.incumbent_median_ns",
                    }
                ),
                on_stage_completed=hook,
            ),
        )
        inputs = build_inputs(pkg, run_dir)

        async def run() -> object:
            return await agent.run(inputs)

        result = asyncio.run(run())
        if getattr(result, "output", None) is None:
            raise SystemExit("fake campaign did not produce output")
        audit_path = Path(agent.trace_path) if hasattr(agent, "trace_path") else run_dir / "run.nemotrace"

        projection = scan_publication(audit_path, archive_name="cvxpygen-publication.nemotrace")
        if not projection.ok:
            raise SystemExit(f"publication scan failed: {projection.findings[:5]}")
        report = projection.report(attested=False, attestation=None)
        report_path = work / "report.json"
        write_publication_report(report_path, report)
        attestation = attestation_from_report(
            report,
            reviewer=REVIEWER,
            license_id=LICENSE,
            consent=CONSENT,
            reviewed_at=REVIEWED_AT,
        )
        attestation_path = write_attestation(work / "attestation.json", attestation)
        prepare_publication(
            audit_path,
            FIXTURE,
            attestation=attestation,
            report_path=work / "final-report.json",
        )

    VIEWER_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    VIEWER_FIXTURE.write_bytes(FIXTURE.read_bytes())
    summary = {
        "fixture": FIXTURE.name,
        "bytes": FIXTURE.stat().st_size,
        "viewer_copy": str(VIEWER_FIXTURE.relative_to(REPO_ROOT)),
        "trials": [payload["verdict"] for payload in emitted],
        "attestation": str(attestation_path),
    }
    print(json.dumps(summary, indent=1))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
