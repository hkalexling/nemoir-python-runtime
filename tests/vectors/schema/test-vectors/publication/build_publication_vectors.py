#!/usr/bin/env python3
"""Regenerate the frozen publication vectors.

Run from the repo root with the Python runtime venv:

```bash
python/nemoir-runtime/.venv/bin/python \
  docs/trace/schema/test-vectors/publication/build_publication_vectors.py
```

The source archive is produced by driving the Phase 1 parity fixture through
the Python recorder: it is a complete-provenance `audit` trace with tool names,
an alias-relative path argument, a policy check, a retry, a dropped record, and
a frozen clock/trace id. Both runtimes then read this one archive and must
reproduce the frozen expected files byte for byte.

`expected-report.json` and `expected-attestation.json` use a fixed
`reviewed_at`, so a reviewer can regenerate the same sidecars on any machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "python" / "nemoir-runtime" / "src"))
sys.path.insert(0, str(ROOT / "python" / "nemoir-runtime"))

from nemoir_runtime.canonical import to_canonical_bytes  # noqa: E402
from nemoir_runtime.publication import (  # noqa: E402
    PublicationOptions,
    attestation_from_report,
    build_attestation,
    scan_publication,
)
from tests.test_parity import _drive_fixture  # noqa: E402

VECTORS = Path(__file__).resolve().parent
SOURCE = VECTORS / "source.nemotrace"
REVIEWED_AT = "2026-09-12T00:00:00.000Z"
CONSENT = "I reviewed the disclosure report and certify this trace is safe to publish."
ARCHIVE_NAME = "source.nemotrace"


def _write(name: str, data: bytes) -> None:
    (VECTORS / name).write_bytes(data)


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        built = _drive_fixture(Path(work) / "src")
        SOURCE.write_bytes(Path(built).read_bytes())
    projection = scan_publication(SOURCE, archive_name=ARCHIVE_NAME)
    entries = projection.entries
    _write("expected-manifest.json", entries["manifest.json"])
    _write("expected-graph.json", entries["public/workflow.graph.json"])
    _write("expected-ledger.ndjson", entries["public/events.ndjson"])
    _write("expected-summary.json", entries["public/summary.json"])
    _write("expected-integrity.json", entries["integrity.json"])
    digests = {
        "source": projection.source.as_dict(),
        "projection_sha256": projection.projection_sha256,
        "trace_id": projection.trace_id,
        "content_identity": projection.content_identity,
        "stats": projection.stats.as_dict(),
        "findings": list(projection.findings),
        "entries": list(projection.scanned_entries),
    }
    _write("expected-digests.json", to_canonical_bytes(digests) + b"\n")
    report = projection.report(attested=False, attestation=None)
    _write("expected-report.json", to_canonical_bytes(report) + b"\n")
    attestation = attestation_from_report(
        report,
        reviewer="Alex Ling",
        license_id="CC-BY-4.0",
        consent=CONSENT,
        reviewed_at=REVIEWED_AT,
    )
    _write("expected-attestation.json", to_canonical_bytes(attestation.as_dict()) + b"\n")
    # Digest-only expectations for the option variants (allowlisted tool names
    # and kept relative paths) so both runtimes can prove option handling
    # without freezing a second ledger.
    variants: dict[str, dict[str, object]] = {}
    for label, options in (
        ("default", PublicationOptions()),
        ("allow_tool_names", PublicationOptions(allow_tool_names=("reader", "writer"))),
        ("keep_relative_paths", PublicationOptions(keep_relative_paths=True)),
        ("both", PublicationOptions(allow_tool_names=("reader",), keep_relative_paths=True)),
    ):
        variant = scan_publication(SOURCE, options=options, archive_name=ARCHIVE_NAME)
        variants[label] = {
            "projection_sha256": variant.projection_sha256,
            "trace_id": variant.trace_id,
            "content_identity": variant.content_identity,
            "stats": variant.stats.as_dict(),
        }
    _write("expected-option-digests.json", to_canonical_bytes(variants) + b"\n")
    # A deterministic in-tree attestation copy for the tests to load.
    _write("attestation.json", to_canonical_bytes(attestation.as_dict()) + b"\n")
    print(  # noqa: T201
        json.dumps(
            {
                "projection_sha256": projection.projection_sha256,
                "trace_id": projection.trace_id,
                "content_identity": projection.content_identity,
                "events": projection.stats.event_count,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
