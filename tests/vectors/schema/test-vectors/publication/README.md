# Publication transform vectors (`publication-v1`)

Frozen cross-runtime inputs and outputs for the Phase 5 publication transform
(`plan.md` §5.4, `redaction-policy.md` §1). The Python
(`nemoir_runtime.publication`) and TypeScript (`web/nemoir-runtime/src/publication.ts`)
transformers read the same source archive and must reproduce every
`expected-*` file byte for byte.

Regenerate with
`python/nemoir-runtime/.venv/bin/python docs/trace/schema/test-vectors/publication/build_publication_vectors.py`.

## Source

`source.nemotrace` is a real `audit` capture produced by driving the Phase 1
parity fixture (`../parity/fake-run.json`) through the Python recorder with its
frozen clock and trace id. It deliberately exercises the interesting shapes:

- complete compiler provenance (a real `ir_sha256`), so it may be published;
- eight static tool names on 16 ledger records (dropped by default);
- one alias-relative `$workspace/f.txt` argument (opaque by default);
- a policy check, a model retry, a scanner-omitted record, and one
  `nemoir.autoresearch/v1` annotation.

The archive is a captured input: both runtimes read it, so its container bytes
never need to match. Everything derived from it is frozen below.

## Expected outputs

| File | Meaning |
|---|---|
| `expected-manifest.json` | projected manifest (`profile=publication`, `attested=true`, fresh trace id) |
| `expected-ledger.ndjson` | projected `public/public/events.ndjson` |
| `expected-graph.json` | rebuilt public workflow graph |
| `expected-summary.json` | recomputed derived summary |
| `expected-integrity.json` | entry index and content identity |
| `expected-digests.json` | source facts, projection digest, trace id, content identity, stats |
| `expected-report.json` | disclosure report (`attested=false`) a reviewer reads |
| `expected-attestation.json` | attestation built from that report |
| `expected-option-digests.json` | digest/trace-id/stats for the option variants |
| `attestation.json` | ready-to-load attestation covering the default projection |

`expected-report.json` and `expected-attestation.json` pin
`reviewed_at=2026-09-12T00:00:00.000Z`. The report carries no timestamp of its
own, so scan output is reproducible on any machine.

## Reviewed showcase fixture

`cvxpygen-publication.nemotrace` is the Phase 3 §10.2 item 5 fixture: a real
compiled `autoresearch.nemo` workflow run against the fake model and fake
harness, captured as an `audit` trace and then transformed by this publication
transform with a recorded attestation. Regenerate with
`build_cvxpygen_publication_fixture.py` (needs
`compiler/target/release/nemo`). It is a captured artifact, not
byte-reproducible; it is verifiable, attestation-bearing, vault-free, and
under the 8 MiB public budget.

`docs/trace/catalog.json` lists it as a local-only library entry (no Gist yet);
`docs/trace/validate_phase5.py` checks both against each other.

## What parity means here

Publication parity is asserted on **logical bytes**, not container bytes:

- every uncompressed entry (`manifest.json`, graph, ledger, summary,
  `integrity.json`) must be byte-identical across runtimes;
- the projection digest, trace id, content identity, disclosure report, and
  attestation document must be byte-identical across runtimes;
- the ZIP container itself is *not* compared: Python uses zlib and the browser
  runtime uses fflate, so the DEFLATE streams differ by a few bytes while the
  decompressed content is identical. Compressed size is therefore deliberately
  kept out of the report and CLI stdout (it stays on the library result for
  hosts that need a size budget).

## Option variants

`expected-option-digests.json` covers the four review decisions that change the
projection:

| Variant | Effect |
|---|---|
| `default` | tool names dropped, paths opaque |
| `allow_tool_names` | `reader`/`writer` retained; different digest |
| `keep_relative_paths` | `$workspace/f.txt` retained; different digest |
| `both` | both reviewed opt-outs together |

Each variant must produce a *different* projection digest from the default,
which is what makes the attestation binding meaningful.
