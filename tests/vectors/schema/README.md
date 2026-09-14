# NemoTrace 0.1 internal format contract

> **Status:** Phase 0 format candidate  
> **Scope:** internal implementation contract for Python/web interoperability  
> **Public documentation:** promote stable semantics to `compiler/docs/trace.md` only after the Phase 1 format freeze

This directory is the source of truth for the first NemoTrace implementation.
It refines Sections 7 and 8 of [`../plan.md`](../plan.md) into testable wire
rules. Runtime and viewer implementations must consume the schemas and vectors
rather than infer a format from one language's objects.

## 1. Versioning and compatibility

The format identifier is `nemoir.trace/0.1`.

- `0` is the format major. Readers must reject an unsupported major.
- `.1` is the initial schema revision within major 0.
- A major-0 reader must ignore unknown object fields after applying archive
  limits and scanning them as untrusted data.
- The checked-in schemas are strict **writer-conformance** schemas and reject
  unknown fields. A forward-compatible reader validates the required known
  projection rather than treating an older writer schema as its parser.
- A major-0 writer emits only fields defined by this contract or by an approved
  namespaced annotation schema.
- JSON Schema is structural validation, not a complete disclosure control.
  Allowed strings and annotation values still require projection and scanning.

Legacy Python snake_case and web camelCase JSONL are import formats, not
`nemoir.trace/0.1` artifacts.

## 2. Single-file container

A `.nemotrace` is a ZIP file with this reserved layout:

```text
manifest.json                    required
integrity.json                   required
public/workflow.graph.json       required
public/events.ndjson             required
public/summary.json              optional, derived cache
private/vault.enc                required only for replay
private/vault.meta.json          required only for replay
```

No other entry is valid in format 0.1. In particular, archive directories,
symlinks, comments, executable files, and arbitrary attachments are not
allowed. Artifact bodies belong in the encrypted vault, not as extra ZIP
entries.

Writers must:

1. sort entry names by their UTF-8 byte sequence;
2. use ZIP method 8 (DEFLATE) with an implementation-fixed compression level
   for JSON/NDJSON and method 0 (STORE) for `private/vault.enc`;
3. set every entry timestamp to `1980-01-01T00:00:00`;
4. emit no archive comment or nonessential extra fields;
5. mark entries as regular non-executable files;
6. finish all payload entries and `integrity.json` before atomically renaming
   the final archive; and
7. leave an explicitly named local partial/journal on interrupted finalization,
   never a seemingly complete `.nemotrace`.

Deflate encoders are not byte-stable across languages. Therefore the normative
cross-language identity is the content identity in Section 5, not the SHA-256
of the physical ZIP. A writer must be byte-deterministic for the same inputs
within its supported implementation/version. Python and web writers must
produce byte-identical **uncompressed canonical entries** and the same content
identity, but their ZIP bytes may differ.

Readers must reject duplicate names, absolute names, backslashes, empty path
segments, `.`/`..` segments, NULs, directory/symlink entries, encrypted ZIP
members, DEFLATE on `vault.enc`, STORE on JSON/NDJSON, unsupported compression
methods, and any entry not in the list above. The reader validates metadata and
size limits before inflation when possible.

### Initial limits

| Limit | Publication/Gist | Local audit/replay viewer |
|---|---:|---:|
| Compressed archive | 8 MiB | 64 MiB |
| Total uncompressed entries | 64 MiB | 256 MiB |
| One uncompressed entry | 48 MiB | 192 MiB |
| Entry count | 7 | 7 |
| Public ledger records | 100,000 | 500,000 |
| Compression ratio, archive or entry | 100:1 | 200:1 |

The entry-count limit includes `integrity.json`; its own index therefore holds
at most six payload entries. Exceeding a limit is a hard load/publication
failure, not a prompt to continue. The 8 MiB Gist budget stays below GitHub's
documented 10 MiB raw-file boundary.

## 3. Canonical JSON and NDJSON

Canonical JSON means RFC 8785 JSON Canonicalization Scheme (JCS): UTF-8,
lexicographically sorted object keys, deterministic ECMAScript number
serialization, no insignificant whitespace, and no duplicate object keys.
Inputs must be valid I-JSON: no NaN/infinity, invalid Unicode, or integers that
cannot be represented exactly where a JavaScript reader must preserve them.

- Each `.json` payload entry is exactly one JCS value with **no trailing LF**.
- Each line of an `.ndjson` payload is one JCS object followed by `LF` (`0x0a`).
- An NDJSON entry ends with one LF and contains no blank lines or CRLF.
- Raw entry SHA-256 values include every byte, including NDJSON line endings.
- Base64url fields use the RFC 4648 URL alphabet without `=` padding.

The checked-in `test-vectors/jcs/` cases cover RFC 8785 primitive
serialization, UTF-16 property ordering, non-finite rejection, lone-surrogate
rejection, and all 26 Appendix B IEEE-754 samples. Every Python, Rust, and web
canonicalizer must pass these vectors; ordinary `sort_keys` JSON output is not
a substitute for JCS.

### Timestamps

Persisted timestamps are UTC with exactly millisecond precision:

```text
YYYY-MM-DDTHH:mm:ss.sssZ
```

Python datetimes are converted to UTC and truncated (not rounded) to
milliseconds. Web timestamps already have this form. Live `WorkflowEvent`
objects retain their existing language-native precision and shape.

## 4. Public ledger

`public/events.ndjson` contains canonical snake_case projections of the existing
14 live event kinds plus the trace-only `annotation` record. It is not a generic
serialization of `WorkflowEvent`.

For workflow events:

- every record's `run_id` equals `manifest.trace_id`; an audit writer may reuse
  the runtime's random run ID, while publication creates a fresh ID and rewrites
  every record to avoid correlating a public artifact with a private run;
- `sequence` is the existing per-run live event sequence and starts at 1;
- filtering/redaction never renumbers it, so gaps are valid;
- all stage-scoped records carry `stage_visit_id`;
- model records carry a run-local `model_call_id`;
- tool records carry a run-local `tool_call_id`; and
- absent values are omitted rather than serialized as `null`, except where a
  schema explicitly uses `null` semantically.

Run-local IDs are deterministic counters assigned in observation order:

```text
stage visit: s-1, s-2, ...
model call:  m-1, m-2, ...
tool call:   t-1, t-2, ...
redaction:   r-1, r-2, ...
```

These IDs have no meaning outside one trace and reveal no source value.

A trace-only annotation has no live `sequence`. It is written on its own line
at the point the recorder receives it and may carry `anchor_sequence` to link
it to the most recent relevant live event. File order is authoritative for
annotation presentation. A recognized namespace/kind must additionally pass
its dedicated schema; an unknown annotation exposes only its namespace/kind and
a redaction marker as payload. Its shape is:

```json
{
  "kind": "annotation",
  "run_id": "...",
  "timestamp": "...",
  "anchor_sequence": 42,
  "annotation": {
    "namespace": "nemoir.autoresearch/v1",
    "kind": "trial_finished",
    "payload": {}
  },
  "redacted_fields": []
}
```

`redacted_fields` contains sorted, unique RFC 6901 JSON Pointers. Removed values
are replaced with this marker where retaining field presence is useful:

```json
{
  "$redacted": {
    "token": "r-1",
    "reason": "private_content",
    "value_type": "string",
    "length": 42
  }
}
```

`length` is optional and means UTF-8 bytes for strings/binary or element count
for arrays. It must be omitted where length itself is sensitive. A token is
opaque and must never be derived from the removed value.

## 5. Integrity and content identity

`integrity.json` lists every payload entry except itself. Each entry records the
path, media type, uncompressed byte length, and SHA-256 of the exact
uncompressed bytes.

`integrity.json` is excluded from its own entry list to avoid a hash cycle. The
content identity is calculated as follows:

1. Sort payload records by `path` UTF-8 bytes.
2. Project each to exactly `path`, `sha256`, and `uncompressed_bytes`.
3. Build this object:

   ```json
   {
     "format": "nemoir.trace.content-identity/0.1",
     "entries": [/* sorted projections */]
   }
   ```

4. RFC-8785-canonicalize the object with no trailing LF.
5. SHA-256 those bytes and render lowercase as `sha256:<64 hex>`.

This detects accidental or unauthenticated modification but does not identify a
publisher. Anyone can replace payloads and recompute an unsigned integrity
index. Publisher signatures are intentionally deferred.

Derived `public/summary.json` is only a cache. `event_count` and
`counts_by_kind` include every ledger line, including annotations;
`stage_visit_count` counts unique visit IDs. A reader recomputes the summary
from the ledger and warns on any mismatch even when its entry hash is valid.

## 6. Workflow fingerprint and public graph

`workflow.ir_sha256` is the SHA-256 of RFC-8785 canonical bytes of the exact
serialized `WorkflowIr` that entered a backend. It includes `source`, prompts,
all ordering, and all otherwise-serialized IR fields. It does not hash the
pretty `src/workflow.json` bytes and has no trailing LF.

The complete IR is sensitive and may be placed only in the encrypted vault.
The cleartext hash is protocol identity for the aggregate compiled object; it
must not be generalized into hashing individual removed secrets. A future
semantic-equivalence hash that excludes source provenance would be a separate,
explicit field.

`public/workflow.graph.json` is an allowlisted projection. It contains IDs,
entry/exits, execution kinds, declared write names/types, transition endpoints,
priorities/guard kinds, capabilities, and opaque policy refs. It never contains
prompts, guard expressions, policy source text, source paths, exec arguments,
or tool descriptions.

## 7. Vault plaintext and encryption metadata

When decrypted, `private/vault.enc` is canonical NDJSON whose lines validate
against `vault-record.schema.json`. It contains only supplemental/private
records keyed to public sequences and run-local call/visit IDs; it does not
needlessly duplicate the public ledger.

`private/vault.meta.json` describes the one 0.1 codec. `vault.enc` is raw
AES-GCM ciphertext followed by the 16-byte authentication tag. The metadata
contains no key or passphrase. Its authenticated data binds the vault to the
trace/IR identity and exact manifest, public ledger, and graph hashes. See
[`../spikes/crypto-interop.md`](../spikes/crypto-interop.md) for the proven
parameters and threat limits.

## 8. Schema inventory

| File | Validates |
|---|---|
| `common.schema.json` | Digests, IDs, timestamps, redaction markers, JSON values |
| `manifest.schema.json` | `manifest.json` and profile invariants |
| `integrity.schema.json` | `integrity.json` |
| `workflow-graph.schema.json` | Safe graph projection |
| `public-event.schema.json` | One public NDJSON record |
| `summary.schema.json` | Optional derived summary |
| `vault-meta.schema.json` | Cleartext KDF/AEAD metadata |
| `vault-record.schema.json` | One decrypted vault record |
| `autoresearch-annotation.schema.json` | `nemoir.autoresearch/v1` `trial_finished` payload |

The `https://nemoir.dev/...` `$id` values are stable logical identifiers during
this internal phase; implementations bundle/register these files locally and
must not depend on network schema retrieval.

These are strict writer schemas. Same-major reader tolerance is an explicit
parser behavior, not `additionalProperties` in the writer schema. Passing a
strict schema still does not make a semantically sensitive allowed value safe;
the field-level policy and scanner remain mandatory.

## 9. Test vectors

- `test-vectors/audit-valid/` contains the canonical entries for a synthetic
  audit trace.
- `test-vectors/audit-valid.nemotrace` is a deterministic Python ZIP assembly
  of those entries. Its physical hash is informative, not protocol identity.
- `test-vectors/ir-fingerprint/` fixes one canonical IR hash.
- `test-vectors/jcs/` carries RFC 8785 Unicode and number conformance cases.
- `test-vectors/crypto/` fixes the cross-language vault codec.
- `test-vectors/redaction/` contains deliberately fake unsafe input and its
  expected safe projection.

All credentials, PII, paths, and private source in these vectors are synthetic.
