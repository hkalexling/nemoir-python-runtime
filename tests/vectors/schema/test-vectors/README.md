# NemoTrace 0.1 test vectors

All values are synthetic and safe to commit.

## Audit archive

`audit-valid.nemotrace` contains only the reserved audit entries from
`audit-valid/`; `content-identity.input.json` is explanatory and is not in the
archive.

Expected values:

```text
content identity: sha256:168cceb6d53a7318054d6a87ab977ea651c3f19adb4bf252af7793de3e535d95
fixture ZIP SHA-256 (informative): sha256:69cddbbc53374314157316cb4524b4d3ac346a367e12a10821b8235917e2ae1a
fixture ZIP bytes: 2252
```

The ZIP fixture uses sorted names, DEFLATE level 6, a
`1980-01-01T00:00:00` timestamp, Unix regular-file mode `0644`, and no archive
comment. Its physical hash is a regression value for the Python fixture builder,
not the cross-language bundle identity.

## IR fingerprint

`ir-fingerprint/minimal-ir.input.json` is a pretty serialization of one valid
minimal workflow. `minimal-ir.canonical.json` is the exact RFC-8785 canonical
byte sequence with no trailing LF. Its expected digest is in
`minimal-ir.sha256`.

## JCS

`jcs/` copies [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785) Section 3
primitive/property-order examples and all 26 Appendix B IEEE-754 samples.
Expected canonical files have no trailing LF.
Run `node docs/trace/validate_jcs.mjs` to verify ECMAScript serialization,
UTF-16 ordering, non-finite rejection, and lone-surrogate rejection.

## Crypto

`crypto/pbkdf2-aes-gcm.json` is a known-answer vector shared by Python and
WebCrypto. It intentionally exposes its passphrase, plaintext, salt, and nonce;
none may be copied into production. `vault.enc` is the vector's raw 121-byte
ciphertext-plus-tag payload (`sha256:ec2be91a16c177baf4cd42996cb3e1164341afda3eb0b9b8ea32a395f8619af9`).
`vault-record.ndjson` and `vault-meta.json` are structural examples for their
corresponding schemas.

## Redaction

`redaction/unsafe-input.json` contains fake credentials, PII, private source,
absolute paths, hidden-oracle text, and policy literals. None of its seeded
values may occur in any cleartext entry or filename in `audit-valid.nemotrace`.
`expected-public-event.json` illustrates one safe capability projection; it is
not a generic transformation of the unsafe object. `multiple-markers-event.json`
fixes sorted RFC 6901 pointer and opaque-marker behavior.

## Annotations

`annotations/autoresearch-trial-finished.json` is a public ledger annotation.
Validate the full record against `public-event.schema.json` and its payload
against `autoresearch-annotation.schema.json`.

`cvxpygen-public.nemotrace` (built by
`annotations/build_cvxpygen_fixture.py`) is a deterministic synthetic
three-trial audit archive (accepted + rejected) for local viewer testing
only. It is `audit` profile with incomplete provenance (synthetic manifest),
secret-free, and well under the 8 MiB public budget. It is not a publication
precedent: no `publication` transform, attestation, catalog entry, or Gist
upload belongs to Phase 3 (see `docs/trace/phase-3-summary.md`).

## Cross-language parity (Phase 1)

`parity/fake-run.json` is a scripted recorder drive: a three-stage manifest,
recorder config (fixed trace id, clock, path aliases, approved metrics, and
one synthetic secret), and an ordered op list of semantic hooks plus live
event observations, including one scanner-omitted record (sequence gap).
`parity/expected-ledger.ndjson` and `parity/expected-graph.json` are the
frozen canonical bytes both recorders must emit byte-for-byte:

- Python: `python/nemoir-runtime/tests/test_parity.py`
- TypeScript: `web/nemoir-runtime/src/__tests__/parity.test.ts`

Regenerate the expected files only by driving the reviewed recorder and
re-validating every ledger line against `public-event.schema.json` plus the
graph against `workflow-graph.schema.json`. A byte difference between the
two recorders is a blocking parity bug, not a fixture update.
