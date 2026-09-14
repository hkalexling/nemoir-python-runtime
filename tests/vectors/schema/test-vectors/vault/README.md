# NemoTrace Phase 4 — vault test vectors

> **Status:** Phase 4 shared fixture contract
> **Drivers:** `python/nemoir-runtime/tests/test_vault.py`,
>   `web/nemoir-runtime/src/__tests__/vault.test.ts`
> **Normative crypto:** `../spikes/crypto-interop.md`

Both runtimes drive their recorder through `vault-fake-run.json` ops in
order (fixed clock + trace id + passphrase) and must emit:

1. byte-identical `public/events.ndjson` and `public/workflow.graph.json`
   (same parity rule as `../parity/`), **and**
2. byte-identical decrypted vault plaintext
   (`expected-vault-records.ndjson`, one JCS object + LF per line).

## Files

| File | Contents |
|---|---|
| `vault-fake-run.json` | MiniReplay workflow (tool → model → tool + deny policy), recorder config, ordered ops including the new Phase 4 vault hooks |
| `expected-vault-records.ndjson` | Frozen decrypted vault plaintext for the driver above (record ids `v-1…`, capture order) |
| `README.md` | This file |

## Driver notes

- `config.passphrase` is test-only (`phase4-vault-fake-passphrase-01`).
  Production passphrases are out-of-band, never checked in.
- The driver deliberately includes hostile values that must **not** survive
  into cleartext *or* vault plaintext:
  - `sk-vault-TEST-secret-9999` (registered secret → opaque marker);
  - `Authorization: Bearer …` / `api_key` fields (credentials → dropped);
  - `/home/vault-user/…` (unregistered absolute path → opaque `path-N`);
  - `reasoning` text with default capture (reasoning excluded unless
    `vault_capture.include_reasoning` is true).
- `record_transition_evaluation` precedes its `transition_selected`
  observation (runtime order); the recorder links the vault record to the
  following ledger sequence for that visit.
- Tamper cases (wrong passphrase, flipped ciphertext bit, flipped AAD
  input, non-600k iterations) are exercised in-code by both runtimes
  against archives they just wrote — no checked-in tamper files.

## Freezing procedure

`expected-vault-records.ndjson` is frozen from the Python implementation
and asserted byte-for-byte by both suites. Regenerate only when the vault
schema/record contract intentionally changes (bump + note here):

```bash
python/nemoir-runtime/.venv/bin/python -m pytest python/nemoir-runtime/tests/test_vault.py -q
# on intentional contract change: update expected-vault-records.ndjson from
# the Python test's regenerated output, then re-freeze web/vault.test.ts
```
