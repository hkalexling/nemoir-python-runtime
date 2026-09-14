# CLI test vectors (`nemotrace` / `nemotrace-js`)

Frozen inputs and expected outputs for the verifier CLI surface
(`plan.md` §11 Phase 5, WS-1). The Python `nemotrace` CLI and the
TypeScript `nemotrace-js` CLI must produce byte-identical stdout and the
same exit codes for every case below.

## Fixtures

| File | Passphrase | Purpose |
|---|---|---|
| `replay-e2e.nemotrace` | `replay-e2e-passphrase` | replay-valid ReplayE2E run (tool → model → tool): verify/unlock/replay goldens |
| `vault-fake-run.nemotrace` | `phase4-vault-fake-passphrase-01` | rich 12-record vault driver (hostile-value scrubbing): verify/unlock goldens and replay-refusal robustness |

Both are captured fixtures (timestamps/ids, plus random vault salt/nonce),
so their bytes are not reproducible across regenerations. Regenerate with
`build_cli_fixture.py` (its docstring has the exact command); the fixtures
are validated by archive verification and unlock in both runtimes. The
passphrases are public test values, not secrets.

## Expected outputs

| Golden | Command |
|---|---|
| `expected-audit-valid.verify.txt` | `verify <audit-valid.nemotrace>` |
| `expected-cvxpygen-public.verify.txt` | `verify <cvxpygen-public.nemotrace>` |
| `expected-vault-fake-run.verify.txt` | `verify vault-fake-run.nemotrace` |
| `expected-vault-fake-run.unlock.txt` | `verify vault-fake-run.nemotrace --unlock env:VAR` |
| `expected-replay-e2e.unlock.txt` | `verify replay-e2e.nemotrace --unlock env:VAR` |
| `expected-replay-e2e.replay.txt` | `verify replay-e2e.nemotrace --replay env:VAR` |

## Contract

Stable stdout lines, in order:

```text
archive: <basename>                 # path-independent so goldens are portable
format: <format|->
trace_id: <id|->
workflow: <workflow id|->
profile: <capture profile|->
status: <status|->
content_identity: <sha256:…|->
integrity: <passed|failed>
structural: <passed|passed-with-warnings|failed>
semantic: <not-evaluated|passed|failed>
replayability: <none|playback-only|taped-replay>
warnings: <n>
errors: <n>
replay: <matched|diverged|error>    # only with --replay
replay_status: <status>             # only with --replay
replay_steps: <n>                   # only with --replay
divergences: <n>                    # only with --replay
warning: …                          # first 10, then warning_more: N
error: …                            # first 10, then error_more: N
divergence: …                       # first 10, then divergence_more: N
result: <ok|failed>
```

Exit codes: `0` = requested level passed, `1` = it failed, `2` = usage
error (missing archive, bad passphrase source, unknown flags). Stderr
carries the `error:` line for exit-2 cases; passphrase values, vault
plaintext, and stack traces are never printed.

## Replay of a degenerate manifest (parity-covered)

`vault-fake-run.nemotrace` verifies and unlocks in both runtimes, but its
manifest is deliberately degenerate (the frozen vault driver builds tool
stages without a capability), so no taped tool can satisfy those stages.
Both libraries now fail closed at replay construction and report the same
error state — `replay: error`, `errors: 1`, `error: taped replay could not
run`, `result: failed`, exit 1 — frozen as
`expected-vault-fake-run.replay.txt`. Reporting `diverged` would overclaim
that the recorded path actually re-executed.

## Parity check

```bash
python3 docs/trace/check_cli_parity.py
```

Runs both CLIs over these fixtures plus tamper / wrong-passphrase / usage
cases and asserts identical stdout and exit codes, checking the goldens at
the same time. Prerequisites: the runtime venv with the `trace` extra
(`python/nemoir-runtime/.venv`) and a built web runtime
(`npm run build` in `web/nemoir-runtime`).
