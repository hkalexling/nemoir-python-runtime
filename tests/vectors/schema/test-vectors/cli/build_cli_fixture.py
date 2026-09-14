#!/usr/bin/env python3
"""Regenerate the CLI acceptance fixtures under this directory.

Two captured archives are built from the frozen Phase 4 drivers with their
public test passphrases:

- ``replay-e2e.nemotrace`` — a real `WorkflowRuntime` run of the tiny
  ReplayE2E workflow (tool -> model -> tool) recorded under
  ``profile="replay"`` (passphrase ``replay-e2e-passphrase``). Replays
  successfully and is the primary ``--unlock``/``--replay`` golden input.
- ``vault-fake-run.nemotrace`` — the rich synthetic vault driver
  (12 records incl. hostile-value scrubbing; passphrase
  ``phase4-vault-fake-passphrase-01``). Its manifest is deliberately
  degenerate, so it exercises ``verify``/``--unlock`` and replay-refusal
  robustness rather than a matched replay.

Both archives have captured timestamps/ids and (for the vault) random
salt/nonce, so their bytes are not reproducible across regenerations; they
are frozen *files*, verified by archive verification and unlock in both
runtimes. Run from the meta checkout root with the runtime venv:

    python/nemoir-runtime/.venv/bin/python \
        docs/trace/schema/test-vectors/cli/build_cli_fixture.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
META_ROOT = HERE.parents[4]
RUNTIME_ROOT = META_ROOT / "python" / "nemoir-runtime"
VAULT_VECTORS = HERE.parent / "vault"


def _replay_e2e() -> Path:
    from tests.test_replay import _record  # noqa: PLC0415

    produced, _adapter, _calls = asyncio.run(_record(HERE, deny=False))
    if produced.name != "run.nemotrace":
        msg = f"unexpected replay-e2e output name: {produced}"
        raise SystemExit(msg)
    target = HERE / "replay-e2e.nemotrace"
    shutil.move(str(produced), str(target))
    return target


def _vault_fake_run() -> Path:
    from tests.test_vault import _drive_fixture  # noqa: PLC0415

    produced = _drive_fixture(VAULT_VECTORS, capture=None)
    if produced.name != "vault.nemotrace":
        msg = f"unexpected vault output name: {produced}"
        raise SystemExit(msg)
    target = HERE / "vault-fake-run.nemotrace"
    shutil.move(str(produced), str(target))
    return target


def main() -> None:
    sys.path.insert(0, str(RUNTIME_ROOT))
    for target in (_replay_e2e(), _vault_fake_run()):
        print(f"wrote {target.relative_to(META_ROOT)} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
