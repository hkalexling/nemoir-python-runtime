"""``nemotrace verify`` CLI: golden outputs, exit codes, and passphrase sources.

The golden stdout files under the vendored
``tests/vectors/schema/test-vectors/cli/`` are shared with the TypeScript suite
(``web/nemoir-runtime/src/__tests__/cli.test.ts``) and asserted by both, so
the two CLIs provably agree byte-for-byte.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from nemoir_runtime import cli

VECTORS = Path(__file__).resolve().parent / "vectors" / "schema" / "test-vectors"
CLI_VECTORS = VECTORS / "cli"
AUDIT_FIXTURE = VECTORS / "audit-valid.nemotrace"
CVXPYGEN_FIXTURE = VECTORS / "cvxpygen-public.nemotrace"
REPLAY_FIXTURE = CLI_VECTORS / "replay-e2e.nemotrace"
VAULT_FIXTURE = CLI_VECTORS / "vault-fake-run.nemotrace"
REPLAY_PASSPHRASE = "replay-e2e-passphrase"  # noqa: S105
VAULT_PASSPHRASE = "phase4-vault-fake-passphrase-01"  # noqa: S105


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, str]:
    code = cli.main(argv)
    return code, capsys.readouterr().out


def _golden(name: str) -> str:
    return (CLI_VECTORS / name).read_text(encoding="utf-8")


def _tamper(source: Path, target: Path) -> Path:
    """Copy an archive with one public entry altered (hash mismatch)."""
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(target, "w") as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "public/events.ndjson":
                data = data + b" "
            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = info.compress_type
            zout.writestr(new_info, data)
    return target


@pytest.mark.parametrize(
    ("fixture", "golden"),
    [
        (AUDIT_FIXTURE, "expected-audit-valid.verify.txt"),
        (CVXPYGEN_FIXTURE, "expected-cvxpygen-public.verify.txt"),
        (VAULT_FIXTURE, "expected-vault-fake-run.verify.txt"),
    ],
)
def test_verify_matches_golden(
    capsys: pytest.CaptureFixture[str], fixture: Path, golden: str
) -> None:
    code, out = _run(capsys, ["verify", str(fixture)])
    assert code == 0
    assert out == _golden(golden)


def test_unlock_replay_fixture_matches_golden(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMOTRACE_TEST_PW", REPLAY_PASSPHRASE)
    code, out = _run(capsys, ["verify", str(REPLAY_FIXTURE), "--unlock", "env:NEMOTRACE_TEST_PW"])
    assert code == 0
    assert out == _golden("expected-replay-e2e.unlock.txt")


def test_unlock_vault_fixture_matches_golden(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMOTRACE_TEST_PW", VAULT_PASSPHRASE)
    code, out = _run(capsys, ["verify", str(VAULT_FIXTURE), "--unlock", "env:NEMOTRACE_TEST_PW"])
    assert code == 0
    assert out == _golden("expected-vault-fake-run.unlock.txt")


def test_replay_matches_golden_from_file_source(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    secret = tmp_path / "passphrase.txt"
    secret.write_text(REPLAY_PASSPHRASE + "\n", encoding="utf-8")
    code, out = _run(capsys, ["verify", str(REPLAY_FIXTURE), "--replay", f"file:{secret}"])
    assert code == 0
    assert out == _golden("expected-replay-e2e.replay.txt")


def test_wrong_passphrase_fails_generically(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMOTRACE_TEST_PW", "definitely-not-the-passphrase")
    code, out = _run(capsys, ["verify", str(REPLAY_FIXTURE), "--unlock", "env:NEMOTRACE_TEST_PW"])
    assert code == 1
    assert "error: vault unlock failed" in out
    assert "result: failed" in out
    assert "definitely-not-the-passphrase" not in out


def test_replay_refuses_audit_archive(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEMOTRACE_TEST_PW", REPLAY_PASSPHRASE)
    code, out = _run(capsys, ["verify", str(AUDIT_FIXTURE), "--replay", "env:NEMOTRACE_TEST_PW"])
    assert code == 1
    assert "replay: diverged" in out
    assert "divergence: archive has no replay vault" in out
    assert "replayability: playback-only" in out
    assert "result: failed" in out


def test_replay_that_cannot_run_reports_generic_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The degenerate vault fixture makes replay fail closed; no traceback.

    Shared with `nemotrace-js` via `expected-vault-fake-run.replay.txt`
    (M-C1): a manifest whose deterministic stages no taped tool can satisfy
    cannot re-execute, so neither runtime may report a synthetic divergence.
    """
    monkeypatch.setenv("NEMOTRACE_TEST_PW", VAULT_PASSPHRASE)
    code, out = _run(capsys, ["verify", str(VAULT_FIXTURE), "--replay", "env:NEMOTRACE_TEST_PW"])
    assert code == 1
    assert out == _golden("expected-vault-fake-run.replay.txt")
    assert "replay: error" in out
    assert "error: taped replay could not run" in out
    assert "result: failed" in out


def test_tampered_archive_fails(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    tampered = _tamper(AUDIT_FIXTURE, tmp_path / "tampered.nemotrace")
    code, out = _run(capsys, ["verify", str(tampered)])
    assert code == 1
    assert "integrity: failed" in out
    assert "result: failed" in out


def test_missing_archive_is_usage_error(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = cli.main(["verify", str(tmp_path / "missing.nemotrace")])
    captured = capsys.readouterr()
    assert code == 2
    assert "archive not found" in captured.err
    assert captured.out == ""


def test_bad_source_spec_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["verify", str(AUDIT_FIXTURE), "--unlock", "bogus"])
    captured = capsys.readouterr()
    assert code == 2
    assert "expected env:VAR | file:PATH | prompt" in captured.err
    assert captured.out == ""


def test_missing_env_var_is_usage_error(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEMOTRACE_MISSING_PW", raising=False)
    code = cli.main(["verify", str(AUDIT_FIXTURE), "--unlock", "env:NEMOTRACE_MISSING_PW"])
    captured = capsys.readouterr()
    assert code == 2
    assert "environment variable is missing or empty" in captured.err


def test_mutually_exclusive_flags_exit_two() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["verify", str(AUDIT_FIXTURE), "--unlock", "prompt", "--replay", "prompt"])
    assert excinfo.value.code == 2
