# Releasing `nemoir-runtime`

The canonical release version is `[project].version` in `pyproject.toml`. A
release is a reviewed, increasing `X.Y.Z` version bump merged to `master`; the
`vX.Y.Z` tag and GitHub Release are created by automation.

## One-time setup

1. Create a GitHub Environment named **`pypi`** in
   `hkalexling/nemoir-python-runtime`. It may require an approval if you want a
   final human gate; it needs no secrets.
2. In PyPI project settings → **Publishing**, add a GitHub Trusted Publisher:
   - owner: `hkalexling`
   - repository: `nemoir-python-runtime`
   - workflow filename: `release.yml`
   - environment: `pypi`

Those values must exactly match `.github/workflows/release.yml`. The publish
job uses OIDC and `pypa/gh-action-pypi-publish`; no PyPI token is stored in
GitHub.

## Automatic release flow

A push to `master` that changes `pyproject.toml` runs `Release`.

1. The workflow compares `[project].version` from `github.event.before` and
   the exact pushed commit. Only an increasing canonical SemVer version is
   eligible; non-version metadata changes do not publish.
2. Unprivileged jobs run Ruff, Pyright, pytest on Python 3.11–3.14, build a
   clean wheel and sdist, run `twine check`, install the wheel into a temporary
   environment, and save the distributions with `SHA256SUMS`.
3. The only privileged job downloads and verifies those artifacts, creates a
   draft GitHub Release at the exact commit, publishes via PyPI Trusted
   Publishing, creates a GitHub provenance attestation when the repository is
   public, and finalizes the release.

The privileged job never checks out, builds, or executes repository source.

## Dry runs and recovery

Use **Actions → Release → Run workflow** for a dry run; `dry_run` defaults to
`true`. Give `ref` an exact commit SHA to validate or recover an older
candidate.

PyPI versions are immutable. A retry is permitted only when an existing PyPI
version, tag, and **draft** GitHub Release all refer to the same commit. Any
final release, different tag target, or unrelated existing PyPI version is
refused rather than overwritten.
