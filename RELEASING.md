# Releasing cliver-core

Both `cliver-core` and the `cliver` CLI are published. **Both packages share one version** —
tests (`test_versions_aligned.py`) fail if the versions or the CLI's exact pin diverge.

## One-time setup: trusted publishers

Publishing uses PyPI trusted publishing (OIDC), not API tokens. On PyPI, register a publisher
for **each** project — `cliver-core` and `cliver` — with these exact values:

| Field | Value |
|---|---|
| Owner | `cliver-project` |
| Repository | `CLIver` |
| Workflow | `release.yml` |
| Environment | `pypi` |

If the project does not exist yet, create a **pending publisher** with the settings above; the
first successful release creates the project. The PyPI environment is what lets you require
manual approval before a publish in the repository settings.

## Version rule

Both packages in `packages/*/pyproject.toml` must carry the exact same version. The release
tag is `v` + that version, e.g. `v0.3.0`. The release workflow refuses to build if the tag
and the `cliver-core` version disagree.

`0.3.0` is the first stable release of the 0.3 series: the public API stays backward compatible,
and `0.3.x` patch releases contain bug fixes and backward-compatible additions. Breaking
changes require a new minor release (`0.4.0`). Downstream projects may pin exactly
(`pip install cliver-core==0.3.0`) or use the compatible range
`cliver-core>=0.3.0,<0.4`. Exact pins remain the safest choice for reproducibility.

## Release candidate gate

Before an `rc` tag, all of the following must be true:

- The public API is stable: changes since the previous pre-release are bug fixes and
  backward-compatible additions only; breaking changes wait for a new minor release.
- `CHANGELOG.md` covers every commit since the previous release, including breaking changes and
  an upgrade section.
- The `cliver-core` README and the documentation install snippets pin the new version.
- `pyproject.toml` carries `Development Status :: 4 - Beta` and the Documentation / Changelog /
  Issues project URLs.
- The CI matrix is green for every Python version listed in the classifiers.
- `make format && make lint && make test && make live-verify` passes; the docs site builds.
- A wheel built from a clean `dist/` installs and imports in a fresh virtual environment.

## Release steps

1. Bump the version in both `packages/*/pyproject.toml` files.
2. Refresh the lockfile: `uv lock`.
3. `make format && make lint && make test && make live-verify`.
4. `make dist` — builds the wheel and sdist for both `cliver-core` and `cliver` into `dist/`.
5. Smoke-test the wheel (never publish an untested artifact):

   ```bash
   checkdir="$(mktemp -d)"
   uv venv "$checkdir/venv"
   uv pip install --python "$checkdir/venv/bin/python" dist/cliver_core-*.whl
   "$checkdir/venv/bin/python" -c "import cliver_core; print(cliver_core.AgentCore)"
   ```

6. Commit the version bump and merge the release branch into `main`.
7. Tag the release commit on `main` and push the tag:

   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

8. The `Release` workflow runs lint and the tests, builds both packages, verifies the tag
   against both versions, and publishes `cliver-core` first, then `cliver`, directly to
   **PyPI**. A failure at any step stops the chain.

## Publishing one package from an existing tag

The workflow also accepts a manual run for backfilling a package from a tag that is already
published. Run **Release** from the Actions tab (or `gh workflow run`) with:

- `tag`: the existing tag, for example `v0.3.2`
- `package`: `cliver`, `cliver-core`, or `all`

The workflow checks out that tag, runs the tests, verifies the tag against both package
versions, and publishes only the selected package. Re-publishing an already-published
`cliver-core` version is safe (PyPI ignores identical files) but unnecessary.

## Notes

- Test files are excluded from the wheel (`[tool.uv.build-backend] wheel-exclude = ["tests"]`);
  they remain in the sdist.
- Re-running a release for an already-published version is safe: PyPI ignores identical files.
  It never allows re-uploading a file with changed contents, so a broken release needs a new
  version.
- If a release turns out to be broken after publishing, bump the pre-release (`0.3.0rc2`, …)
  and release again.
- No `PYPI_TOKEN` secret is used or required.
