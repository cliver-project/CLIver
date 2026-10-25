# Releasing cliver-core

Only `cliver-core` is published today. The `cliver` CLI application stays unpublished, but
**both packages share one version** — a test (`test_versions_aligned.py`) fails if they
diverge.

## One-time setup: trusted publishers

Publishing uses PyPI trusted publishing (OIDC), not API tokens. On PyPI, register a publisher
for `cliver-core` with these exact values:

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
tag is `v` + that version, e.g. `v0.3.0a5`. The release workflow refuses to build if the tag
and the `cliver-core` version disagree.

Because `0.3.0a5` is a pre-release, downstream projects should pin it exactly:
`pip install cliver-core==0.3.0a5`. The exact pin selects the pre-release without `--pre`;
using `--pre` also allows pre-release versions of transitive dependencies (it can pull an
unrelated dev build), so prefer the exact pin.

## Release steps

1. Bump the version in both `packages/*/pyproject.toml` files.
2. Refresh the lockfile: `uv lock`.
3. `make format && make lint && make test`.
4. `make dist` — builds `dist/cliver_core-<version>-py3-none-any.whl` and the sdist.
5. Smoke-test the wheel (never publish an untested artifact):

   ```bash
   uv venv /tmp/cliver-core-check
   uv pip install --python /tmp/cliver-core-check/bin/python dist/cliver_core-*.whl
   /tmp/cliver-core-check/bin/python -c "import cliver_core; print(cliver_core.AgentCore)"
   ```

6. Commit the version bump, then tag and push:

   ```bash
   git tag v0.3.0a5
   git push origin v0.3.0a5
   ```

7. The `Release` workflow runs the tests, builds, verifies the tag, and publishes directly to
   **PyPI**. A failure at any step stops the chain.

## Notes

- Test files are excluded from the wheel (`[tool.uv.build-backend] wheel-exclude = ["tests"]`);
  they remain in the sdist.
- Re-running a release for an already-published version is safe: PyPI ignores identical files.
  It never allows re-uploading a file with changed contents, so a broken release needs a new
  version.
- If a release turns out to be broken after publishing, bump the pre-release (`0.3.0a5`, …)
  and release again.
- No `PYPI_TOKEN` secret is used or required.
