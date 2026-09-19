# Releasing minviol

Publishing is close to irreversible: a PyPI release can be *yanked* but never
deleted, and the project name is claimed permanently. The steps below are
ordered so that everything reversible happens first.

## One-time setup

### 1. Configure Trusted Publishing on PyPI

This is the only step that needs your PyPI account, and it is deliberately not
automatable — it is what proves to PyPI that this repository may publish under
this name.

Go to <https://pypi.org/manage/account/publishing/> and add a **pending
publisher**:

| Field | Value |
|---|---|
| PyPI Project Name | `minviol` |
| Owner | `mahdims` |
| Repository name | `minviol` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

A pending publisher works before the project exists; the first successful run
creates it. **No API token is involved at any point.** PyPI verifies a
short-lived OIDC token that GitHub mints for this repository, workflow and
environment, so there is no secret to store, leak or rotate.

### 2. Create the `pypi` environment on GitHub

Settings → Environments → New environment → `pypi`. Add yourself as a required
reviewer if you want a manual approval between the tag and the upload; the
workflow is already structured for it.

## Each release

1. **Update the version** in `pyproject.toml`. The package reads `__version__`
   from the installed distribution, so this is the only place it appears.
2. **Check the README's claims still hold.** It quotes measured numbers. If the
   search changed, they changed.
3. Commit, push, and let CI go green.
4. **Tag and push:**

   ```bash
   git tag -a v0.1.0 -m "First release"
   git push origin v0.1.0
   ```

The workflow then runs the tests on two Python versions, builds, refuses to
continue if the tag disagrees with the version in `pyproject.toml`, and uploads.

## Dry run first

To exercise everything without claiming anything, add a TestPyPI pending
publisher with the same settings and point the publish step at
`https://test.pypi.org/legacy/`. TestPyPI is periodically wiped and is not a
real publication.

Locally, this much is worth doing before any tag:

```bash
python -m build
python -m twine check dist/*
python -m venv /tmp/fresh && /tmp/fresh/bin/pip install dist/*.whl
/tmp/fresh/bin/python -c "import minviol; print(minviol.__version__)"
```

## What a release claims

Be careful here. The README is accurate and should stay that way:

- The search is a **heuristic**. No certificate, no infeasibility proof.
- **CUDA is unmeasured.** Every published number is Apple MPS or x86 CPU. This
  is a GPU package whose main target is untested, and that is the most important
  thing for a prospective user to know.
- The sparse backend is faster at **scoring** and slower over a **whole pass**.
- Two benchmark instances never reach feasibility.

`Development Status :: 3 - Alpha` in `pyproject.toml` says the same thing to
anyone reading the classifiers, and should not be raised without the measurements
to back it.
