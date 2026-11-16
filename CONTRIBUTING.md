# Contributing

## Development

Python 3.12+ with [uv](https://docs.astral.sh/uv/).
Local development runs on 3.12 (the floor of the supported range) so 3.12-incompatible code is caught immediately; CI runs the full 3.12–3.15 matrix.

The repository is one uv workspace.
It covers `llm-catalog-core`, `llm-catalog-pydantic-ai`, `llm-catalog-ai-sdk`, and `llm-catalog-litellm`, all resolved into a single lockfile.

```bash
uv sync                                       # one venv, all members editable
uv run pytest                                 # mock-only; no real gateway or keys
uv run ruff check . && uv run ruff format --check .   # covers every package
uv run ty check packages                      # strict
```

Dependencies resolve with `exclude-newer = "1 day"`, so `uv lock --upgrade` and Dependabot only pick releases that have been public for at least a day.

## Commit messages

This repository uses [Conventional Commits](https://www.conventionalcommits.org); release-please derives each package's version bump and changelog from them.

- `feat: …` — a new feature.
- `fix: …` — a bug fix.
- `feat!: …` / `fix!: …`, or a `BREAKING CHANGE:` footer — a breaking change.
- `chore: …` / `docs: …` / `test: …` / `refactor: …` / `ci: …` — no release on their own.

Add a package scope when it helps, e.g. `feat(litellm): …`.

## Releases

Releases are automated: merging the release-please PR tags each changed package and publishes it to PyPI.
See the [README](README.md#releases) for the full flow.

## Boundaries

This is a public repository: ship only generic code, placeholder examples, and mock tests.
Never commit real gateway base URLs, model ids, capability values, or secrets.
