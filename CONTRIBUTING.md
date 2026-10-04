# Contributing

## Development

Python 3.12+ with [uv](https://docs.astral.sh/uv/).
Local development runs on 3.12 (the floor of the supported range) so 3.12-incompatible code is caught immediately; CI runs the full 3.12–3.14 matrix.

The repository holds two uv projects.
The workspace at the root covers `llm-catalog-core`, `llm-catalog-pydantic-ai`, and `llm-catalog-ai-sdk`.
`llm-catalog-litellm` is a standalone project with its own lockfile, because LiteLLM requires `openai<3` while the workspace members resolve to `openai>=3`.

```bash
uv sync                                       # the workspace: one venv, members editable
uv run pytest                                 # mock-only; no real gateway or keys
uv run ruff check . && uv run ruff format --check .   # covers every package
uv run ty check packages                      # strict

cd packages/llm-catalog-litellm               # the standalone project
uv sync
uv run pytest
uv run ty check
```

A change to `llm-catalog-core` can affect both projects, so run both test suites.

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
