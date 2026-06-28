# Contributing

## Development

Python 3.10+ with [uv](https://docs.astral.sh/uv/).
Local development runs on 3.10 (the floor of the supported range) so 3.10-incompatible code is caught immediately; CI runs the full 3.10–3.14 matrix.

```bash
uv sync                                       # one venv, all three members editable
uv run pytest                                 # mock-only; no real gateway or keys
uv run ruff check . && uv run ruff format --check .
uv run ty check packages                      # strict
```

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
