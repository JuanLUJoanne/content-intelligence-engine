# P3-1: Add GitHub Actions CI (Both Projects)

## Context
Both projects have 240+ tests each but no CI pipeline. Adding GitHub Actions shows engineering discipline and ensures the eval benchmarks stay valid.

## What to do

### Part A: content-intelligence-engine

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml', '**/requirements*.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Install dependencies
        run: pip install -e ".[dev]"
        # IMPORTANT: Read pyproject.toml to confirm the correct install command.
        # It might be: pip install -e . && pip install pytest pytest-asyncio
        # Or: pip install -r requirements.txt && pip install -r requirements-dev.txt

      - name: Lint
        run: ruff check src/ tests/
        # If ruff is not configured, use: python -m py_compile src/**/*.py
        continue-on-error: true  # Don't block on lint for now

      - name: Run tests
        run: pytest tests/ -v --tb=short

      - name: Run eval (DummyLLM)
        run: python scripts/run_eval.py --provider dummy
        # Only if P0-1 has been completed. If script doesn't exist yet, remove this step.
        continue-on-error: true
```

### Part B: agentic-rag

Same structure at `/Users/joelle/Projects/side-projects/agentic-rag/.github/workflows/ci.yml`.

Adapt:
- Install command to match agentic-rag's dependency setup
- Test path: `pytest tests/unit/ -v --tb=short`
- Eval step: `python scripts/benchmark_retrieval.py` (if exists from P0-3)

### IMPORTANT NOTES
- Read each project's actual install method before writing the CI file
- Use `continue-on-error: true` for lint and eval steps — don't block CI on these initially
- Keep it simple — one job, no Docker builds in CI, no deployment
- If the project uses Poetry, use `pip install poetry && poetry install` instead
