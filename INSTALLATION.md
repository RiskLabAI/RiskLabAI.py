# Installation & Development Setup

## Install (users)

```bash
pip install RiskLabAI
```

Optional extras pull in heavier dependencies only when you need them:

```bash
pip install "RiskLabAI[pde]"     # torch — the Deep-BSDE PDE solver
pip install "RiskLabAI[plot]"    # matplotlib / seaborn / plotly — plotting helpers
pip install "RiskLabAI[synth]"   # quantecon — synthetic-data utilities
pip install "RiskLabAI[all]"     # everything above
```

The base install is intentionally lightweight: `import RiskLabAI` does not pull
in torch or plotting libraries — sub-packages that need them are imported lazily.

## Development setup (contributors)

### 1. Create and activate an environment

```bash
conda create -n risklab python=3.11 -y
conda activate risklab
```

(Any Python 3.9–3.12 works; a venv is fine too.)

### 2. Install in editable mode with all extras and test tooling

Dependencies are declared in `pyproject.toml` (there is **no** `requirements.txt`).
From the repository root:

```bash
pip install -e ".[all]" pytest black ruff
```

The editable install (`-e`) links the package to your source tree so the test
suite imports your local code.

> On some setuptools versions the plain editable install does not expose all
> sub-modules. If `import RiskLabAI.backtest.bet_sizing` fails, reinstall with
> the compatibility mode:
> ```bash
> pip install -e . --config-settings editable_mode=compat
> ```

### 3. Run the tests

```bash
pytest -q --ignore=test/pde
```

`test/pde` is skipped unless you have a working `torch` runtime (install the
`[pde]` extra to include it).

### 4. Lint and format before committing

```bash
black RiskLabAI test
ruff check RiskLabAI test
```
