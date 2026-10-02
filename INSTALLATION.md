# Installation and development setup

## Supported environments

RiskLabAI 3.2.0 supports:

- CPython 3.12, 3.13, and 3.14
- NumPy `>=2.2,<3`

The complete tested matrix and feature-specific limitations are documented in
[`docs/compatibility.md`](docs/compatibility.md).

## Install from PyPI

Create an isolated environment with either `venv` or Conda.

### `venv`

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### Conda

```bash
conda create -n risklabai-3 python=3.12 -y
conda activate risklabai-3
```

Then install RiskLabAI:

```bash
python -m pip install --upgrade pip
python -m pip install RiskLabAI
```

Confirm the installed version:

```bash
python -c "import RiskLabAI; print(RiskLabAI.__version__)"
```

## Optional dependency groups

Install only the capabilities you need:

```bash
python -m pip install "RiskLabAI[speed]"
python -m pip install "RiskLabAI[pde]"
python -m pip install "RiskLabAI[synth]"
python -m pip install "RiskLabAI[hpo]"
python -m pip install "RiskLabAI[plot]"
python -m pip install "RiskLabAI[symbolic]"
python -m pip install "RiskLabAI[profile]"
python -m pip install "RiskLabAI[simulation]"
python -m pip install "RiskLabAI[changepoints]"
```

The groups provide:

| Extra | Capability |
|---|---|
| `speed` | Numba acceleration |
| `pde` | PyTorch support for Deep-BSDE, neural SDEs, TimeGAN, and deep hedging |
| `synth` | synthetic-control utilities using QuantEcon |
| `hpo` | hyperparameter tuning using Optuna |
| `plot` | Matplotlib, Seaborn, and Plotly helpers |
| `symbolic` | symbolic analysis using SymPy |
| `profile` | memory profiling |
| `simulation` | simulation progress support |
| `changepoints` | changepoint detection on Python 3.12-3.13 |
| `test` | the supported pytest test runner |

There is intentionally no `all` extra. To reproduce the complete optional CI
environment, install the declared groups explicitly:

```bash
python -m pip install "RiskLabAI[speed,pde,synth,hpo,plot,symbolic,profile,simulation,changepoints,test]"
```

Notes:

- The `speed` group currently constrains NumPy to `<2.6`; the base package does
  not have that additional restriction.
- The `changepoints` dependency is available on Python 3.12 and 3.13. On
  Python 3.14, the rest of RiskLabAI remains supported while this optional
  backend is unavailable.
- PyWavelets is part of the base installation because its absence changes
  analytical behavior rather than only disabling acceleration.

## Additional method dependencies

Selected methods added in 3.2.0 use CVXPY (constrained or robust optimization),
River (ADWIN drift adaptation), or PySensemakr (sensitivity analysis).
These packages are installed separately when needed; they are not base
dependencies or named extras. Local Python 3.13 validation used CVXPY 1.9.3,
River 0.26.1, and PySensemakr 0.0.8. This does not establish compatibility with
every version of those packages or every supported interpreter.

The 3.2.0 version is a release candidate until published. An unpinned PyPI
installation retrieves the latest published version.

## Development installation

Clone the repository, switch to a supported Python version, and run the
following commands from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]" "black==26.5.1" "ruff==0.15.17"
```

To work on optional features, install the relevant groups. To reproduce the
complete optional CI environment:

```bash
python -m pip install -e ".[speed,pde,synth,hpo,plot,symbolic,profile,simulation,changepoints,test]"
```

## Tests and static checks

Run the complete test suite:

```bash
python -m pytest -q
```

Run the static checks used by the release workflow:

```bash
black --check src/RiskLabAI/causal_factor_analysis test/causal_factor_analysis
ruff check src/RiskLabAI/causal_factor_analysis test/causal_factor_analysis
```

The package uses a `src` layout. An editable installation is therefore the
supported way to import local source during development.
