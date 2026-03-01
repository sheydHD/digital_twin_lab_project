# Development Guide

| Field        | Value                                       |
|--------------|---------------------------------------------|
| **Author**   | Antoni Dudij, Maksim Feldmann — RWTH Aachen |
| **Status**   | Review                                      |
| **Last Updated** | 2026-03-01                              |

---

## TL;DR

This guide covers everything a contributor needs: environment setup, running individual pipeline stages, executing the test suite, formatting and linting, debugging MCMC failures, and extending the system with new beam theories or FEM solvers. The recommended toolchain is `uv` for package management and `ruff` for all code quality checks.

---

## Environment Setup

All dependencies are declared in `pyproject.toml`. The project requires Python ≥ 3.10.

```bash
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project
make install        # creates .venv, installs all deps (uses uv if available, falls back to pip)
make check-deps     # verify every package imports correctly
make test           # confirm the test suite is green before making changes
```

Run `make` with no arguments to display all available targets with descriptions.

### Frontend Setup (optional)

The web dashboard requires [Bun](https://bun.sh). If you only intend to run the CLI pipeline, this step is not necessary.

```bash
make backend-dev    # FastAPI with hot-reload → http://localhost:8000
make frontend-dev   # Vite with HMR → http://localhost:5173
```

In dev mode, Vite proxies all `/api/*` and `/static/*` requests to the backend (configured in `apps/frontend/vite.config.ts`). Alternatively, `make up` launches both services via Docker Compose.

---

## Running the Pipeline

The full pipeline is invoked via `make run` (~30–40 minutes for the default 10 aspect ratios). Individual stages can be run in sequence when iterating on a specific component:

```bash
make run                               # all stages

python main.py --stage data            # data generation only
python main.py --stage calibration     # requires completed data stage
python main.py --stage analysis        # requires completed calibration stage
python main.py --stage report          # requires completed analysis stage

python main.py -a 5 -a 10 -a 20 --stage all   # subset of aspect ratios
python main.py --config configs/my_config.yaml  # custom config
python main.py --verbose --debug               # maximum logging
```

---

## Testing

The test suite lives in `tests/` and covers all domain layers. Each test file maps to a single module.

```bash
make test                                         # full suite
make test-cov                                     # with coverage report

pytest tests/test_beam_models.py -v               # single file
pytest tests/test_beam_models.py::test_deflection # single test
pytest tests/test_beam_models.py -v -s --pdb      # drop into debugger on failure
```

| Test File | Coverage |
|---|---|
| `test_beam_models.py` | EB and Timoshenko deflection, strain, and frequency formulas |
| `test_fem.py` | Element stiffness matrix, mesh generation, boundary conditions |
| `test_bayesian.py` | Prior sampling, likelihood evaluation, convergence diagnostics |
| `test_normalization.py` | Normalisation parameter computation and round-trip fidelity |
| `test_bridge_sampling.py` | Bridge estimator convergence and log-ML accuracy |
| `test_model_selection.py` | Bayes factor computation and evidence classification |
| `test_orchestrator.py` | Full pipeline integration over a small aspect-ratio subset |

---

## Code Quality

All formatting and linting is handled by `ruff`, which replaces `black`, `isort`, `flake8`, and `pylint` at 10–100× the speed. Type checking uses `mypy`.

```bash
make format          # auto-format and auto-fix all violations
make lint            # ruff check + mypy (CI gate)
make format-check    # CI-safe dry run — exits non-zero on any diff
make typecheck       # mypy only
make security        # bandit (static security analysis) + pip-audit (dependency CVEs)
```

---

## Project Structure

The repository separates delivery mechanisms from domain logic. The `apps/backend/core/` tree has zero import dependencies on FastAPI, React, or any other delivery framework.

```
apps/backend/
  api/               FastAPI application factory and route handlers
  core/
    models/          EulerBernoulliBeam, TimoshenkoBeam, BaseBeamModel
    fem/             TimoshenkoBeamFEM (active); legacy CantileverFEM (archived)
    data/            SyntheticDataGenerator, NoiseModel, SensorConfiguration
    bayesian/        BayesianCalibrator subclasses, BridgeSampler, BayesianModelSelector
    analysis/        BeamVisualization, ResultsReporter
    pipeline/        PipelineOrchestrator
    utils/           load_config, setup_logging
  schemas/           Pydantic v2 request/response models
  services/          Thin HTTP-to-domain adapter (simulation.py)
apps/frontend/
  src/
    views/           React components (Dashboard.tsx)
    viewmodels/      React hooks (useSimulationViewModel.ts)
    models/          TypeScript interfaces (types.ts)
```

---

## Debugging Guide

### PyMC Sampling Failure

When sampling raises `SamplingError: Initial evaluation of model at starting point failed!`, the log-probability is $-\infty$ at the initial point. Diagnose by evaluating the log-probability directly:

```python
with pm.Model() as model:
    # rebuild model identically to calibrator...
    test_point = model.initial_point()
    print(model.logp(test_point))   # should be a finite negative number
```

Common causes are a prior whose support excludes the initial point, or a normalisation scale that is zero (check `NormalizationParams.is_active`).

### Memory Errors During MCMC

Reduce the chain count and sample budget in the configuration:

```yaml
bayesian:
  n_samples: 400
  n_chains: 2
```

### MCMC Not Converging ($\hat{R} > 1.01$)

Increase the warmup budget and tighten the acceptance target to force more conservative step sizes:

```yaml
bayesian:
  n_tune: 1200
  target_accept: 0.99
```

If $\hat{R}$ remains elevated after 2 000 tuning steps, the posterior geometry is likely degenerate. Check whether the data contains sufficient signal relative to the noise level (`data_generation.noise_fraction`).

### Divergences Detected

Fewer than 1% divergent transitions is generally acceptable. Above 5%, increase `target_accept` to 0.99. Persistent divergences can indicate a prior–likelihood conflict; examine the posterior with `az.plot_pair(trace, divergences=True)`.

### Numerical Instability in Forward Model

All quantities entering the PyMC graph must be $\mathcal{O}(1)$. If a custom forward model produces very large or very small values, normalise explicitly:

```python
def safe_divide(num, denom, eps=1e-10):
    return num / (denom + eps)
```

---

## Profiling

```bash
# Function-level CPU profile
python -m cProfile -o profile.stats main.py
python -m pstats profile.stats

# Line-level CPU profile
kernprof -l -v main.py

# Memory profile
python -m memory_profiler main.py
```

The dominant cost is MCMC sampling (~6–8 minutes per aspect ratio). If profiling reveals unexpected time elsewhere, the most common culprit is unintentional serialisation overhead in `ProcessPoolExecutor` argument preparation.

---

## Extending the System

### Adding a New Beam Theory

1. Create a subclass of `BaseBeamModel` in `apps/backend/core/models/` and implement `compute_deflection`, `compute_rotation`, `compute_strain`, and `compute_natural_frequencies`.
2. Create a corresponding `BayesianCalibrator` subclass in `apps/backend/core/bayesian/calibration.py`. Implement `_pytensor_forward_normalized()` for symbolic MCMC compatibility.
3. Add a prior factory function alongside `create_default_priors` and `create_timoshenko_priors`.
4. Register the new model in `PipelineOrchestrator._setup_components()`. No other module requires modification.

### Adding a New FEM Solver

1. Implement the solver in `apps/backend/core/fem/` following the `TimoshenkoBeamFEM` interface (`solve()` returning a `BeamFEMResult`).
2. Update `SyntheticDataGenerator` to accept a solver class parameter and call it instead of `TimoshenkoBeamFEM`. The rest of the pipeline is agnostic to the solver used.

---

## Common Make Targets Reference

| Target | Action |
|---|---|
| `make install` | Create `.venv` and install all runtime + dev dependencies |
| `make run` | Execute the full pipeline |
| `make run-data` | Data generation stage only |
| `make run-calibration` | Calibration stage only |
| `make run-analysis` | Analysis stage only |
| `make run-report` | Reporting stage only |
| `make test` | Run the test suite |
| `make test-cov` | Test suite with HTML coverage report |
| `make format` | Auto-format and auto-fix with ruff |
| `make lint` | Lint + type-check (CI gate) |
| `make clean` | Remove `outputs/` and all Python cache directories |
| `make up` | Start backend + frontend via Docker Compose |
| `make backend-dev` | FastAPI dev server with hot-reload |
| `make frontend-dev` | Vite dev server with HMR |
