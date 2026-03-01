# Bayesian Model Selection for Beam Theory in Digital Twins

| Field | Value |
|---|---|
| **Authors** | Antoni Dudij, Maksim Feldmann — RWTH Aachen |
| **Status** | Review |
| **Last Updated** | 2026-03-01 |

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **TL;DR** — A Bayesian framework that uses MCMC calibration and bridge sampling to determine which beam theory — Euler-Bernoulli or Timoshenko — is best supported by sensor data for a given geometry. The empirical transition point is $L/h \approx 19.2$, providing quantitative probabilistic justification for the classical engineering rule of thumb. A React + FastAPI web dashboard is included for interactive use.

## Overview

This project uses Bayesian model selection to decide when Euler-Bernoulli (EB) beam theory is sufficient and when Timoshenko theory is needed. For each aspect ratio $L/h$ under study, the pipeline generates synthetic deflection and strain measurements from a high-fidelity 1D Timoshenko FEM, calibrates both beam theories via MCMC (PyMC/NUTS), estimates their marginal likelihoods via bridge sampling, and computes the log Bayes factor $\ln B_{EB/Timo}$.

The primary result is a data-driven transition point at $L/h \approx 19.2$: below this value Timoshenko is decisively preferred; above it Euler-Bernoulli is sufficient. This is consistent with the classical engineering rule of thumb ($L/h \approx 20$) but now carries a formal probabilistic justification rather than a heuristic one.

## Quick Start

```bash
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project
make install        # creates .venv and installs all deps (prefers uv, falls back to pip)
make check-deps     # verify every package imports correctly
make run            # full pipeline (~10–15 min on Linux with default config, 16GB RAM
```

```bash
make backend-dev    # terminal 1 → FastAPI on http://localhost:8000
make frontend-dev   # terminal 2 → Vite on  http://localhost:5173
```

Or bring up the full stack with Docker Compose:

```bash
make up
```

## Project Structure

The repository separates delivery (HTTP/UI) from domain logic. `apps/backend/core/` has no dependency on FastAPI or React and can be driven from the CLI, tests, or a notebook independently.

```
apps/
  backend/
    api/          FastAPI app factory and route handlers
    core/         Domain logic — fully decoupled from HTTP
      analysis/   Plotting and reporting
      bayesian/   PyMC calibration, bridge sampling, model selection
      data/       Synthetic FEM data generation
      fem/        1D Timoshenko beam FEM (ground truth)
      models/     Euler-Bernoulli and Timoshenko implementations
      pipeline/   PipelineOrchestrator (coordinates all stages)
      utils/      Config loader, logging setup
    schemas/      Pydantic v2 request / response models
    services/     Thin HTTP-to-domain adapter
  frontend/       React 19 + TypeScript + Vite dashboard (MVVM)

configs/          YAML configuration
outputs/          Generated data, figures, reports
tests/            Unit tests
docs/             Documentation
```

## Installation

```bash
make install        # creates .venv, installs all deps (prefers uv)
make check-deps     # verify every package imports correctly
```

To install manually into an existing environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+. Core dependencies: PyMC ≥ 5.10, ArviZ ≥ 0.17, FastAPI ≥ 0.110, NumPy, SciPy, Matplotlib.

## Usage

```bash
make run                                              # full pipeline
python main.py --stage data                           # data generation only
python main.py --stage calibration                    # calibration only
python main.py -a 5 -a 10 -a 20 --stage all          # custom aspect ratios
python main.py --config configs/my_config.yaml        # custom config file
```

Run `python main.py --help` for the full option reference.

## Results

Default configuration: structural steel ($E = 210$ GPa, $\nu = 0.3$), 1 m cantilever, 1 kN tip load.

| $L/h$ | $\ln B_{EB/Timo}$ | Recommended Model |
|---|---|---|
| 5 | −10.830 | Timoshenko |
| 8 | −7.377 | Timoshenko |
| 10 | −4.146 | Timoshenko |
| 12 | −3.595 | Timoshenko |
| 15 | −2.109 | Timoshenko |
| 20 | +0.420 | inconclusive |
| 30 | +0.255 | Euler-Bernoulli |
| 50 | −0.031 | Euler-Bernoulli |

**Transition point: $L/h \approx 19.2$.** Negative $\ln B$ favours Timoshenko; positive favours EB; $|\ln B| < 0.5$ is inconclusive. For the thick-beam regime ($L/h \leq 15$), Timoshenko is strongly preferred with $\ln B < -2.3$ (strong evidence on the Kass–Raftery scale). At very high $L/h$, both theories predict essentially identical deflections and the evidence cannot distinguish them.

## Configuration

All parameters live in `configs/default_config.yaml`. The most impactful knobs:

```yaml
beam_parameters:
  length: 1.0
  width: 0.1
  aspect_ratios: [5, 8, 10, 12, 15, 20, 30, 50]  # each adds ~7 min runtime

material:
  elastic_modulus: 210.0e9   # Pa — structural steel
  poisson_ratio: 0.3

bayesian:
  n_samples: 800    # increase to 3000 for publication-quality posteriors
  n_tune: 400
  n_chains: 2

data:
  noise_fraction: 0.0005    # Gaussian noise as fraction of signal amplitude
```

See [docs/parameters.md](docs/parameters.md) for the full parameter reference and study grid.

## Documentation

| Document | Description |
|---|---|
| [getting started](docs/getting-started.md) | Installation, first run, troubleshooting |
| [architecture](docs/architecture.md) | System design, component map, design decisions |
| [technical spec](docs/technical-spec.md) | Full design rationale, LaTeX formulas, sequence diagrams |
| [API reference](docs/api-reference.md) | Every public function and class |
| [parameters](docs/parameters.md) | Parameter tables, priors, sign conventions, study grid |
| [bayesian glossary](docs/bayesian-glossary.md) | All 22 statistical concepts, code locations, run-time status |
| [development guide](docs/development-guide.md) | Testing, linting, debugging, extension patterns |
| [backend README](apps/backend/README.md) | FastAPI REST API and service layer |
| [frontend README](apps/frontend/README.md) | React dashboard and MVVM architecture |
| [contributing](CONTRIBUTING.md) | Contribution guidelines |
| [changelog](CHANGELOG.md) | Version history |

## References

1. Timoshenko, S. P. (1921). On the correction for shear of the differential equation for transverse vibrations of prismatic bars. *Philosophical Magazine*, 41(245), 744–746.
2. Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *Journal of the American Statistical Association*, 90(430), 773–795.
3. Meng, X.-L., & Wong, W. H. (1996). Simulating ratios of normalizing constants via a simple identity. *Statistica Sinica*, 6, 831–860.
4. Gronau, Q. F., et al. (2017). A tutorial on bridge sampling. *Journal of Mathematical Psychology*, 81, 80–97.
5. Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using LOO-CV and WAIC. *Statistics and Computing*, 27(5), 1413–1432.
6. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.

## License

MIT — see [LICENSE](LICENSE).

Authors: Antoni Dudij, Maksim Feldmann — RWTH Aachen University, Digital Twins Lab.
