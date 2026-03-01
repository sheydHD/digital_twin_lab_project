# Getting Started

| Field        | Value                                       |
|--------------|---------------------------------------------|
| **Author**   | Antoni Dudij, Maksim Feldmann — RWTH Aachen |
| **Status**   | Review                                      |
| **Last Updated** | 2026-03-01                              |

---

## TL;DR

Clone the repository, run `make install` then `make run`, and the full pipeline will generate model selection results in `outputs/reports/study_summary.txt`. For the interactive web dashboard, start the FastAPI backend and Vite frontend in two terminals using `make backend-dev` and `make frontend-dev`. The pipeline takes 30–40 minutes on the default configuration; for a faster smoke test, reduce `aspect_ratios` to three values and `n_samples` to 400.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | Required for all pipeline modes |
| Git | any | — |
| RAM | ≥ 4 GB | 8 GB recommended for full MCMC run |
| [Bun](https://bun.sh) | latest | Frontend only; skip if running CLI pipeline |

---

## Installation

### Using `uv` (recommended — 10–100× faster than pip)

```bash
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Using pip

```bash
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`make install` wraps either of the above automatically, preferring `uv` if it is on the PATH.

### Verifying the Installation

```bash
make check-deps     # verifies every package imports cleanly
make test           # runs the full test suite; all tests should pass
```

---

## Quick Start

```bash
make run            # full pipeline (~30–40 min)
```

When the run completes, the key outputs are:

| File | Contents |
|---|---|
| `outputs/reports/study_summary.txt` | Model selection table: log Bayes factor and recommendation per $L/h$ |
| `outputs/reports/results.csv` | Raw numerical results for downstream analysis |
| `outputs/reports/results.json` | Same data in JSON format |
| `outputs/figures/aspect_ratio_study.png` | Bayes factor vs aspect ratio plot |
| `outputs/figures/detailed/` | Per-aspect-ratio calibration and convergence plots |

`study_summary.txt` is the primary result. It reports the log Bayes factor $\ln B_{EB/Timo}$ and the recommended beam theory for each aspect ratio, along with the interpolated transition point $L/h \approx 19.2$.

---

## Running the Web Dashboard

Start the backend and frontend in two separate terminals:

```bash
# Terminal 1
make backend-dev    # FastAPI → http://localhost:8000

# Terminal 2
make frontend-dev   # Vite → http://localhost:5173
```

Open `http://localhost:5173` in a browser. The dashboard allows you to configure beam geometry, material, and sampling hyperparameters, then launch the pipeline and watch live progress — all without touching the command line again.

Alternatively, bring up both services and a reverse proxy with Docker Compose:

```bash
make up             # backend :8000 + frontend :5173
```

---

## Configuration

All parameters are in `configs/default_config.yaml`. The most important knobs are:

```yaml
beam_parameters:
  length: 1.0               # beam length [m]
  width: 0.1                # beam width [m]
  aspect_ratios:            # L/h values to sweep
    - 5
    - 8
    - 10
    - 12
    - 15
    - 20
    - 30
    - 50

material:
  elastic_modulus: 2.1e+11  # Young's modulus [Pa] — structural steel
  poisson_ratio: 0.3

bayesian:
  n_samples: 800            # NUTS draws per chain
  n_tune: 400               # warmup steps (discarded)
  n_chains: 2               # parallel chains per calibration

data_generation:
  n_displacement_sensors: 5
  noise_fraction: 0.0005    # Gaussian noise as fraction of signal amplitude
```

### Reducing Runtime for a Smoke Test

To get results in ~5 minutes rather than 40, restrict the sweep to three aspect ratios and halve the sample count:

```yaml
beam_parameters:
  aspect_ratios: [5, 15, 30]

bayesian:
  n_samples: 400
  n_tune: 200
```

### Increasing Accuracy for Publication Quality

For more reliable marginal likelihood estimates and tighter posterior intervals:

```yaml
bayesian:
  n_samples: 3000
  n_tune: 1500

data_generation:
  n_displacement_sensors: 10
  noise_fraction: 0.0002
```

---

## Interpreting Results

The log Bayes factor $\ln B_{EB/Timo} = \ln p(\mathbf{y}\mid M_{EB}) - \ln p(\mathbf{y}\mid M_{Timo})$ is the central decision statistic. Positive means EB is better supported by the data; negative means Timoshenko is. Magnitude determines confidence following the Kass–Raftery (1995) scale:

| $\ln B_{EB/Timo}$ | Interpretation |
|---|---|
| $< -2.3$ | Strong preference for Timoshenko |
| $-2.3$ to $-0.5$ | Moderate preference for Timoshenko |
| $-0.5$ to $+0.5$ in transition zone ($L/h \approx 15$–$19$) | Inconclusive — defaults to Euler-Bernoulli |
| $|\ln B| \approx 0$ for $L/h \geq 20$ | Euler-Bernoulli — shear negligible; near-zero log BF is physically correct, not ambiguous |
| $+0.5$ to $+2.3$ | Moderate preference for Euler-Bernoulli |
| $> +2.3$ | Strong preference for Euler-Bernoulli |

On the default steel cantilever, the transition from Timoshenko to EB preference occurs at $L/h \approx 19.2$.

---

## Troubleshooting

**pip install fails with compilation errors (Ubuntu/Debian):**
```bash
sudo apt-get install python3-dev build-essential
```

**pip install fails with compilation errors (macOS):**
```bash
xcode-select --install
```

**PyMC import error despite successful install:** PyMC has a known conda/pip environment conflict. Install inside a fresh conda environment:
```bash
conda install -c conda-forge pymc
```

**Memory error during MCMC:** Reduce chain count and sample budget in the config:
```yaml
bayesian:
  n_samples: 400
  n_chains: 2
```

**MCMC not converging ($\hat{R} > 1.01$):** Increase the warmup budget:
```yaml
bayesian:
  n_tune: 1200
  target_accept: 0.99
```

**Divergences detected:** Fewer than 1% is generally acceptable. Above 5%, raise `target_accept` to 0.99.

---

## Next Steps

Once the pipeline runs successfully, the natural reading order for deeper understanding is:

1. [architecture.md](architecture.md) — how the six domain layers fit together.
2. [technical-spec.md](technical-spec.md) — full design rationale, LaTeX formulas, and sequence diagrams.
3. [bayesian-glossary.md](bayesian-glossary.md) — definitions and code locations for every statistical concept.
4. [api-reference.md](api-reference.md) — function signatures for every public interface.
5. [parameters.md](parameters.md) — parametric study grid and sign conventions.
6. [development-guide.md](development-guide.md) — testing, linting, debugging, and extension patterns.
