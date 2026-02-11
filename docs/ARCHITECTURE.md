# architecture

## overview

The project is a modular pipeline with four stages: data generation, Bayesian calibration, model selection, and reporting. Each stage is independent and can be run separately.

### design principles

1. separation of concerns — clear boundaries between data, inference, analysis, reporting
2. reproducibility — configuration-driven, deterministic outputs
3. extensibility — easy to add new beam theories or analysis methods

## high-level architecture

```
config (YAML)
    |
    v
pipeline orchestrator (apps/pipeline/orchestrator.py)
    |
    +-- data generation (apps/data/)
    |       uses: FEM models (apps/fem/), beam theories (apps/models/)
    |
    +-- bayesian calibration (apps/bayesian/calibration.py)
    |       uses: PyMC, normalization, beam forward models
    |
    +-- model selection (apps/bayesian/model_selection.py, bridge_sampling.py)
    |       uses: bridge sampling for marginal likelihoods
    |
    +-- reporting (apps/analysis/)
            uses: visualization.py, reporter.py
```

## core components

### 1. data generation (`apps/data/`)

Generates synthetic measurement data from a 1D Timoshenko beam FEM.

```
input: beam geometry, material properties, loading
  -> 1D Timoshenko beam FEM solver
  -> ground truth: deflections and strains at sensor locations
  -> add 0.05% Gaussian noise
  -> output: synthetic dataset
```

Uses 1D Timoshenko beam FEM instead of 2D continuum FEM because:
- exact consistency with analytical beam theory assumptions
- 100x faster (200 vs 20,000 elements)
- 0.0000% error vs analytical Timoshenko solution

### 2. FEM layer (`apps/fem/`)

**`beam_fem.py`** — active solver:
- `TimoshenkoBeamFEM`: 2-node Timoshenko beam elements with coupled bending-shear stiffness
- DOFs per node: deflection w, rotation theta
- adaptive mesh: n_elements = 4 * (L/h), capped at 200

**`cantilever_fem.py`** — legacy 2D plane stress FEM, kept for reference, not used in production.

### 3. beam theory models (`apps/models/`)

Analytical beam theory implementations for forward predictions during MCMC.

**`EulerBernoulliBeam`**: plane sections stay perpendicular. Deflection: delta = PL^3/(3EI). Valid for L/h > 20.

**`TimoshenkoBeam`**: plane sections rotate independently. Deflection: delta = PL^3/(3EI) + PL/(kGA). Valid for all L/h, especially < 20.

Both provide `deflection(x, P)` and `strain(x, y, P)` methods.

### 4. bayesian inference (`apps/bayesian/`)

**`calibration.py`** — PyMC model definitions and MCMC sampling:
- builds a PyMC model with priors on E (and nu for Timoshenko) plus Gaussian likelihood
- all quantities normalized to O(1) for stable NUTS sampling
- runs 2 chains x 800 samples, 400 tuning steps, target_accept = 0.95
- computes WAIC as a diagnostic
- estimates marginal likelihood via bridge sampling

**`bridge_sampling.py`** — marginal likelihood estimation:
- fits a multivariate normal proposal to the posterior
- iterates the optimal bridge function (Meng & Wong, 1996)
- returns log marginal likelihood estimate

**`model_selection.py`** — model comparison:
- computes log Bayes factor from bridge sampling marginal likelihoods
- interprets evidence using Kass & Raftery (1995) scale
- finds transition aspect ratio via interpolation

**`normalization.py`** — scaling for MCMC stability:
- E_scale = 210e9, displacement_scale = max|w_observed|
- all sampling in normalized space, results denormalized after

**`hyperparameter_optimization.py`** — Optuna-based prior tuning and frequency analysis.

### 5. analysis and visualization (`apps/analysis/`)

**`visualization.py`** — plotting functions for aspect ratio study, deflection profiles, convergence diagnostics, etc.

**`reporter.py`** — generates text, JSON, and CSV reports with model selection results.

### 6. pipeline orchestration (`apps/pipeline/`)

**`orchestrator.py`** coordinates the full workflow:

```
stage 1: data generation
  for each L/h: run FEM, add noise, save dataset

stage 2: bayesian calibration
  for each dataset: calibrate EB and Timoshenko via MCMC
  compute marginal likelihoods via bridge sampling

stage 3: model selection
  compute log Bayes factors, determine preferred models
  find transition aspect ratio

stage 4: reporting
  generate plots, text reports, CSV/JSON output

stage 5: frequency analysis
  analytical EB vs Timoshenko natural frequency comparison
```

Aspect ratios are processed sequentially to manage memory. MCMC chains run in parallel within each calibration.

## data flow

```
config YAML
  |
  v
data generation: for each L/h -> TimoshenkoBeamFEM.solve() -> add noise -> save
  |
  v
calibration: for each dataset -> PyMC model (EB) -> NUTS -> trace
                               -> PyMC model (Timo) -> NUTS -> trace
                               -> bridge sampling -> log marginal likelihood
  |
  v
model selection: log BF = log ML(EB) - log ML(Timo) -> evidence labels
  |
  v
reporting: summary tables, plots, CSV/JSON
```

## design decisions

### 1. why 1D FEM instead of 2D FEM?

1D Timoshenko beam FEM as ground truth because:
- same assumptions as analytical theories (no 2D constraint effects)
- 100x faster
- 0.0000% error vs analytical Timoshenko
- cannot model 2D/3D effects, but those are not needed for beam theory validation

### 2. why bridge sampling instead of WAIC for model selection?

Bridge sampling computes true marginal likelihoods, giving proper Bayes factors. WAIC approximates predictive accuracy but is not a direct evidence measure. Bridge sampling is more computationally expensive but gives principled model selection results. WAIC is still computed as a diagnostic.

### 3. why NUTS sampler?

- fewer samples needed for convergence than random-walk MH
- automatic step size tuning
- handles correlated parameters well
- cost per sample is higher, but offset by needing ~800 vs ~10,000+ samples

### 4. why sequential aspect ratios?

- PyMC traces are large (~100 MB each)
- avoids multiprocessing conflicts with PyMC
- individual runs already parallelized (MCMC chains)
- total runtime ~30-40 min for 8 aspect ratios

## performance

| component | time |
|-----------|------|
| data generation (8 L/h) | ~5 seconds |
| calibration per L/h (EB + Timo) | ~6-8 minutes |
| model selection | ~10 seconds |
| reporting | ~30 seconds |
| full pipeline | ~30-40 minutes |

Peak memory: ~1 GB during calibration.

## configuration schema

```yaml
beam_parameters:
  length: float              # beam length [m]
  width: float               # beam width [m]
  height: float              # beam height [m]
  aspect_ratios: list[float] # L/h ratios to study

loading:
  force: float               # applied force [N]

material:
  elastic_modulus: float     # E [Pa]
  poisson_ratio: float       # nu [-]

bayesian:
  n_samples: int             # MCMC samples per chain
  n_tune: int                # tuning samples
  n_chains: int              # parallel chains
  target_accept: float       # NUTS acceptance target

data:
  n_sensors: int             # measurement points
  noise_fraction: float      # relative noise level

output:
  save_traces: bool
  save_figures: bool
  figure_dpi: int
```

## testing

Unit tests in `tests/`:
- `test_beam_models.py` — deflection and strain formula validation
- `test_fem.py` — element stiffness, mesh generation, boundary conditions
- `test_bayesian.py` — prior sampling, likelihood evaluation
- `test_normalization.py` — normalization parameter computation

Run with `make test` or `pytest tests/ -v`.

## extending the project

### adding a new beam theory

1. create a new class in `apps/models/` inheriting from `BeamModel`
2. implement `deflection()` and `strain()` methods
3. add a calibrator subclass in `apps/bayesian/calibration.py`
4. update the orchestrator to include it in model comparison

### adding a new FEM solver

1. implement in `apps/fem/`
2. update `SyntheticDataGenerator` to use the new solver

## dependencies

Core: PyMC >= 5.10, ArviZ >= 0.17, NumPy >= 1.24, SciPy >= 1.11
Visualization: Matplotlib >= 3.7, Seaborn >= 0.12
Utilities: PyYAML >= 6.0, Rich >= 13.0, Click >= 8.1

## references

- Gelman et al., Bayesian Data Analysis (3rd ed.)
- Hughes, T.J.R., The Finite Element Method
- Timoshenko, S.P., Theory of Elastic Stability
- Meng & Wong (1996), Simulating ratios of normalizing constants
