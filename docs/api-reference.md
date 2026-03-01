# API Reference

| Field        | Value                                       |
|--------------|---------------------------------------------|
| **Author**   | Antoni Dudij, Maksim Feldmann — RWTH Aachen |
| **Status**   | Review                                      |
| **Last Updated** | 2026-03-01                              |

---

## TL;DR

This document is the authoritative function-level reference for all public Python interfaces in `apps/backend/core/`. Beam models, the FEM solver, the data generator, all Bayesian components, visualization, reporting, orchestration, and configuration loading are covered in the order they are invoked by the pipeline.

---

## Beam Models (`apps/backend/core/models/`)

### Dataclasses

`BeamGeometry`, `MaterialProperties`, and `LoadCase` are the three shared dataclasses defined in `apps.backend.core.models.base_beam`. They are the universal currency passed between FEM, forward models, and calibrators.

```python
from apps.backend.core.models.base_beam import BeamGeometry, MaterialProperties, LoadCase

geometry = BeamGeometry(length=1.0, height=0.1, width=0.1)
material = MaterialProperties(elastic_modulus=210e9, poisson_ratio=0.3)
load = LoadCase(point_load=1000.0)
```

`BeamGeometry` automatically computes `area` ($A = b \cdot h$) and `moment_of_inertia` ($I = bh^3/12$) for rectangular cross-sections and exposes an `aspect_ratio` property ($L/h$).

`MaterialProperties` derives `shear_modulus` ($G = E / (2(1+\nu))$) if not supplied directly. It also carries `density` and `shear_correction_factor` ($\kappa$, defaulting to $5/6$ for rectangular sections).

`LoadCase` supports `point_load`, `distributed_load`, `moment`, and an optional `frequency` field for dynamic analyses.

### BaseBeamModel

Abstract base class for all beam theory implementations. The constructor signature is `(geometry: BeamGeometry, material: MaterialProperties, name: str)`.

```python
class BaseBeamModel(ABC):
    @abstractmethod
    def compute_deflection(self, x: np.ndarray, load: LoadCase) -> np.ndarray: ...

    @abstractmethod
    def compute_rotation(self, x: np.ndarray, load: LoadCase) -> np.ndarray: ...

    @abstractmethod
    def compute_strain(self, x: np.ndarray, y: float, load: LoadCase) -> np.ndarray: ...

    @abstractmethod
    def compute_natural_frequencies(self, n_modes: int = 5) -> np.ndarray: ...

    # Concrete — no override needed:
    def compute_moment(self, x: np.ndarray, load: LoadCase) -> np.ndarray: ...
    def compute_shear(self, x: np.ndarray, load: LoadCase) -> np.ndarray: ...
```

### EulerBernoulliBeam

Classical beam theory that assumes plane sections remain perpendicular to the neutral axis. Valid when $L/h > 20$.

```python
from apps.backend.core.models.euler_bernoulli import EulerBernoulliBeam

beam = EulerBernoulliBeam(geometry, material)
x = np.linspace(0, 1.0, 50)
deflection = beam.compute_deflection(x, load)   # shape (50,)
strain     = beam.compute_strain(x, y=-0.05, load=load)
```

Deflection (cantilever, tip load $P$):

$$w(x) = -\frac{P x^2}{6EI}(3L - x)$$

Bending strain at distance $y$ from neutral axis:

$$\varepsilon(x, y) = -y \frac{M(x)}{EI}$$

### TimoshenkoBeam

Includes transverse shear deformation; required when $L/h < 20$, decisive when $L/h < 10$.

```python
from apps.backend.core.models.timoshenko import TimoshenkoBeam

beam = TimoshenkoBeam(geometry, material)
deflection = beam.compute_deflection(x, load)
```

Total deflection is the superposition of bending and shear components:

$$w(x) = w_{bending}(x) + w_{shear}(x) = -\frac{Px^2}{6EI}(3L - x) - \frac{Px}{\kappa GA}$$

Computed properties: `flexural_rigidity` ($EI$), `shear_rigidity` ($\kappa GA$), `shear_parameter` ($\Phi = 12EI / (\kappa GAL^2)$).

---

## Finite Element Models (`apps/backend/core/fem/`)

### TimoshenkoBeamFEM

Ground-truth solver based on 2-node Timoshenko beam elements with two DOFs per node (transverse deflection $w$, rotation $\theta$).

```python
from apps.backend.core.fem.beam_fem import TimoshenkoBeamFEM

fem = TimoshenkoBeamFEM(
    length=1.0, height=0.1, width=0.1,
    elastic_modulus=210e9, poisson_ratio=0.3,
    shear_correction_factor=5/6,
    n_elements=40,          # defaults to min(4 * L/h, 200)
)
result = fem.solve(point_load=1000.0)
```

`solve()` returns a `BeamFEMResult` dataclass with:

| Field | Type | Description |
|---|---|---|
| `x` | `ndarray (n_nodes,)` | Node x-coordinates |
| `deflections` | `ndarray (n_nodes,)` | Transverse displacements |
| `rotations` | `ndarray (n_nodes,)` | Section rotations |
| `tip_deflection` | `float` (property) | Deflection at $x = L$ |

The method `get_deflection_at(x_points)` performs linear interpolation to return deflections at arbitrary positions.

---

## Data Generation (`apps/backend/core/data/`)

### SyntheticDataGenerator

Wraps `TimoshenkoBeamFEM` to produce sensor measurement arrays with configurable noise. The generator is deterministic when `NoiseModel.seed` is fixed.

```python
from apps.backend.core.data.synthetic_generator import (
    SyntheticDataGenerator, SensorConfiguration, NoiseModel, save_dataset
)

sensors = SensorConfiguration(
    displacement_locations=np.linspace(0.2, 1.0, 5),
    strain_locations=np.linspace(0.1, 0.9, 4),
)
noise = NoiseModel(
    relative_noise=True,
    noise_fraction=0.0005,
    seed=42,
)
generator = SyntheticDataGenerator(sensors, noise)
dataset = generator.generate_static_dataset(geometry, material, load)
save_dataset(dataset, Path("outputs/data/dataset_Lh_10.h5"))
```

`generate_static_dataset()` returns a `SyntheticDataset` containing measured displacements, strains, true (noise-free) values, sensor coordinates, noise standard deviations, and beam metadata.

Relevant config keys: `data_generation.n_displacement_sensors`, `data_generation.n_strain_gauges`, `data_generation.noise_fraction`.

---

## Bayesian Inference (`apps/backend/core/bayesian/`)

### BayesianCalibrator (abstract base)

All calibrators share a common interface. The constructor accepts `priors`, `n_samples`, `n_tune`, `n_chains`, and `target_accept`. The primary public method is `calibrate(dataset)`.

```python
from apps.backend.core.bayesian.calibration import (
    EulerBernoulliCalibrator, TimoshenkoCalibrator,
    create_default_priors, create_timoshenko_priors,
)

priors = create_default_priors(config)
calibrator = EulerBernoulliCalibrator(
    priors=priors, n_samples=800, n_tune=400, n_chains=2
)
result = calibrator.calibrate(dataset)
```

`calibrate()` returns a `CalibrationResult` dataclass:

| Field | Description |
|---|---|
| `model_name` | `"EulerBernoulli"` or `"Timoshenko"` |
| `trace` | ArviZ `InferenceData` with full posterior |
| `posterior_summary` | Summary statistics in physical units |
| `waic` | WAIC value (diagnostic only) |
| `marginal_likelihood_estimate` | $\ln p(\mathbf{y} \mid M)$ from bridge sampling |
| `convergence_diagnostics` | Dict of $\hat{R}$ and ESS per parameter |
| `normalization_params` | `NormalizationParams` used during fitting |

`compute_marginal_likelihood()` can be called after `calibrate()` to re-run bridge sampling with different settings.

**Prior factory functions:**
- `create_default_priors(config=None)` — returns priors for $E$ (LogNormal) and $\sigma$ (HalfNormal).
- `create_timoshenko_priors(config=None)` — extends the above with $\nu$ (Normal).

Both accept an optional `config` dict; prior hyperparameters are read from the `priors` section if present.

### BridgeSampler

Estimates $\ln p(\mathbf{y} \mid M)$ via the Meng & Wong (1996) iterative algorithm.

```python
from apps.backend.core.bayesian.bridge_sampling import BridgeSampler

sampler = BridgeSampler(
    trace=pymc_trace,
    log_likelihood_func=log_lik_fn,
    log_prior_func=log_prior_fn,
    seed=42,
)
result = sampler.estimate()
log_ml = result.log_marginal_likelihood   # float
se     = result.standard_error            # float
```

`BridgeSamplingResult` fields: `log_marginal_likelihood`, `standard_error`, `n_iterations`, `converged`, `proposal_fit`.

The module also exposes `compute_bayes_factor_bridge(trace1, trace2, ...)` as a convenience function that instantiates two `BridgeSampler` objects and returns `(log_bayes_factor, combined_se, result1, result2)`.

### BayesianModelSelector

Computes the log Bayes factor from two `CalibrationResult` objects and classifies the evidence.

```python
from apps.backend.core.bayesian.model_selection import BayesianModelSelector

selector = BayesianModelSelector()
comparison = selector.compare_models(result1=eb_result, result2=timo_result)
```

`ModelComparisonResult` fields: `log_bayes_factor`, `bayes_factor`, `evidence_interpretation` (string label), `model1_probability`, `model2_probability`, `waic_difference`, `recommended_model`.

`analyze_aspect_ratio_study(eb_results, timo_results, aspect_ratios)` runs `compare_models` for every ratio, interpolates the $\ln B = 0$ zero-crossing, and returns a study summary dict with a `transition_point` field.

### Normalization

`compute_normalization_params(displacements, strains, E_nominal, method)` returns a `NormalizationParams` dataclass holding `displacement_scale`, `strain_scale`, `E_scale`, `sigma_scale`, and `is_active`. All MCMC sampling occurs in the normalised coordinate system; `CalibrationResult.posterior_summary` always contains physical-unit values.

---

## Visualization (`apps/backend/core/analysis/`)

### BeamVisualization

```python
from apps.backend.core.analysis.visualization import BeamVisualization

viz = BeamVisualization(output_dir=Path("outputs/figures"))
viz.plot_beam_comparison(geometry, material, load)
viz.plot_model_comparison(comparisons, aspect_ratios)
viz.plot_aspect_ratio_study(study_results)
viz.plot_frequency_comparison(freq_results)
viz.plot_prior_posterior_comparison(result)
viz.plot_waic_comparison(eb_result, timo_result)
viz.plot_deflection_error(eb_result, timo_result, dataset)
```

All `plot_*` methods write PNG files to the `output_dir` and return the `matplotlib.Figure` object.

### ResultsReporter

```python
from apps.backend.core.analysis.reporter import ResultsReporter

reporter = ResultsReporter(output_dir=Path("outputs/reports"))
reporter.generate_calibration_report(result)
reporter.generate_comparison_report(comparison, dataset)
reporter.generate_study_summary(study_results, aspect_ratios)
reporter.export_results_json(study_results)
reporter.export_results_csv(comparisons, aspect_ratios)
```

Output files: `study_summary.txt`, `results.json`, `results.csv`, and one `calibration_*.txt` per aspect ratio.

---

## Pipeline Orchestration (`apps/backend/core/pipeline/`)

### PipelineOrchestrator

```python
from apps.backend.core.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(config)  # config is a dict from load_config()
results = orchestrator.run_full_pipeline()
```

Individual stage methods can be called independently when resuming a partial run:

| Method | Description |
|---|---|
| `run_full_pipeline()` | Execute all five stages end-to-end. Returns the study results dict. |
| `run_data_generation()` | Generate and persist HDF5 datasets for all configured aspect ratios. |
| `run_calibration()` | Calibrate EB and Timoshenko for each dataset; requires `run_data_generation()` first. |
| `run_analysis()` | Compute Bayes factors and transition point; requires `run_calibration()` first. |
| `run_frequency_analysis()` | Analytical natural frequency comparison (no MCMC). |
| `generate_report()` | Write all figures and reports; requires `run_analysis()` first. |

---

## Configuration (`apps/backend/core/utils/`)

```python
from apps.backend.core.utils.config import load_config

config = load_config("configs/default_config.yaml")
```

`load_config` reads the YAML file, coerces numeric strings to floats, and validates that the required sections (`beam_parameters`, `material`, `load`) are present. The full configuration schema is documented in [architecture.md](architecture.md).

---

## Type Conventions

All public functions carry PEP 484 type annotations. The package ships a `py.typed` marker (`apps/backend/core/py.typed`) so downstream type checkers can verify call sites. Modern built-in generics (`dict[str, float]`, `list[int]`) are used throughout with `from __future__ import annotations` for Python 3.10 compatibility.
