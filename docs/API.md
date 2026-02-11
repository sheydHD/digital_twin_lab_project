# API reference

## beam models (`apps.models`)

### base class

`BeamModel` — abstract base class for all beam theory implementations.

```python
class BeamModel(ABC):
    def __init__(self, length, height, width, elastic_modulus, poisson_ratio): ...

    @abstractmethod
    def deflection(self, x: float, P: float) -> float: ...

    @abstractmethod
    def strain(self, x: float, y: float, P: float) -> float: ...

    @property
    def aspect_ratio(self) -> float: ...
```

### Euler-Bernoulli beam

`EulerBernoulliBeam` — classical beam theory, no shear deformation.

```python
from apps.models.euler_bernoulli import EulerBernoulliBeam

beam = EulerBernoulliBeam(length=1.0, height=0.1, width=0.05,
                          elastic_modulus=210e9, poisson_ratio=0.3)

deflection = beam.deflection(x=1.0, P=1000.0)
strain = beam.strain(x=0.5, y=-0.05, P=1000.0)
```

Deflection: $w(x) = \frac{P}{6EI}x^2(3L - x)$

Strain: $\varepsilon(x, y) = -y\frac{d^2w}{dx^2}$

### Timoshenko beam

`TimoshenkoBeam` — includes shear deformation, valid for thick beams.

```python
from apps.models.timoshenko import TimoshenkoBeam

beam = TimoshenkoBeam(length=1.0, height=0.1, width=0.05,
                      elastic_modulus=210e9, poisson_ratio=0.3,
                      shear_correction_factor=5/6)

deflection = beam.deflection(x=1.0, P=1000.0)
```

Total deflection: $w(x) = w_{bending}(x) + w_{shear}(x)$

Shear component: $w_s = \frac{P}{\kappa GA}x$

Properties: `shear_modulus`, `shear_area`.

## finite element models (`apps.fem`)

### 1D Timoshenko beam FEM

`TimoshenkoBeamFEM` — ground truth solver. 2-node Timoshenko beam elements with coupled bending-shear stiffness.

```python
from apps.fem.beam_fem import TimoshenkoBeamFEM

fem = TimoshenkoBeamFEM(length=1.0, width=0.05, height=0.1,
                        elastic_modulus=210e9, poisson_ratio=0.3,
                        n_elements=40, shear_correction_factor=5/6)

result = fem.solve(force=1000.0)
```

Returns `BeamFEMResult` with:
- `node_coordinates` — shape (n_nodes,)
- `deflections` — shape (n_nodes,)
- `rotations` — shape (n_nodes,)
- `strains` — shape (n_nodes,) (optional)

DOFs per node: deflection w, rotation theta.

## data generation (`apps.data`)

### synthetic data generator

`SyntheticDataGenerator` — generates synthetic measurement data from FEM solutions with configurable noise.

```python
from apps.data.synthetic_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(config)

# single dataset
path = generator.generate_static_dataset(aspect_ratio=8.0, force=1000.0,
                                         output_dir=Path("outputs/data"))

# multiple datasets
datasets = generator.generate_parametric_study(
    aspect_ratios=[5, 8, 10, 20], force=1000.0,
    output_dir=Path("outputs/data"))
```

Config keys: `data.n_sensors`, `data.noise_fraction`, `data.sensor_distribution`.

## bayesian inference (`apps.bayesian`)

### bayesian calibrator

`BayesianCalibrator` — MCMC calibration using PyMC.

```python
from apps.bayesian.calibration import BayesianCalibrator

calibrator = BayesianCalibrator(beam_model="euler-bernoulli", config=config)
trace = calibrator.calibrate(data, data_type="displacement")
summary = calibrator.get_posterior_summary(trace)
log_ml = calibrator.compute_marginal_likelihood()
```

Methods:
- `calibrate(data, data_type, output_dir)` — run MCMC, returns `az.InferenceData`
- `get_posterior_summary(trace)` — returns DataFrame with mean, sd, HDI, R-hat, ESS
- `compute_marginal_likelihood()` — estimates log marginal likelihood via bridge sampling

Subclasses: `EulerBernoulliCalibrator`, `TimoshenkoCalibrator`.

PyMC model structure:
```python
with pm.Model() as model:
    E = pm.Normal("E", mu=1.0, sigma=0.05)       # normalized
    sigma = pm.HalfNormal("sigma", sigma=1.0)     # normalized
    y_pred = forward_model(E)                      # beam theory
    y_obs = pm.Normal("y_obs", mu=y_pred, sigma=sigma, observed=data)
    trace = pm.sample(draws=800, tune=400, chains=2, target_accept=0.95)
```

Strain calibration uses the same interface with `data_type="strain"`.

### bridge sampler

`BridgeSampler` — marginal likelihood estimation via the Meng & Wong (1996) algorithm.

```python
from apps.bayesian.bridge_sampling import BridgeSampler

sampler = BridgeSampler(posterior_samples, log_likelihood_fn, log_prior_fn)
log_ml = sampler.compute_log_marginal_likelihood()
```

Also: `compute_bayes_factor_bridge(calibrator_1, calibrator_2)` convenience function.

### model selector

`BayesianModelSelector` — model comparison using bridge sampling marginal likelihoods.

```python
from apps.bayesian.model_selection import BayesianModelSelector

selector = BayesianModelSelector()

# compute WAIC (diagnostic)
waic = selector.compute_waic(trace)

# compare models
result = selector.compare_models(
    traces={"EB": eb_trace, "Timoshenko": timo_trace},
    marginal_likelihoods={"EB": log_ml_eb, "Timo": log_ml_timo})
```

Methods:
- `compute_waic(trace)` — returns WAIC value (diagnostic)
- `compare_models(traces, marginal_likelihoods)` — computes log Bayes factor
- `analyze_aspect_ratio_study(results)` — full study analysis with transition point

### normalization

`compute_normalization_params(data, config)` — computes E_scale and displacement_scale for MCMC stability.

`NormalizationParams` — dataclass holding scale factors.

## visualization (`apps.analysis`)

### visualization functions

```python
from apps.analysis.visualization import (
    plot_model_comparison,
    plot_bayes_factors,
)
```

`plot_model_comparison(comparison_df, output_path)` — model comparison with error bars.

`plot_bayes_factors(aspect_ratios, log_bayes_factors, output_path)` — log Bayes factors vs aspect ratio with transition point.

### reporter

```python
from apps.analysis.reporter import AnalysisReporter

reporter = AnalysisReporter(output_dir)
reporter.generate_study_report(results)
```

Outputs: text summary, JSON, CSV.

## pipeline orchestration (`apps.pipeline`)

### pipeline orchestrator

`PipelineOrchestrator` — coordinates full analysis workflow.

```python
from apps.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(config, output_dir)
orchestrator.run(stage="all")  # or "data", "calibration", "analysis", "report"
```

Methods:
- `run(stage)` — execute pipeline stages
- `run_data_generation()` — generate datasets
- `run_calibration()` — calibrate both models for each dataset
- `run_model_selection()` — compare models, compute Bayes factors
- `run_reporting()` — generate plots and reports

## configuration (`apps.utils`)

```python
from apps.utils.config import load_config, validate_config

config = load_config("configs/default_config.yaml")
validate_config(config)
```

See [architecture](ARCHITECTURE.md#configuration-schema) for the full schema.

## type hints

All public functions include type annotations. The codebase uses `typing` module types: `Dict`, `List`, `Optional`, `Tuple`, `Union`, and `Path` from `pathlib`.
