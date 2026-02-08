# API Reference

## Overview

This document provides detailed API documentation for all public classes and functions in the Digital Twin Bayesian Model Selection framework.

---

## Beam Models (`apps.models`)

### Base Class

#### `BeamModel` (Abstract Base Class)

Abstract base class defining the interface for all beam theory implementations.

```python
from abc import ABC, abstractmethod

class BeamModel(ABC):
    """Abstract base class for beam structural models."""
    
    def __init__(
        self,
        length: float,
        height: float,
        width: float,
        elastic_modulus: float,
        poisson_ratio: float,
    ):
        """
        Initialize beam geometry and material properties.
        
        Args:
            length: Beam length [m]
            height: Beam height (depth) [m]
            width: Beam width [m]
            elastic_modulus: Young's modulus [Pa]
            poisson_ratio: Poisson's ratio [-]
        """
```

**Methods**:

```python
@abstractmethod
def deflection(self, x: float, P: float) -> float:
    """
    Compute vertical deflection at position x.
    
    Args:
        x: Position along beam [m], 0 ≤ x ≤ L
        P: Applied point load at tip [N]
    
    Returns:
        Vertical deflection [m] (positive downward)
    """

@abstractmethod
def strain(self, x: float, y: float, P: float) -> float:
    """
    Compute axial strain at position (x, y).
    
    Args:
        x: Position along beam [m]
        y: Position from neutral axis [m] (positive downward)
        P: Applied point load at tip [N]
    
    Returns:
        Axial strain [-] (dimensionless)
    """

@property
def aspect_ratio(self) -> float:
    """Return beam aspect ratio L/h."""
    return self.length / self.height
```

---

### Euler-Bernoulli Beam

#### `EulerBernoulliBeam`

Implementation of classical Euler-Bernoulli beam theory (slender beams, no shear deformation).

```python
from apps.models.euler_bernoulli import EulerBernoulliBeam

beam = EulerBernoulliBeam(
    length=1.0,
    height=0.1,
    width=0.05,
    elastic_modulus=210e9,
    poisson_ratio=0.3,
)
```

**Key Equations**:

- **Deflection**: $w(x) = \frac{P}{6EI}x^2(3L - x)$
- **Rotation**: $\theta(x) = \frac{P}{2EI}x(2L - x)$
- **Strain**: $\varepsilon(x, y) = -y\frac{d^2w}{dx^2}$

**Usage Example**:

```python
# Tip deflection under 1000 N load
deflection = beam.deflection(x=1.0, P=1000.0)  # Returns: 0.000196 m

# Strain at bottom fiber, mid-span
strain = beam.strain(x=0.5, y=-0.05, P=1000.0)  # Returns: 1.43e-4

# Maximum moment location
x_max_moment = beam.length  # Always at fixed end for cantilever
```

---

### Timoshenko Beam

#### `TimoshenkoBeam`

Implementation of Timoshenko beam theory (includes shear deformation, valid for thick beams).

```python
from apps.models.timoshenko import TimoshenkoBeam

beam = TimoshenkoBeam(
    length=1.0,
    height=0.1,
    width=0.05,
    elastic_modulus=210e9,
    poisson_ratio=0.3,
    shear_correction_factor=5/6,  # Optional, defaults to 5/6
)
```

**Key Equations**:

- **Total Deflection**: $w(x) = w_{bending}(x) + w_{shear}(x)$
- **Bending Component**: $w_b = \frac{P}{6EI}x^2(3L - x)$
- **Shear Component**: $w_s = \frac{P}{\kappa GA}x$
- **Shear Correction Factor**: $\kappa = \frac{5}{6}$ (rectangular section)

**Properties**:

```python
@property
def shear_modulus(self) -> float:
    """Shear modulus G = E / (2(1 + ν))"""
    return self.elastic_modulus / (2 * (1 + self.poisson_ratio))

@property
def shear_area(self) -> float:
    """Effective shear area = κ × A"""
    return self.shear_correction_factor * (self.width * self.height)
```

**Usage Example**:

```python
# Compare EB vs Timoshenko
L, h = 1.0, 0.2  # Short, thick beam
eb_beam = EulerBernoulliBeam(L, h, 0.05, 210e9, 0.3)
timo_beam = TimoshenkoBeam(L, h, 0.05, 210e9, 0.3)

P = 1000.0
eb_tip = eb_beam.deflection(L, P)
timo_tip = timo_beam.deflection(L, P)

shear_contribution = (timo_tip - eb_tip) / timo_tip * 100
print(f"Shear contributes {shear_contribution:.1f}% to total deflection")
# Output: Shear contributes 12.5% to total deflection
```

---

## Finite Element Models (`apps.fem`)

### 1D Timoshenko Beam FEM

#### `TimoshenkoBeamFEM`

High-fidelity 1D finite element solver for Timoshenko beam theory.

```python
from apps.fem.beam_fem import TimoshenkoBeamFEM

fem = TimoshenkoBeamFEM(
    length=1.0,
    width=0.05,
    height=0.1,
    elastic_modulus=210e9,
    poisson_ratio=0.3,
    n_elements=40,
    shear_correction_factor=5/6,
)
```

**Methods**:

```python
def solve(self, force: float) -> BeamFEMResult:
    """
    Solve FEM system for given loading.
    
    Args:
        force: Tip load [N] (positive downward)
    
    Returns:
        BeamFEMResult with deflections, rotations, strains
    """

# Result object
class BeamFEMResult:
    node_coordinates: np.ndarray  # Shape: (n_nodes,)
    deflections: np.ndarray       # Shape: (n_nodes,)
    rotations: np.ndarray         # Shape: (n_nodes,)
    strains: Optional[np.ndarray] # Shape: (n_nodes,)
```

**Usage Example**:

```python
# Solve FEM system
result = fem.solve(force=1000.0)

# Extract results
x = result.node_coordinates
w = result.deflections
theta = result.rotations

# Plot deflection
import matplotlib.pyplot as plt
plt.plot(x, w * 1000)  # Convert to mm
plt.xlabel("Position [m]")
plt.ylabel("Deflection [mm]")
plt.title("FEM Deflection Profile")
plt.grid(True)
plt.show()

# Get tip deflection
tip_deflection = result.deflections[-1]
print(f"Tip deflection: {tip_deflection*1000:.3f} mm")
```

**Element Formulation**:

Each 2-node element has DOFs: `[w1, θ1, w2, θ2]`

Element stiffness matrix:
```
K_e = ∫₀ᴸᵉ B^T C B dx

where:
  B = strain-displacement matrix (bending + shear)
  C = constitutive matrix [EI, κGA]
```

---

### 1D Euler-Bernoulli FEM

#### `EulerBernoulliFEM`

Cubic Hermite element formulation for Euler-Bernoulli beam theory.

```python
from apps.fem.beam_fem import EulerBernoulliFEM

fem = EulerBernoulliFEM(
    length=1.0,
    width=0.05,
    height=0.1,
    elastic_modulus=210e9,
    poisson_ratio=0.3,
    n_elements=20,
)
```

**Differences from Timoshenko FEM**:
- Uses cubic Hermite shape functions (C¹ continuous)
- No shear DOFs (pure bending)
- Typically needs fewer elements for convergence

---

## Data Generation (`apps.data`)

### Synthetic Data Generator

#### `SyntheticDataGenerator`

Generate synthetic measurement data from FEM solutions with configurable noise.

```python
from apps.data.synthetic_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(config)
```

**Methods**:

```python
def generate_static_dataset(
    self,
    aspect_ratio: float,
    force: float,
    output_dir: Path,
) -> Path:
    """
    Generate synthetic dataset for given aspect ratio.
    
    Args:
        aspect_ratio: Beam length-to-height ratio (L/h)
        force: Applied tip load [N]
        output_dir: Directory to save dataset
    
    Returns:
        Path to saved CSV file
    
    Output Format (CSV):
        x,y,displacement,strain,displacement_noisy,strain_noisy
        0.0,0.0,0.0,0.0,0.0001,-0.0002
        0.1,0.05,0.002,0.00015,0.00201,0.000148
        ...
    """

def generate_parametric_study(
    self,
    aspect_ratios: List[float],
    force: float,
    output_dir: Path,
) -> Dict[float, Path]:
    """
    Generate datasets for multiple aspect ratios.
    
    Args:
        aspect_ratios: List of L/h values to generate
        force: Applied tip load [N]
        output_dir: Base directory for output
    
    Returns:
        Dictionary mapping aspect_ratio → dataset_path
    """
```

**Configuration**:

```python
config = {
    "data": {
        "n_sensors": 20,
        "sensor_distribution": "uniform",
        "noise_fraction": 0.0005,  # 0.05% relative noise
        "noise_type": "gaussian",
    },
    "beam_parameters": {
        "length": 1.0,
        "width": 0.05,
    },
}
```

**Usage Example**:

```python
from pathlib import Path

# Generate single dataset
dataset_path = generator.generate_static_dataset(
    aspect_ratio=8.0,
    force=1000.0,
    output_dir=Path("outputs/data"),
)

# Load generated data
import pandas as pd
data = pd.read_csv(dataset_path)

print(data.head())
#        x      y  displacement  strain  displacement_noisy  strain_noisy
# 0  0.000  0.000      0.000000  0.0000            0.000012     -0.000002
# 1  0.053  0.025      0.000012  0.0001            0.000013      0.000099
# ...

# Generate multiple datasets
aspect_ratios = [5, 8, 10, 12, 15, 20, 30, 50]
datasets = generator.generate_parametric_study(
    aspect_ratios=aspect_ratios,
    force=1000.0,
    output_dir=Path("outputs/data"),
)
```

---

## Bayesian Inference (`apps.bayesian`)

### Bayesian Calibrator

#### `BayesianCalibrator`

Perform Bayesian parameter calibration using PyMC.

```python
from apps.bayesian.calibration import BayesianCalibrator

calibrator = BayesianCalibrator(
    beam_model="euler-bernoulli",  # or "timoshenko"
    config=config,
)
```

**Methods**:

```python
def calibrate(
    self,
    data: pd.DataFrame,
    data_type: str = "displacement",
    output_dir: Optional[Path] = None,
) -> az.InferenceData:
    """
    Run MCMC calibration on measurement data.
    
    Args:
        data: DataFrame with columns [x, y, displacement, strain]
        data_type: Type of data to fit — 'displacement' or 'strain'
        output_dir: Directory to save trace (optional)
    
    Returns:
        ArviZ InferenceData object with MCMC samples
    
    Raises:
        SamplingError: If MCMC fails to converge
    """

def get_posterior_summary(
    self,
    trace: az.InferenceData,
) -> pd.DataFrame:
    """
    Get summary statistics of posterior distributions.
    
    Returns DataFrame with columns:
        - mean: Posterior mean
        - sd: Standard deviation
        - hdi_3%: Lower bound of 94% HDI
        - hdi_97%: Upper bound of 94% HDI
        - r_hat: Convergence diagnostic
        - ess_bulk: Effective sample size (bulk)
        - ess_tail: Effective sample size (tail)
    """
```

**PyMC Model Structure**:

```python
with pm.Model() as model:
    # Priors
    E = pm.Normal("E", mu=210e9, sigma=20e9)
    nu = pm.TruncatedNormal("nu", mu=0.3, sigma=0.05, lower=0, upper=0.5)
    
    # Likelihood
    y_pred = compute_predictions(E, nu)  # Forward model
    sigma = pm.HalfNormal("sigma", sigma=0.001)
    y_obs = pm.Normal("y_obs", mu=y_pred, sigma=sigma, observed=data)
    
    # Sample
    trace = pm.sample(
        draws=800,
        tune=400,
        chains=2,
        target_accept=0.95,
    )
```

**Strain Calibration**:

The calibrator also supports strain-based calibration. The strain forward model computes surface strain at sensor locations:

$$\varepsilon(x) = -\frac{h}{2} \cdot \frac{P(L - x)}{EI}$$

This formula is shared by both Euler-Bernoulli and Timoshenko theories, since shear deformation does not affect bending curvature (and hence axial strain). Usage:

```python
# Calibrate on strain data instead of displacement
eb_trace_strain = eb_calibrator.calibrate(data, data_type="strain")
timo_trace_strain = timo_calibrator.calibrate(data, data_type="strain")
```

**Usage Example**:

```python
# Load data
data = pd.read_csv("outputs/data/Lh_8.0.csv")

# Calibrate Euler-Bernoulli model
eb_calibrator = BayesianCalibrator("euler-bernoulli", config)
eb_trace = eb_calibrator.calibrate(data)

# Calibrate Timoshenko model
timo_calibrator = BayesianCalibrator("timoshenko", config)
timo_trace = timo_calibrator.calibrate(data)

# Get posterior summaries
eb_summary = eb_calibrator.get_posterior_summary(eb_trace)
print(eb_summary)
#           mean         sd      hdi_3%     hdi_97%  r_hat  ess_bulk
# E     2.09e+11  1.5e+09   2.06e+11   2.12e+11   1.000      1523
# nu    3.01e-01  4.2e-03   2.93e-01   3.09e-01   1.001      1487

# Check convergence
assert (eb_summary["r_hat"] < 1.01).all(), "MCMC did not converge!"
```

---

### Model Selector

#### `ModelSelector`

Compare models using Bayesian information criteria.

```python
from apps.bayesian.model_selection import ModelSelector

selector = ModelSelector()
```

**Methods**:

```python
def compute_waic(
    self,
    trace: az.InferenceData,
) -> az.stats.ELPDData:
    """
    Compute WAIC (Widely Applicable Information Criterion).
    
    Returns object with attributes:
        - waic: WAIC value
        - waic_se: Standard error
        - p_waic: Effective number of parameters
        - warning: True if WAIC may be unreliable
    """

def compute_loo(
    self,
    trace: az.InferenceData,
) -> az.stats.ELPDData:
    """
    Compute LOO-CV (Leave-One-Out Cross-Validation).
    
    Uses Pareto-Smoothed Importance Sampling (PSIS).
    
    Returns object with attributes:
        - loo: LOO-CV value
        - loo_se: Standard error
        - p_loo: Effective number of parameters
        - pareto_k: Diagnostic for each observation
        - warning: True if Pareto k > 0.7 for any point
    """

def compare_models(
    self,
    traces: Dict[str, az.InferenceData],
) -> pd.DataFrame:
    """
    Compare multiple models using WAIC/LOO.
    
    Args:
        traces: Dictionary mapping model_name → trace
    
    Returns:
        Comparison DataFrame sorted by WAIC (best first):
            - rank: Model rank (0 = best)
            - waic: WAIC value
            - waic_se: Standard error
            - dw AIC: Difference from best model
            - weight: Model weight (probability)
            - dse: Standard error of difference
    """

def compute_bayes_factor(
    self,
    trace1: az.InferenceData,
    trace2: az.InferenceData,
) -> float:
    """
    Compute log Bayes factor: log(p(y|M1) / p(y|M2)).
    
    Approximated via: (WAIC_2 - WAIC_1) / 2
    
    Interpretation:
        log(BF) < -5: Strong evidence for M1
        -5 < log(BF) < 0: Moderate evidence for M1
        0 < log(BF) < 5: Moderate evidence for M2
        log(BF) > 5: Strong evidence for M2
    
    Returns:
        Log Bayes factor (positive favors M1)
    """
```

**Usage Example**:

```python
# Compare EB vs Timoshenko
traces = {
    "Euler-Bernoulli": eb_trace,
    "Timoshenko": timo_trace,
}

comparison = selector.compare_models(traces)
print(comparison)
#                    rank  waic  waic_se  dwaic  weight    dse
# Timoshenko            0 -142.3     8.2    0.0   0.998    0.0
# Euler-Bernoulli       1 -127.5     7.8   14.8   0.002   3.2

# Compute Bayes factor
log_bf = selector.compute_bayes_factor(timo_trace, eb_trace)
print(f"log(BF) = {log_bf:.2f}")  # Negative favors Timoshenko
# Output: log(BF) = -7.40

# Interpret
if log_bf < -5:
    print("Strong evidence for Timoshenko")
elif log_bf < 0:
    print("Moderate evidence for Timoshenko")
elif log_bf < 5:
    print("Moderate evidence for Euler-Bernoulli")
else:
    print("Strong evidence for Euler-Bernoulli")
```

---

## Visualization (`apps.analysis`)

### Visualization Functions

```python
from apps.analysis.visualization import (
    plot_posterior_distributions,
    plot_model_comparison,
    plot_bayes_factors,
    plot_summary_report,
)
```

#### `plot_posterior_distributions`

```python
def plot_posterior_distributions(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot posterior distributions with priors and credible intervals.
    
    Args:
        trace: MCMC samples
        var_names: Variables to plot (default: all)
        figsize: Figure size
        output_path: Save path (if None, display only)
    
    Returns:
        Matplotlib figure object
    """
```

#### `plot_model_comparison`

```python
def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "waic",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot model comparison with error bars.
    
    Args:
        comparison_df: Output from ModelSelector.compare_models()
        metric: "waic" or "loo"
        output_path: Save path
    
    Returns:
        Figure with comparison plot
    """
```

#### `plot_bayes_factors`

```python
def plot_bayes_factors(
    aspect_ratios: List[float],
    log_bayes_factors: List[float],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot log Bayes factors vs aspect ratio.
    
    Shows transition point where model preference switches.
    
    Args:
        aspect_ratios: L/h values
        log_bayes_factors: Corresponding log BF values
        output_path: Save path
    
    Returns:
        Figure with Bayes factor trend plot
    """
```

---

## Pipeline Orchestration (`apps.pipeline`)

### Pipeline Orchestrator

#### `PipelineOrchestrator`

Coordinate full analysis workflow.

```python
from apps.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator(config, output_dir)
```

**Methods**:

```python
def run(self, stage: str = "all") -> None:
    """
    Execute pipeline stages.
    
    Args:
        stage: One of ["all", "data", "calibration", "analysis", "report"]
    
    Stages:
        1. data: Generate synthetic datasets
        2. calibration: Run Bayesian inference
        3. analysis: Model selection and comparison
        4. report: Generate summary outputs
    """

def run_data_generation(self) -> Dict[float, Path]:
    """Generate datasets for all aspect ratios."""

def run_calibration(self) -> Dict[float, Dict[str, az.InferenceData]]:
    """Calibrate EB and Timoshenko models for each dataset."""

def run_model_selection(self) -> pd.DataFrame:
    """Compare models and compute Bayes factors."""

def run_reporting(self) -> None:
    """Generate plots and summary reports."""
```

**Usage Example**:

```python
from pathlib import Path

config = load_config("configs/default_config.yaml")
output_dir = Path("outputs")

# Create orchestrator
orchestrator = PipelineOrchestrator(config, output_dir)

# Run full pipeline
orchestrator.run(stage="all")

# Or run individual stages
orchestrator.run(stage="data")
orchestrator.run(stage="calibration")
orchestrator.run(stage="analysis")
orchestrator.run(stage="report")
```

---

## Configuration (`apps.utils`)

### Config Loader

```python
from apps.utils.config import load_config, validate_config

# Load YAML configuration
config = load_config("configs/default_config.yaml")

# Validate configuration
validate_config(config)  # Raises ValueError if invalid
```

**Configuration Schema**:

See [`ARCHITECTURE.md#configuration-schema`](ARCHITECTURE.md#configuration-schema) for complete schema.

---

## Error Handling

### Custom Exceptions

```python
from apps.utils.exceptions import (
    CalibrationError,
    ConvergenceError,
    InvalidConfigError,
    DataGenerationError,
)

try:
    trace = calibrator.calibrate(data)
except ConvergenceError as e:
    logger.error(f"MCMC failed to converge: {e}")
    # Retry with more tuning steps
except CalibrationError as e:
    logger.error(f"Calibration failed: {e}")
    # Handle error
```

---

## Type Hints

All public functions include complete type annotations:

```python
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az

def example_function(
    data: pd.DataFrame,
    param: float,
    optional_param: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Fully typed function signature."""
    pass
```

---

## Further Reading

- [ARCHITECTURE.md](../ARCHITECTURE.md): System design and architecture
- [CONTRIBUTING.md](../CONTRIBUTING.md): Development guidelines
- [docs/DEVELOPMENT.md](DEVELOPMENT.md): Development workflow
- [PyMC Documentation](https://www.pymc.io/): Bayesian inference
- [ArviZ Documentation](https://python.arviz.org/): MCMC diagnostics
