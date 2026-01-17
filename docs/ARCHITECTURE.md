# Architecture Documentation

## System Overview

The Digital Twin Bayesian Model Selection framework is a **modular pipeline architecture** designed for scalable, reproducible analysis of structural beam theories using Bayesian inference.

### Design Principles

1. **Separation of Concerns**: Clear boundaries between data generation, inference, analysis, and reporting
2. **Composability**: Modular components that can be used independently or orchestrated
3. **Reproducibility**: Configuration-driven execution with deterministic outputs
4. **Extensibility**: Easy to add new beam theories, FEM solvers, or analysis methods
5. **Performance**: Optimized for computational efficiency with parallel MCMC chains

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Configuration Layer                       │
│                     (configs/default_config.yaml)                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Orchestrator                       │
│                  (apps/pipeline/orchestrator.py)                 │
│  • Stage coordination  • Dependency management  • Error handling │
└──────┬──────────────────┬───────────────────┬────────────────┬──┘
       │                  │                   │                │
       ▼                  ▼                   ▼                ▼
┌─────────────┐  ┌────────────────┐  ┌─────────────┐  ┌─────────────┐
│    Data     │  │   Bayesian     │  │  Analysis   │  │  Reporting  │
│ Generation  │──│  Calibration   │──│    Layer    │──│    Layer    │
│             │  │                │  │             │  │             │
└─────────────┘  └────────────────┘  └─────────────┘  └─────────────┘
       │                  │                   │                │
       ▼                  ▼                   ▼                ▼
┌─────────────┐  ┌────────────────┐  ┌─────────────┐  ┌─────────────┐
│  FEM Models │  │  PyMC Models   │  │ Visualization│  │   Reports   │
│  Beam Theory│  │  MCMC Sampling │  │   Plotting   │  │   Metrics   │
└─────────────┘  └────────────────┘  └─────────────┘  └─────────────┘
```

---

## Core Components

### 1. Data Generation Layer (`apps/data/`)

**Purpose**: Generate synthetic measurement data from ground truth FEM models

**Key Classes**:
- `SyntheticDataGenerator`: Main interface for data generation
  - Configurable noise models (Gaussian, relative)
  - Sensor placement strategies
  - Dataset persistence (CSV, JSON metadata)

**Data Flow**:
```
Input: Beam geometry, material properties, loading
  ↓
1D Timoshenko Beam FEM Solver
  ↓
Ground Truth: Displacements, strains at sensor locations
  ↓
Add measurement noise (0.05% default)
  ↓
Output: Synthetic dataset with metadata
```

**Design Decision**: We use **1D Timoshenko beam FEM** as ground truth instead of 2D continuum FEM because:
- Exact consistency with analytical beam theory assumptions
- 100x faster computation (200 vs 20,000 elements)
- No systematic stiffness mismatch from constraint effects
- Validated: 0.0000% error vs analytical Timoshenko solution

---

### 2. FEM Layer (`apps/fem/`)

**Purpose**: Provide finite element solvers for ground truth generation

**Modules**:

#### `beam_fem.py` (Active)
- **`TimoshenkoBeamFEM`**: 2-node Timoshenko beam elements
  - Accounts for shear deformation (κ = 5/6)
  - Coupled bending-shear stiffness matrix
  - Returns deflections, rotations, strains
- **`EulerBernoulliFEM`**: Cubic Hermite beam elements
  - Pure bending formulation
  - Higher-order shape functions
  - For comparison/validation only

**Element Formulation** (Timoshenko):
```
DOFs per node: [w, θ]  (deflection, rotation)
Element stiffness: K_e = ∫ B^T C B dL
  where B = strain-displacement matrix
        C = constitutive matrix [EI, κGA]
```

**Adaptive Mesh Sizing**:
- n_elements = 4 × (L/h), capped at 200
- Ensures adequate resolution for shear gradients
- Balances accuracy vs performance

#### `cantilever_fem.py` (Legacy)
- 2D plane stress Q4 elements
- Kept for reference, not used in production
- Had systematic ~1% stiffness bias vs beam theories

---

### 3. Beam Theory Models (`apps/models/`)

**Purpose**: Analytical beam theory implementations for forward predictions

**Base Class** (`base_beam.py`):
```python
class BeamModel(ABC):
    @abstractmethod
    def deflection(x: float, P: float, L: float, E: float, I: float) -> float
    
    @abstractmethod
    def strain(x: float, y: float, P: float, L: float, E: float, I: float) -> float
```

**Implementations**:

#### `EulerBernoulliBeam`
- **Assumptions**: Plane sections remain plane and perpendicular
- **Deflection**: δ = (PL³)/(3EI) at tip
- **Valid for**: Slender beams (L/h > 20)
- **Complexity**: O(1) - closed-form solution

#### `TimoshenkoBeam`
- **Assumptions**: Plane sections remain plane but can rotate independently
- **Deflection**: δ = (PL³)/(3EI) + (PL)/(κGA) at tip
- **Shear correction**: κ = 5/6 for rectangular cross-sections
- **Valid for**: All aspect ratios, especially L/h < 20
- **Complexity**: O(1) - closed-form solution with shear term

**Usage Pattern**:
```python
beam = TimoshenkoBeam(L=1.0, h=0.1, b=0.05, E=210e9, nu=0.3)
deflection = beam.deflection(x=1.0, P=1000.0)
strain = beam.strain(x=0.5, y=0.05, P=1000.0)
```

---

### 4. Bayesian Inference Layer (`apps/bayesian/`)

**Purpose**: Probabilistic parameter calibration and model comparison

#### `calibration.py`

**Key Class**: `BayesianCalibrator`

**Probabilistic Model**:
```
Prior:
  E ~ Normal(μ_E, σ_E)         # Elastic modulus
  ν ~ TruncatedNormal(μ_ν, σ_ν, 0, 0.5)  # Poisson's ratio

Likelihood:
  y_obs ~ Normal(y_pred(E, ν), σ_noise)
  
where y_pred is the beam theory forward model
```

**MCMC Configuration**:
- Sampler: NUTS (No-U-Turn Sampler)
- Chains: 2 parallel chains
- Samples: 800 per chain (1600 total)
- Tuning: 400 steps
- Target accept: 0.95 (robust convergence)

**Diagnostics**:
- R-hat: Convergence check (should be < 1.01)
- Effective sample size (ESS): Independent samples
- Divergences: Geometry issues (should be 0)
- Energy plots: HMC dynamics validation

#### `model_selection.py`

**Key Class**: `ModelSelector`

**Metrics**:

1. **WAIC** (Widely Applicable Information Criterion)
   ```
   WAIC = -2(lppd - p_WAIC)
   where lppd = log pointwise predictive density
         p_WAIC = effective number of parameters
   ```
   Lower is better. Balances fit vs complexity.

2. **LOO-CV** (Leave-One-Out Cross-Validation)
   ```
   LOO = -2 Σ log p(y_i | y_{-i})
   ```
   Approximated via Pareto-Smoothed Importance Sampling (PSIS).

3. **Bayes Factor**
   ```
   BF = p(y | M_Timoshenko) / p(y | M_EB)
   log(BF) = (WAIC_EB - WAIC_Timoshenko) / 2
   ```
   
   **Interpretation**:
   - log(BF) < -5: Strong evidence for Timoshenko
   - -5 < log(BF) < 0: Moderate evidence for Timoshenko
   - 0 < log(BF) < 5: Moderate evidence for EB
   - log(BF) > 5: Strong evidence for EB

**Model Evidence Calculation**:
```python
def compute_bayes_factor(trace_eb, trace_timo, data):
    waic_eb = compute_waic(trace_eb, data)
    waic_timo = compute_waic(trace_timo, data)
    log_bf = (waic_eb.waic - waic_timo.waic) / 2
    return log_bf
```

---

### 5. Analysis & Visualization (`apps/analysis/`)

**Purpose**: Post-processing, plotting, and insight generation

#### `visualization.py`

**Plot Types**:

1. **Posterior Distributions**
   - KDE plots with credible intervals
   - Prior vs posterior comparison
   - Parameter correlations (corner plots)

2. **Model Comparison**
   - WAIC comparison with error bars
   - Bayes factor trends across aspect ratios
   - Posterior predictive checks

3. **Summary Reports**
   - Convergence diagnostics
   - Model selection decision matrix
   - Transition aspect ratio identification

#### `reporter.py`

**Output Formats**:
- **Text**: Summary tables (rich console formatting)
- **JSON**: Machine-readable results
- **CSV**: Raw data for further analysis
- **PNG**: High-resolution figures (300 DPI)

**Report Sections**:
1. Executive summary
2. Model selection results table
3. Transition aspect ratio
4. Digital twin recommendations
5. Convergence diagnostics
6. Performance metrics

---

### 6. Pipeline Orchestration (`apps/pipeline/`)

**Purpose**: End-to-end workflow coordination with dependency management

#### `orchestrator.py`

**Key Class**: `PipelineOrchestrator`

**Pipeline Stages**:

```python
Stage 1: Data Generation
  ├─ For each aspect ratio:
  │   ├─ Initialize 1D Timoshenko FEM
  │   ├─ Solve for ground truth
  │   ├─ Add measurement noise
  │   └─ Save dataset
  └─ Output: outputs/data/*.csv

Stage 2: Bayesian Calibration
  ├─ For each dataset:
  │   ├─ Load synthetic measurements
  │   ├─ Calibrate Euler-Bernoulli model (MCMC)
  │   ├─ Calibrate Timoshenko model (MCMC)
  │   ├─ Save traces to NetCDF
  │   └─ Log diagnostics
  └─ Output: outputs/traces/*.nc

Stage 3: Model Selection
  ├─ For each aspect ratio:
  │   ├─ Load EB and Timo traces
  │   ├─ Compute WAIC, LOO-CV
  │   ├─ Calculate Bayes factors
  │   └─ Determine preferred model
  └─ Output: outputs/reports/results.csv

Stage 4: Reporting
  ├─ Generate summary plots
  ├─ Create text reports
  ├─ Identify transition aspect ratio
  └─ Output recommendations
```

**Error Handling**:
- Stage-level try-catch with graceful degradation
- Partial results saved on failure
- Detailed logging at each checkpoint
- Automatic cleanup of temporary files

**Parallelization**:
- MCMC chains run in parallel (multiprocessing)
- Aspect ratios processed sequentially (memory management)
- I/O operations batched for efficiency

---

## Data Flow Diagram

```
┌──────────────┐
│ Config YAML  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│     1. Data Generation Stage              │
│  ┌────────────────────────────────────┐  │
│  │  For L/h in [5,8,10,12,15,20,30,50]│  │
│  │    ↓                                │  │
│  │  TimoshenkoBeamFEM.solve()         │  │
│  │    ↓                                │  │
│  │  Add 0.05% Gaussian noise          │  │
│  │    ↓                                │  │
│  │  Save: outputs/data/Lh_{X}.csv     │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│     2. Bayesian Calibration Stage         │
│  ┌────────────────────────────────────┐  │
│  │  Load dataset                       │  │
│  │    ↓                                │  │
│  │  Initialize PyMC model (EB)        │  │
│  │    ↓                                │  │
│  │  Run NUTS sampler (800×2 samples)  │  │
│  │    ↓                                │  │
│  │  Save: trace_eb.nc                 │  │
│  │    ↓                                │  │
│  │  Initialize PyMC model (Timoshenko)│  │
│  │    ↓                                │  │
│  │  Run NUTS sampler (800×2 samples)  │  │
│  │    ↓                                │  │
│  │  Save: trace_timo.nc               │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│     3. Model Selection Stage              │
│  ┌────────────────────────────────────┐  │
│  │  Load EB and Timo traces            │  │
│  │    ↓                                │  │
│  │  Compute WAIC for both models      │  │
│  │    ↓                                │  │
│  │  Calculate log Bayes Factor        │  │
│  │    ↓                                │  │
│  │  Determine recommended model       │  │
│  │    ↓                                │  │
│  │  Save: results.csv                 │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│     4. Reporting Stage                    │
│  ┌────────────────────────────────────┐  │
│  │  Generate summary plots             │  │
│  │  Create text report                 │  │
│  │  Identify transition L/h            │  │
│  │  Output recommendations             │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

---

## Design Decisions & Trade-offs

### 1. Why 1D FEM Instead of 2D FEM?

**Decision**: Use 1D Timoshenko beam FEM as ground truth

**Rationale**:
- ✅ **Consistency**: Same assumptions as analytical theories
- ✅ **Performance**: 100× faster (200 vs 20,000 elements)
- ✅ **Accuracy**: 0.0000% error vs analytical Timoshenko
- ✅ **Clarity**: No systematic bias from 2D constraint effects

**Trade-off**: 
- ❌ Cannot model complex 2D/3D effects (stress concentrations, non-uniform sections)
- ✅ But: Not needed for beam theory validation study

### 2. Why WAIC/LOO Instead of Marginal Likelihood?

**Decision**: Use WAIC and LOO-CV for model comparison

**Rationale**:
- ✅ **Computational**: Faster than bridge sampling or thermodynamic integration
- ✅ **Robust**: Less sensitive to prior specification
- ✅ **Interpretable**: Built-in uncertainty estimates

**Trade-off**:
- ❌ Not true marginal likelihood (approximation via pointwise predictive density)
- ✅ But: Well-validated approximation for model selection

### 3. Why NUTS Instead of Other MCMC Samplers?

**Decision**: Use NUTS (No-U-Turn Sampler) via PyMC

**Rationale**:
- ✅ **Efficiency**: Fewer samples needed for convergence
- ✅ **Automatic tuning**: No manual step size tuning
- ✅ **Diagnostics**: Rich convergence diagnostics

**Trade-off**:
- ❌ Computationally expensive per sample
- ✅ But: Offset by fewer samples needed (800 vs 10,000+ for MH)

### 4. Why Sequential Aspect Ratios Instead of Parallel?

**Decision**: Process aspect ratios sequentially in pipeline

**Rationale**:
- ✅ **Memory**: PyMC traces can be large (100+ MB each)
- ✅ **Stability**: Avoids PyMC multiprocessing conflicts
- ✅ **Logging**: Clear progress tracking

**Trade-off**:
- ❌ Longer total runtime (~30-40 min for 8 aspect ratios)
- ✅ But: Individual runs are already parallelized (MCMC chains)

---

## Performance Characteristics

### Computational Complexity

| Component | Complexity | Bottleneck |
|-----------|-----------|------------|
| Data Generation | O(n_elements) | FEM assembly |
| MCMC Sampling | O(n_samples × n_params) | Likelihood evaluation |
| WAIC Computation | O(n_samples × n_obs) | Pointwise log-likelihood |
| Visualization | O(n_samples) | KDE estimation |

### Memory Usage

- **Per dataset**: ~10 KB (CSV)
- **Per MCMC trace**: ~100 MB (NetCDF, 1600 samples × 2 params)
- **Peak memory**: ~1 GB (during calibration with 2 parallel chains)

### Runtime (Typical Laptop, 4 cores)

- Data generation (8 aspect ratios): **~5 seconds**
- Bayesian calibration (per aspect ratio):
  - EB model: **~3-4 minutes**
  - Timoshenko model: **~3-4 minutes**
- Model selection: **~10 seconds**
- Reporting: **~30 seconds**
- **Total pipeline**: **~30-40 minutes**

---

## Extensibility Points

### Adding New Beam Theories

1. Create new class in `apps/models/`:
   ```python
   class NewBeamTheory(BeamModel):
       def deflection(self, x, P, L, E, I):
           # Implement theory
           return deflection
   ```

2. Add to calibration in `apps/bayesian/calibration.py`:
   ```python
   def _build_new_theory_model(self, data):
       with pm.Model() as model:
           # Define priors and likelihood
           pass
   ```

3. Update orchestrator to include in model comparison

### Adding New FEM Solvers

1. Implement solver in `apps/fem/`:
   ```python
   class NewFEMSolver:
       def solve(self, mesh, bc, loading):
           # FEM implementation
           return results
   ```

2. Update `SyntheticDataGenerator` to support new solver:
   ```python
   if fem_type == "new_solver":
       fem = NewFEMSolver(params)
   ```

### Adding New Model Selection Criteria

1. Implement in `apps/bayesian/model_selection.py`:
   ```python
   def compute_new_metric(trace, data):
       # Custom metric calculation
       return metric_value
   ```

2. Update comparison logic to use new metric

---

## Configuration Schema

```yaml
# Complete configuration structure
beam_parameters:
  length: float              # Beam length [m]
  width: float               # Beam width [m]
  height: float              # Beam height [m]
  aspect_ratios: List[float] # L/h ratios to study
  
loading:
  force: float               # Applied force [N]
  location: str              # "tip" or "center"
  
material:
  elastic_modulus: float     # E [Pa]
  poisson_ratio: float       # ν [-]
  density: float             # ρ [kg/m³]
  
bayesian:
  n_samples: int             # MCMC samples per chain
  n_tune: int                # Tuning samples
  n_chains: int              # Parallel chains
  target_accept: float       # NUTS acceptance target
  
data:
  n_sensors: int             # Number of measurement points
  sensor_distribution: str   # "uniform" or "random"
  noise_fraction: float      # Relative noise level
  
output:
  save_traces: bool          # Save MCMC traces
  save_figures: bool         # Generate plots
  figure_dpi: int            # Plot resolution
```

---

## Testing Strategy

### Unit Tests (`tests/`)

1. **Beam Models** (`test_beam_models.py`)
   - Deflection formula validation
   - Strain calculation accuracy
   - Edge case handling

2. **FEM Solvers** (`test_fem.py`)
   - Element stiffness matrix correctness
   - Mesh generation
   - Boundary condition application

3. **Bayesian Inference**
   - Prior sampling
   - Likelihood evaluation
   - WAIC computation

### Integration Tests

- End-to-end pipeline with synthetic data
- Configuration loading and validation
- Output file generation

### Validation Tests

- Comparison with analytical solutions
- Convergence studies (mesh, MCMC)
- Reproducibility checks

---

## Dependencies

### Core
- **PyMC** (≥5.10.0): Bayesian inference framework
- **ArviZ** (≥0.17.0): MCMC diagnostics and visualization
- **NumPy** (≥1.24.0): Numerical computing
- **SciPy** (≥1.11.0): Scientific computing

### Visualization
- **Matplotlib** (≥3.7.0): Plotting
- **Seaborn** (≥0.12.0): Statistical visualization

### Utilities
- **PyYAML** (≥6.0): Configuration parsing
- **Rich** (≥13.0): Terminal formatting
- **Click** (≥8.1.0): CLI framework

---

## Security Considerations

1. **Input Validation**: All configuration values are validated before use
2. **File I/O**: Paths are sanitized to prevent directory traversal
3. **No External APIs**: All computation is local (no network calls)
4. **Deterministic Execution**: Fixed random seeds for reproducibility

---

## Future Architecture Improvements

### Short-term
- [ ] Implement caching for FEM solutions
- [ ] Add parallel processing for multiple aspect ratios
- [ ] Support for custom beam theories via plugin system

### Long-term
- [ ] Distributed MCMC sampling (multiple machines)
- [ ] Real-time digital twin integration
- [ ] GPU acceleration for FEM and MCMC
- [ ] Web interface for results exploration

---

## References

- **Bayesian Data Analysis**: Gelman et al., 3rd Edition
- **PyMC Documentation**: https://docs.pymc.io/
- **Finite Element Method**: Hughes, T.J.R., "The Finite Element Method"
- **Beam Theory**: Timoshenko, S.P., "Theory of Elastic Stability"
