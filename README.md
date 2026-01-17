# Bayesian Model Selection for Beam Theory in Structural Health Monitoring Digital Twins

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements **Bayesian model selection** to systematically compare **Euler-Bernoulli** and **Timoshenko beam theories** using simulated sensor data from a cantilever structure. The goal is to determine when each theory is justified and develop practical guidelines for model selection during digital twin initialization and recalibration.

### Context

Long-span structures such as bridges, building frames, and elevated infrastructure are increasingly monitored by digital twins that combine physics-based models with real-time sensor data for structural health monitoring and predictive maintenance.

The choice of beam theory is fundamental to digital twin predictions:
- **Euler-Bernoulli beam theory** neglects shear deformation and is accurate for long, slender beams
- **Timoshenko beam theory** accounts for shear effects and performs better for shorter, thicker cross-sections and higher frequencies

In practice, the appropriate theory depends on beam geometry, loading frequency, and material properties, yet current digital twin implementations often default to one theory without rigorous model selection.

## Project Goals

1. **Generate synthetic measurement data** (displacements, strains) from a cantilever beam with varying length-to-height ratios using a reference 1D Timoshenko beam finite element model

2. **Implement Bayesian calibration** for each beam theory separately, learning material and geometric parameters from the synthetic data, and compute marginal likelihoods to quantify evidence for each model

3. **Analyze model selection results** across different beam aspect ratios, providing evidence thresholds and practical criteria for when each theory should be selected in digital twin applications

## Key Results

The project successfully demonstrates **physics-aligned model selection**:

- **Thick beams (L/h ≤ 15)**: Timoshenko strongly preferred (log BF down to -10.8 at L/h=5)
- **Transition region (L/h ≈ 19.2)**: Model preference switches from Timoshenko to Euler-Bernoulli
- **Slender beams (L/h = 20, 30)**: Euler-Bernoulli preferred
- **Very slender beams (L/h = 50)**: Models become indistinguishable (both correct within noise level)

The critical fix: Using **1D Timoshenko beam FEM** as ground truth ensures exact consistency between reference solution and beam theories, enabling proper model discrimination.

## Project Structure

```
digital_twin_lab_project/
├── main.py                      # Main entry point
├── Makefile                     # Build and run commands
├── pyproject.toml               # Project dependencies and configuration
├── README.md                    # This file
│
├── apps/                        # Main application code
│   ├── __init__.py
│   │
│   ├── models/                  # Beam theory implementations
│   │   ├── __init__.py
│   │   ├── base_beam.py         # Abstract base class for beams
│   │   ├── euler_bernoulli.py   # Euler-Bernoulli beam model
│   │   └── timoshenko.py        # Timoshenko beam model
│   │
│   ├── fem/                     # Finite Element Method
│   │   ├── __init__.py
│   │   ├── cantilever_fem.py    # Legacy 2D plane stress FEM (not used)
│   │   └── beam_fem.py          # 1D Timoshenko beam FEM (ground truth)
│   │
│   ├── data/                    # Data generation
│   │   ├── __init__.py
│   │   └── synthetic_generator.py  # Synthetic measurement data from 1D FEM
│   │
│   ├── bayesian/                # Bayesian inference
│   │   ├── __init__.py
│   │   ├── calibration.py       # Bayesian parameter calibration with PyMC
│   │   └── model_selection.py   # Model comparison and Bayes factors
│   │
│   ├── analysis/                # Analysis and visualization
│   │   ├── __init__.py
│   │   ├── visualization.py     # Plotting functions
│   │   └── reporter.py          # Report generation
│   │
│   ├── pipeline/                # Workflow orchestration
│   │   ├── __init__.py
│   │   └── orchestrator.py      # Pipeline coordination
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── config.py            # Configuration loading
│       └── logging_setup.py     # Logging configuration
│
├── configs/                     # Configuration files
│   └── default_config.yaml      # Default parameters
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_beam_models.py      # Tests for beam models
│   └── test_fem.py              # Tests for FEM module
│
└── outputs/                     # Generated outputs (created at runtime)
    ├── data/                    # Synthetic datasets
    ├── figures/                 # Visualization outputs
    └── reports/                 # Text and JSON reports
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project

# Install using Make (recommended)
make install

# Or manually create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

### Run the Full Pipeline

```bash
# Using Make (recommended)
make run

# Or directly with Python
python main.py --config configs/default_config.yaml --output-dir outputs --stage all
```

### Run Individual Stages

```bash
# Generate synthetic data only
make run-data

# Run Bayesian calibration only
make run-calibration

# Run model selection analysis only
make run-analysis

# Generate reports only
make run-report
```

### Command Line Options

```bash
python main.py --help

Options:
  -c, --config PATH       Path to configuration file
  -o, --output-dir PATH   Directory for output files
  -s, --stage [all|data|calibration|analysis|report]
                          Pipeline stage to run
  -a, --aspect-ratios FLOAT
                          Beam aspect ratios (L/h) to analyze (multiple)
  -v, --verbose           Enable verbose output
  --debug                 Enable debug mode
  --help                  Show this message and exit
```

### Custom Aspect Ratios

```bash
python main.py -a 5 -a 10 -a 15 -a 20 -a 30 --stage all
```

## Key Concepts

### Beam Theories

#### Euler-Bernoulli Theory
- Assumes plane sections remain plane and **perpendicular** to the neutral axis
- Neglects shear deformation
- Accurate for **slender beams** (L/h > 10-20)
- Deflection formula: δ = PL³/(3EI)

#### Timoshenko Theory
- Accounts for shear deformation and rotary inertia
- Sections remain plane but **not perpendicular** to the deformed neutral axis
- Accurate for **thick beams** and **higher frequencies**
- Deflection formula: δ = PL³/(3EI) + PL/(κGA)

### Bayesian Model Selection

The project uses Bayesian inference to:

1. **Calibrate parameters** (elastic modulus E, Poisson's ratio ν) from synthetic measurement data
2. **Compute marginal likelihoods** p(y|M) for each model
3. **Calculate Bayes factors** BF = p(y|M₁)/p(y|M₂) to quantify evidence

#### Bayes Factor Interpretation (Kass & Raftery, 1995)

| Bayes Factor | log(BF) | Evidence Strength |
|--------------|---------|-------------------|
| 1-3          | 0-1     | Weak              |
| 3-10         | 1-2.3   | Moderate          |
| 10-100       | 2.3-4.6 | Strong            |
| >100         | >4.6    | Decisive          |

## TODO Tasks

Most core functionality has been implemented. Remaining enhancements:

### Testing & Validation
- [ ] Expand unit test coverage for edge cases
- [ ] Add integration tests for full pipeline
- [ ] Validate against experimental data

### Features
- [ ] Add frequency-domain analysis
- [ ] Implement dynamic loading scenarios
- [ ] Support additional boundary conditions

## Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
beam_parameters:
  length: 1.0
  width: 0.1
  aspect_ratios: [5, 8, 10, 12, 15, 20, 30, 50]

material:
  elastic_modulus: 210.0e9
  poisson_ratio: 0.3

bayesian:
  n_samples: 800
  n_tune: 400
  n_chains: 2

data:
  noise_fraction: 0.0005  # 0.05% noise
```

## Expected Results

The full pipeline produces physically accurate model selection results:

### Model Selection Summary

| Aspect Ratio (L/h) | Log Bayes Factor | Recommended Model |
|--------------------|------------------|-------------------|
| 5.0                | -10.830          | Timoshenko        |
| 8.0                | -7.377           | Timoshenko        |
| 10.0               | -4.146           | Timoshenko        |
| 12.0               | -3.595           | Timoshenko        |
| 15.0               | -2.109           | Timoshenko        |
| 20.0               | +0.420           | Euler-Bernoulli   |
| 30.0               | +0.255           | Euler-Bernoulli   |
| 50.0               | -0.031           | Inconclusive      |

**Note**: Log BF > 0 favors Euler-Bernoulli, Log BF < 0 favors Timoshenko

**Transition aspect ratio**: L/h ≈ 19.2

### Key Findings

1. **Thick beams (L/h ≤ 15)**: Strong evidence for Timoshenko theory, with log BF ranging from -2.1 to -10.8
2. **Critical transition**: Model preference switches at L/h ≈ 19.2, aligning with beam theory expectations
3. **Slender beams (L/h = 20-30)**: Euler-Bernoulli preferred but evidence is weaker (log BF ≈ 0.2-0.4)
4. **Very slender beams (L/h = 50)**: Both models statistically equivalent (difference below noise level)

### Digital Twin Recommendation

For practical digital twin implementation:
1. Check beam aspect ratio against thresholds during initialization
2. Default to Timoshenko for safety in uncertain cases (conservative approach)
3. Recalibrate model selection when geometry or loading changes
4. For high-frequency analysis (f > f₁), prefer Timoshenko regardless of aspect ratio

## References

1. **Euler vs Timoshenko comparison**: [J. Sound Vib. 2020](https://doi.org/10.1016/j.jsv.2020.115432)
2. **Bayesian Data Analysis**: Gelman et al., 3rd edition
3. **Bayes Factors**: Kass & Raftery (1995), JASA
4. **PyMC Documentation**: https://www.pymc.io/

## Development

### Running Tests

```bash
make test          # Run all tests
make test-cov      # Run with coverage report
```

### Code Quality

```bash
make lint          # Check code style
make format        # Format code
```

## License

MIT License - see LICENSE file for details.

## Author

Digital Twins Lab - WS2025/2026