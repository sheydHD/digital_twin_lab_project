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

1. **Generate synthetic measurement data** (displacements, strains) from a cantilever beam with varying length-to-height ratios using a reference high-fidelity finite element model

2. **Implement Bayesian calibration** for each beam theory separately, learning material and geometric parameters from the synthetic data, and compute marginal likelihoods to quantify evidence for each model

3. **Analyze model selection results** across different beam aspect ratios and loading frequencies, providing evidence thresholds and practical criteria for when each theory should be selected in digital twin applications

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
│   │   └── cantilever_fem.py    # High-fidelity 2D FEM reference model
│   │
│   ├── data/                    # Data generation
│   │   ├── __init__.py
│   │   └── synthetic_generator.py  # Synthetic measurement data generation
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

The codebase contains `TODO` markers indicating tasks to be completed. These are organized by category:

### Beam Models (Tasks 1-9)
- [ ] Task 1.1-1.6: Verify and implement deflection formulas
- [ ] Task 2.1-2.5: Implement rotation formulas
- [ ] Task 3.1-3.3: Implement strain computation
- [ ] Task 4.1-4.4: Implement natural frequency calculation
- [ ] Task 5.1-5.5: Implement Timoshenko deflection with shear correction
- [ ] Task 6.1-6.5: Implement Timoshenko rotation and shear angle
- [ ] Task 7.1-7.3: Implement Timoshenko strain
- [ ] Task 8.1-8.5: Implement Timoshenko frequency correction
- [ ] Task 9.1: Implement shear deformation ratio for model selection

### Finite Element Method (Tasks 10)
- [ ] Task 10.1-10.4: Implement complete FEM solver
- [ ] Task 10.5-10.11: Complete mesh and element stiffness
- [ ] Task 10.12-10.14: Implement assembly and boundary conditions
- [ ] Task 10.15-10.20: Implement solver and post-processing

### Data Generation (Tasks 11-14)
- [ ] Task 11.1-11.8: Complete FEM-based data generation
- [ ] Task 12.1-12.4: Implement parametric study
- [ ] Task 13.1-13.3: Implement dynamic frequency response
- [ ] Task 14.1-14.2: Implement HDF5 data storage

### Bayesian Calibration (Tasks 15-20)
- [ ] Task 15.1-15.8: Complete Bayesian calibration framework
- [ ] Task 16.1-16.3: Implement calibration pipeline
- [ ] Task 17.1-17.6: Implement marginal likelihood estimation
- [ ] Task 18.1-18.2: Implement posterior predictive checks
- [ ] Task 19.1-19.2: Implement forward models for calibration
- [ ] Task 20.1-20.3: Tune prior distributions

### Model Selection (Tasks 21-24)
- [ ] Task 21.1-21.5: Implement model selection pipeline
- [ ] Task 22.1-22.5: Implement aspect ratio study analysis
- [ ] Task 23.1-23.3: Develop practical guidelines
- [ ] Task 24.1: Implement robust Bayes factor estimation

### Visualization (Tasks 25-28)
- [ ] Task 25.1-25.5: Complete visualization functions
- [ ] Task 26.1-26.3: Implement posterior plotting
- [ ] Task 27.1-27.2: Implement model comparison plots
- [ ] Task 28.1-28.3: Create summary report

### Reporting & Pipeline (Tasks 29-31)
- [ ] Task 29.1-29.6: Complete reporting functionality
- [ ] Task 30.1-30.4: Complete pipeline implementation
- [ ] Task 31.1-31.6: Implement all pipeline stages

### Testing (Tasks T1-T8)
- [ ] Task T1-T3: Add beam model tests
- [ ] Task T4-T5: Add theory comparison tests
- [ ] Task T6-T8: Add FEM validation tests

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
  n_samples: 2000
  n_tune: 1000
  n_chains: 4
```

## Expected Results

After completing the TODOs and running the full pipeline, you should obtain:

1. **Transition Aspect Ratio**: The L/h value where model preference switches from Timoshenko to Euler-Bernoulli (typically around L/h = 10-15)

2. **Bayes Factor Analysis**: Quantitative evidence for each model across the aspect ratio range

3. **Practical Guidelines**: Recommendations for digital twin implementation based on beam slenderness

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