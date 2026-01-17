# Bayesian Model Selection for Beam Theory in Structural Health Monitoring Digital Twins

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyMC](https://img.shields.io/badge/PyMC-5.10+-red.svg)](https://www.pymc.io/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/sheydHD/digital_twin_lab_project/graphs/commit-activity)

> **Automated Bayesian model selection framework for comparing Euler-Bernoulli and Timoshenko beam theories using probabilistic inference and information criteria.**

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Goals](#project-goals)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Concepts](#key-concepts)
- [Configuration](#configuration)
- [Expected Results](#expected-results)
- [Documentation](#documentation)
- [Development](#development)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

---

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

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run full pipeline (~30-40 minutes)
make run

# View results
cat outputs/reports/study_summary.txt
```

**That's it!** The pipeline will generate data, calibrate models, perform Bayesian model selection, and produce comprehensive reports.

---

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

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with tests
4. **Ensure** all tests pass (`make test`)
5. **Format** your code (`make format`)
6. **Commit** with conventional commits (`feat: add new beam theory`)
7. **Push** to your fork
8. **Open** a Pull Request

### Development Commands

```bash
make install       # Install with dev dependencies
make test          # Run all tests
make test-cov      # Run tests with coverage
make format        # Auto-format code (black, isort)
make lint          # Check code quality (pylint)
make clean         # Clean outputs and cache files
```

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed development guide.

---

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

---

## 📚 Documentation

### Core Documentation

- **[README.md](README.md)** (this file): Overview and quick start
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System design, patterns, and trade-offs
- **[API.md](docs/API.md)**: Complete API reference with examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Development guidelines and workflow
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: Development setup and best practices
- **[EXPLANATION.md](EXPLANATION.md)**: Plain-English project explanation

### Additional Resources

- **[CHANGELOG.md](CHANGELOG.md)**: Version history and migration guides
- **[LICENSE](LICENSE)**: MIT License
- **[SECURITY.md](SECURITY.md)**: Security policy and reporting

### Quick Links

- 🎯 [**Getting Started Guide**](#quick-start)
- 🏗️ [**Architecture Overview**](ARCHITECTURE.md#high-level-architecture)
- 🔧 [**API Reference**](docs/API.md)
- 🤝 [**Contributing Guidelines**](CONTRIBUTING.md)
- 🐛 [**Issue Tracker**](https://github.com/sheydHD/digital_twin_lab_project/issues)

---

## 💻 Development

## References

1. **Timoshenko Beam Theory**: Timoshenko, S.P. (1921). "On the correction for shear of the differential equation for transverse vibrations of prismatic bars." *Philosophical Magazine*, 41(245), 744-746.

2. **Bayesian Model Selection**: Kass, R.E., & Raftery, A.E. (1995). "Bayes factors." *Journal of the American Statistical Association*, 90(430), 773-795.

3. **WAIC**: Watanabe, S. (2010). "Asymptotic equivalence of Bayes cross validation and widely applicable information criterion in singular learning theory." *Journal of Machine Learning Research*, 11, 3571-3594.

4. **PyMC**: Salvatier, J., Wiecki, T.V., & Fonnesbeck, C. (2016). "Probabilistic programming in Python using PyMC3." *PeerJ Computer Science*, 2, e55.

5. **Digital Twins for SHM**: Worden, K., & Cross, E.J. (2018). "On switching response surface models, with applications to the structural health monitoring of bridges." *Mechanical Systems and Signal Processing*, 98, 139-156.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Digital Twins Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text in LICENSE file]
```

---

## 🙏 Acknowledgments

- **PyMC Development Team** for the excellent probabilistic programming framework
- **ArviZ Team** for MCMC diagnostics and visualization tools
- **Digital Twins Lab** for project supervision and feedback
- **Contributors** (see [CONTRIBUTORS.md](CONTRIBUTORS.md) when available)

---

## 📧 Contact

- **Issues**: [GitHub Issue Tracker](https://github.com/sheydHD/digital_twin_lab_project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sheydHD/digital_twin_lab_project/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for vulnerability reporting

---

## 📊 Project Stats

![GitHub last commit](https://img.shields.io/github/last-commit/sheydHD/digital_twin_lab_project)
![GitHub issues](https://img.shields.io/github/issues/sheydHD/digital_twin_lab_project)
![GitHub pull requests](https://img.shields.io/github/issues-pr/sheydHD/digital_twin_lab_project)
![Lines of code](https://img.shields.io/tokei/lines/github/sheydHD/digital_twin_lab_project)

---

<div align="center">

**[⬆ Back to Top](#bayesian-model-selection-for-beam-theory-in-structural-health-monitoring-digital-twins)**

Made with ❤️ by the Digital Twins Lab

</div>