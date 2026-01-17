# Getting Started

This guide will help you install and run the Digital Twin Bayesian Model Selection framework.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10 or higher** ([download](https://www.python.org/downloads/))
- **Git** ([download](https://git-scm.com/downloads))
- **4 GB RAM minimum** (8 GB recommended)
- **2 GB free disk space**
- **Basic understanding of:**
  - Command line / terminal
  - Python virtual environments
  - Bayesian statistics (helpful but not required)

## Quick Start

Get up and running in under 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Run the pipeline
make run
```

That's it! The pipeline will:
- Generate synthetic datasets (8 aspect ratios)
- Calibrate Euler-Bernoulli and Timoshenko models  
- Perform Bayesian model selection
- Generate reports and visualizations

Expected runtime: **30-40 minutes** on a typical laptop.

## Detailed Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project
```

Or download the ZIP from GitHub and extract it.

### Step 2: Set Up Python Environment

We strongly recommend using a virtual environment to avoid dependency conflicts.

#### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Linux/Mac:
source .venv/bin/activate

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

#### Using conda

```bash
conda create -n digital_twin python=3.10
conda activate digital_twin
```

### Step 3: Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or just core dependencies
pip install -e .
```

**Dependencies installed:**
- PyMC ≥ 5.10.0 (Bayesian inference)
- ArviZ ≥ 0.17.0 (MCMC diagnostics)
- NumPy ≥ 1.24.0 (numerical computing)
- SciPy ≥ 1.11.0 (scientific computing)
- Matplotlib ≥ 3.7.0 (plotting)
- Pandas ≥ 2.0.0 (data handling)
- PyYAML ≥ 6.0 (configuration)

### Step 4: Verify Installation

```bash
# Run tests to verify everything works
make test

# Or manually
pytest tests/ -v
```

If all tests pass, you're ready to go! ✅

## Configuration

### Default Configuration

The default configuration is in `configs/default_config.yaml`:

```yaml
beam_parameters:
  length: 1.0          # Beam length [m]
  width: 0.05          # Beam width [m]
  aspect_ratios:       # L/h ratios to study
    - 5.0
    - 8.0
    - 10.0
    - 12.0
    - 15.0
    - 20.0
    - 30.0
    - 50.0

loading:
  force: 1000.0        # Tip load [N]

material:
  elastic_modulus: 210.0e9  # Steel: 210 GPa
  poisson_ratio: 0.3

bayesian:
  n_samples: 800       # MCMC samples per chain
  n_tune: 400          # Tuning samples
  n_chains: 2          # Parallel chains
  target_accept: 0.95  # NUTS acceptance target

data:
  n_sensors: 20        # Number of measurement points
  noise_fraction: 0.0005  # 0.05% relative noise
```

### Custom Configuration

Create your own configuration file:

```bash
cp configs/default_config.yaml configs/my_config.yaml
# Edit my_config.yaml
python main.py --config configs/my_config.yaml
```

### Common Customizations

#### Study Different Aspect Ratios

```yaml
beam_parameters:
  aspect_ratios: [3, 5, 7, 10, 15, 20, 25, 30]
```

#### Reduce Runtime (For Testing)

```yaml
bayesian:
  n_samples: 400  # Reduced from 800
  n_tune: 200     # Reduced from 400

beam_parameters:
  aspect_ratios: [5, 10, 20]  # Only 3 ratios
```

#### Increase Accuracy

```yaml
bayesian:
  n_samples: 1200  # More samples
  n_tune: 600      # More tuning

data:
  n_sensors: 50           # More measurement points
  noise_fraction: 0.0002  # Lower noise (0.02%)
```

## Running the Pipeline

### Full Pipeline

Run all stages in sequence:

```bash
# Using Makefile (recommended)
make run

# Or directly with Python
python main.py --stage all
```

### Individual Stages

Run specific stages for faster iteration:

```bash
# Stage 1: Data generation only
python main.py --stage data

# Stage 2: Bayesian calibration only (requires data)
python main.py --stage calibration

# Stage 3: Model selection analysis only (requires calibration)
python main.py --stage analysis

# Stage 4: Generate reports only (requires analysis)
python main.py --stage report
```

### Custom Options

```bash
# Specific aspect ratios
python main.py -a 5 -a 10 -a 20 --stage all

# Custom config file
python main.py --config my_config.yaml

# Custom output directory
python main.py --output-dir results/experiment_01

# Verbose output
python main.py --verbose

# Debug mode
python main.py --debug
```

### Using Make Commands

The `Makefile` provides convenient shortcuts:

```bash
make install       # Install dependencies
make run           # Run full pipeline
make run-data      # Data generation only
make run-calibration  # Calibration only
make run-analysis  # Analysis only
make run-report    # Reporting only
make test          # Run tests
make test-cov      # Tests with coverage
make clean         # Clean outputs
```

## Understanding the Output

After running the pipeline, check the `outputs/` directory:

```
outputs/
├── data/                    # Synthetic datasets
│   ├── Lh_5.0.csv
│   ├── Lh_8.0.csv
│   └── ...
│
├── traces/                  # MCMC traces (NetCDF format)
│   ├── Lh_5.0_euler_bernoulli.nc
│   ├── Lh_5.0_timoshenko.nc
│   └── ...
│
├── figures/                 # Plots and visualizations
│   ├── summary_report.png
│   ├── aspect_ratio_study.png
│   ├── posterior_Lh_5.0.png
│   └── ...
│
└── reports/                 # Text reports and summaries
    ├── study_summary.txt    # Main results
    ├── results.csv          # Raw data
    └── model_selection.json # JSON format
```

### Key Output Files

**`reports/study_summary.txt`** - Main results table:
```
Model Selection Summary
┌────────────────────┬──────────────────┬───────────────────┐
│ Aspect Ratio (L/h) │ Log Bayes Factor │ Recommended Model │
├────────────────────┼──────────────────┼───────────────────┤
│ 5.0                │ -10.830          │ Timoshenko        │
│ 8.0                │ -7.377           │ Timoshenko        │
│ ...                │ ...              │ ...               │
└────────────────────┴──────────────────┴───────────────────┘
```

**`figures/summary_report.png`** - Visual overview with:
- Bayes factors vs aspect ratio
- Model selection decision boundaries
- Posterior distributions
- Convergence diagnostics

**`results.csv`** - Raw data for further analysis:
```csv
aspect_ratio,log_bayes_factor,waic_eb,waic_timo,recommended_model
5.0,-10.830,-120.5,-142.3,Timoshenko
8.0,-7.377,-135.2,-150.0,Timoshenko
...
```

## Troubleshooting

### Installation Issues

**Problem**: `pip install` fails with compilation errors

**Solution**: Install build dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

**Problem**: PyMC import error

**Solution**: Try installing with conda:
```bash
conda install -c conda-forge pymc
```

### Runtime Issues

**Problem**: "MemoryError" during MCMC sampling

**Solution**: Reduce memory usage in config:
```yaml
bayesian:
  n_samples: 400
  n_chains: 2
```

**Problem**: MCMC not converging (R-hat > 1.01)

**Solution**: Increase tuning:
```yaml
bayesian:
  n_tune: 800
  target_accept: 0.99
```

**Problem**: Pipeline crashes at high aspect ratios

**Solution**: This shouldn't happen with the 1D FEM implementation, but if it does:
```yaml
beam_parameters:
  aspect_ratios: [5, 8, 10, 12, 15, 20, 30]  # Remove 50
```

### Common Warnings

**"WAIC starting to fail"** - Expected, indicates model doesn't fit perfectly (which is informative!)

**"Pareto k > 0.7"** - Expected for some data points, LOO-CV may be unreliable but WAIC compensates

**"Divergences detected"** - If < 1%, usually okay. If > 5%, increase `target_accept` to 0.99

## Next Steps

Now that you have the pipeline running:

1. **Explore the results**: Read `outputs/reports/study_summary.txt`
2. **Understand the code**: See [ARCHITECTURE.md](ARCHITECTURE.md)
3. **Modify parameters**: Edit `configs/default_config.yaml`
4. **Learn the API**: Read [API.md](API.md)
5. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Getting Help

- **Documentation**: Start with [docs/README.md](README.md)
- **Examples**: Check `examples/` directory (if available)
- **Issues**: Search [GitHub Issues](https://github.com/sheydHD/digital_twin_lab_project/issues)
- **Discussions**: Ask on [GitHub Discussions](https://github.com/sheydHD/digital_twin_lab_project/discussions)

---

**Ready to dive deeper?** Continue to [ARCHITECTURE.md](ARCHITECTURE.md) to understand how it all works!
