# getting started

## prerequisites

- Python 3.10+
- git
- 4 GB RAM minimum (8 GB recommended)

## quick start

```bash
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

make run
```

The pipeline generates synthetic datasets, calibrates both beam models, performs Bayesian model selection, and produces reports. Takes ~30-40 minutes.

## installation

### using venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### using conda

```bash
conda create -n digital_twin python=3.10
conda activate digital_twin
pip install -e ".[dev]"
```

### verify

```bash
make test
```

Key dependencies: PyMC >= 5.10, ArviZ >= 0.17, NumPy, SciPy, Matplotlib, Click, Rich, PyYAML.

## configuration

Default config: `configs/default_config.yaml`

```yaml
beam_parameters:
  length: 1.0
  width: 0.05
  aspect_ratios: [5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0]

loading:
  force: 1000.0

material:
  elastic_modulus: 210.0e9
  poisson_ratio: 0.3

bayesian:
  n_samples: 800
  n_tune: 400
  n_chains: 2
  target_accept: 0.95

data:
  n_sensors: 20
  noise_fraction: 0.0005
```

### reduce runtime (for testing)

```yaml
bayesian:
  n_samples: 400
  n_tune: 200
beam_parameters:
  aspect_ratios: [5, 10, 20]
```

### increase accuracy

```yaml
bayesian:
  n_samples: 1200
  n_tune: 600
data:
  n_sensors: 50
  noise_fraction: 0.0002
```

## running the pipeline

```bash
# full pipeline
make run

# individual stages
python main.py --stage data
python main.py --stage calibration
python main.py --stage analysis
python main.py --stage report

# custom aspect ratios
python main.py -a 5 -a 10 -a 20 --stage all

# custom config
python main.py --config configs/my_config.yaml

# verbose
python main.py --verbose --debug
```

### make commands

```bash
make install          # install dependencies
make run              # full pipeline
make run-data         # data generation only
make run-calibration  # calibration only
make run-analysis     # analysis only
make run-report       # reporting only
make test             # run tests
make test-cov         # tests with coverage
make clean            # clean outputs
```

## output

After running, check `outputs/`:

```
outputs/
  data/               synthetic datasets (HDF5)
  figures/            plots
    detailed/         detailed analysis plots
  reports/
    study_summary.txt main results table
    results.csv       raw data
    results.json      JSON format
```

`reports/study_summary.txt` contains the model selection summary with log Bayes factors and recommendations for each aspect ratio.

## troubleshooting

### pip install fails with compilation errors

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install
```

### PyMC import error

```bash
conda install -c conda-forge pymc
```

### memory error during MCMC

Reduce in config:
```yaml
bayesian:
  n_samples: 400
  n_chains: 2
```

### MCMC not converging (R-hat > 1.01)

Increase tuning:
```yaml
bayesian:
  n_tune: 800
  target_accept: 0.99
```

### divergences detected

If < 1%, usually fine. If > 5%, increase `target_accept` to 0.99.

## next steps

- read [architecture](ARCHITECTURE.md) for system design
- read [API reference](API.md) for function documentation
- read [explanation](EXPLANATION.md) for a plain-language overview
- edit `configs/default_config.yaml` to customize parameters
