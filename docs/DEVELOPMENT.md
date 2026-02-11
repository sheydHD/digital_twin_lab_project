# development guide

## setup

```bash
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test
```

Requires Python >= 3.10.

## running the pipeline

```bash
# full pipeline
make run

# individual stages
python main.py --stage data
python main.py --stage calibration
python main.py --stage analysis
python main.py --stage report

# specific aspect ratios
python main.py -a 5 -a 8 -a 10 --stage all

# verbose/debug
python main.py --verbose --debug
```

## testing

```bash
# all tests
make test

# with coverage
make test-cov

# specific file
pytest tests/test_beam_models.py -v

# specific test
pytest tests/test_beam_models.py::test_deflection -v

# with debugger on failure
pytest tests/test_beam_models.py -v -s --pdb
```

## code quality

```bash
# format
make format
# or: black apps/ tests/

# lint
make lint
# or: pylint apps/ tests/

# type check
mypy apps/

# import order
isort --check-only apps/ tests/
isort apps/ tests/  # fix
```

## project structure

```
apps/
  models/              beam theory implementations
    base_beam.py       abstract base class
    euler_bernoulli.py
    timoshenko.py

  fem/                 finite element solvers
    beam_fem.py        1D Timoshenko FEM (active)
    cantilever_fem.py  2D FEM (legacy)

  data/                data generation
    synthetic_generator.py

  bayesian/            probabilistic inference
    calibration.py     PyMC model definitions
    model_selection.py model comparison
    bridge_sampling.py marginal likelihood estimation
    normalization.py   MCMC normalization
    hyperparameter_optimization.py

  analysis/            post-processing
    visualization.py
    reporter.py

  pipeline/            orchestration
    orchestrator.py

  utils/               shared utilities
    config.py
    logging_setup.py
```

## debugging guide

### PyMC sampling fails

```
SamplingError: Initial evaluation of model at starting point failed!
```

Check the test point:
```python
with pm.Model() as model:
    # define model...
    test_point = model.initial_point()
    logp = model.logp(test_point)
    print(f"log probability: {logp}")
    # if -inf, the likelihood function has a problem
```

### memory issues

Reduce sample count:
```python
config["bayesian"]["n_samples"] = 500
config["bayesian"]["n_chains"] = 2
```

### numerical instability

Add guards:
```python
def safe_divide(num, denom, eps=1e-10):
    return num / (denom + eps)
```

Normalize inputs — all quantities should be O(1) before MCMC.

### slow MCMC convergence

Signs: R-hat > 1.01, low ESS, high divergences.

Fixes:
```python
n_tune = 2000      # more tuning
target_accept = 0.99  # smaller steps
```

## profiling

```bash
# function-level profiling
python -m cProfile -o profile.stats main.py
python -m pstats profile.stats

# line-level profiling
kernprof -l -v main.py

# memory profiling
python -m memory_profiler main.py
```

## common make commands

```bash
make install          # install with dev dependencies
make run              # full pipeline
make run-data         # data generation
make run-calibration  # calibration
make run-analysis     # analysis
make run-report       # reporting
make test             # run tests
make test-cov         # tests with coverage
make format           # auto-format (black, isort)
make lint             # check code quality
make clean            # clean outputs and caches
```
