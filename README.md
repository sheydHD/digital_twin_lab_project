# bayesian model selection for beam theory in digital twins

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Bayesian framework for comparing Euler-Bernoulli and Timoshenko beam theories using bridge sampling and MCMC inference. Determines which beam theory is appropriate for a given geometry based on data-driven evidence.

## overview

This project uses Bayesian model selection to decide when Euler-Bernoulli (EB) beam theory is sufficient and when Timoshenko theory is needed. It generates synthetic sensor data from a 1D Timoshenko FEM, calibrates both theories via MCMC (PyMC/NUTS), computes marginal likelihoods via bridge sampling, and outputs Bayes factors for a range of beam aspect ratios (L/h).

The main result: the transition from Timoshenko to Euler-Bernoulli preference occurs at L/h ~ 19.2, consistent with the engineering rule of thumb (L/h ~ 20), but now with quantitative probabilistic justification.

## quick start

```bash
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make run
```

The pipeline takes ~30-40 minutes. Results go to `outputs/`.

## project structure

```
apps/
  models/           beam theory implementations (EB, Timoshenko)
  fem/              1D Timoshenko beam FEM (ground truth)
  data/             synthetic data generation
  bayesian/         PyMC calibration, bridge sampling, model selection
  analysis/         plotting and reporting
  pipeline/         orchestration
  utils/            config, logging

configs/            YAML configuration
outputs/            generated data, figures, reports
tests/              unit tests
examples/           demo scripts
docs/               documentation
```

## installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+. Key dependencies: PyMC 5.10+, ArviZ 0.17+, NumPy, SciPy, Matplotlib.

## usage

```bash
# full pipeline
make run

# individual stages
python main.py --stage data
python main.py --stage calibration
python main.py --stage analysis
python main.py --stage report

# custom aspect ratios
python main.py -a 5 -a 10 -a 15 -a 20 -a 30 --stage all
```

See `python main.py --help` for all options.

## results

| L/h | log Bayes factor | recommended model |
|-----|-----------------|-------------------|
| 5   | -10.830         | Timoshenko        |
| 8   | -7.377          | Timoshenko        |
| 10  | -4.146          | Timoshenko        |
| 12  | -3.595          | Timoshenko        |
| 15  | -2.109          | Timoshenko        |
| 20  | +0.420          | Euler-Bernoulli   |
| 30  | +0.255          | Euler-Bernoulli   |
| 50  | -0.031          | inconclusive      |

Transition point: L/h ~ 19.2. Log BF < 0 favors Timoshenko, > 0 favors EB.

For thick beams (L/h <= 15), Timoshenko is strongly preferred. For slender beams (L/h >= 20), EB is sufficient. At very high L/h, both models are indistinguishable.

## configuration

Edit `configs/default_config.yaml`:

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
  noise_fraction: 0.0005
```

## documentation

- [getting started](docs/GETTING_STARTED.md) — installation and first run
- [architecture](docs/ARCHITECTURE.md) — system design and components
- [API reference](docs/API.md) — function and class documentation
- [explanation](docs/EXPLANATION.md) — plain-language project description
- [development](docs/DEVELOPMENT.md) — developer setup and workflow
- [parameters](docs/Parameters.md) — parameter tables and conventions
- [bayesian glossary](docs/BAYESIAN_GLOSSARY.md) — glossary of methods used
- [presentation guide](docs/PRESENTATION_GUIDE.md) — slide-by-slide guide
- [contributing](CONTRIBUTING.md) — contribution guidelines
- [changelog](CHANGELOG.md) — version history

## references

1. Timoshenko, S.P. (1921). On the correction for shear of the differential equation for transverse vibrations of prismatic bars. *Philosophical Magazine*, 41(245), 744-746.
2. Kass, R.E., & Raftery, A.E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
3. Meng, X.-L., & Wong, W.H. (1996). Simulating ratios of normalizing constants via a simple identity. *Statistica Sinica*, 6, 831-860.
4. Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and WAIC. *JMLR*, 11, 3571-3594.
5. Salvatier, J., Wiecki, T.V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.

## license

MIT License. See [LICENSE](LICENSE).

Authors: Antoni Dudij, Maksim Feldmann — RWTH Aachen University, Digital Twins Lab.
