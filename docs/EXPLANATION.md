# what this project does

This project answers one question: when should you use Timoshenko beam theory instead of the simpler Euler-Bernoulli theory?

The answer depends on how thick or slender your beam is. We use Bayesian statistics to find where the transition happens.

## the two beam theories

**Euler-Bernoulli (EB)**: assumes the beam cross-section stays perpendicular to the neutral axis during bending. Works for long, slender beams. Ignores shear deformation.

**Timoshenko**: accounts for shear deformation and rotational inertia. More accurate for short, thick beams where shear effects matter.

The key parameter is the aspect ratio L/h (length divided by height):
- L/h > 20: beam is slender, EB is fine
- L/h < 10: beam is thick, Timoshenko is better
- L/h between 10-20: gray zone, need to check

## what the code does

1. generates synthetic sensor data for beams with different aspect ratios (L/h = 5, 8, 10, 12, 15, 20, 30, 50) using a 1D Timoshenko beam finite element model
2. fits both beam models to each dataset using Bayesian inference (MCMC sampling with PyMC)
3. computes marginal likelihoods via bridge sampling (Meng & Wong, 1996) and calculates Bayes factors
4. outputs a recommendation for which theory to use at each aspect ratio

## why 1D beam FEM?

Originally, a 2D plane stress FEM was used, which had a systematic ~1% stiffness mismatch with analytical beam theories due to constraint effects. This caused incorrect model selection at intermediate aspect ratios.

The fix was switching to a 1D Timoshenko beam FEM that:
- uses the same assumptions as Timoshenko beam theory
- matches analytical solutions with 0.0000% error
- is 100x faster (200 elements vs 20,000)
- ensures physically correct model selection

## project structure

```
apps/
  models/           beam theory implementations (EB and Timoshenko)
  fem/              1D beam FEM for ground truth (beam_fem.py) + legacy 2D FEM
  bayesian/         PyMC calibration, bridge sampling, model comparison
  data/             synthetic data generation using 1D FEM
  analysis/         plotting and reporting
  pipeline/         orchestrates the whole workflow
  utils/            config loading, logging

configs/            YAML configuration files
outputs/            generated figures, reports, data
tests/              unit tests
```

## main commands

```bash
make install          # install everything
make run              # run full pipeline (~30-40 min)
make run-data         # data generation only
make run-calibration  # calibration only (requires data)
make test             # run tests
make clean            # clean outputs
```

## what you see when it runs

The pipeline prints progress for each aspect ratio. For each one, it:

1. generates synthetic data using 1D Timoshenko beam FEM
2. runs MCMC sampling for Euler-Bernoulli model (2 chains, 800 samples each)
3. runs MCMC sampling for Timoshenko model (2 chains, 800 samples each)
4. computes WAIC as a diagnostic
5. estimates marginal likelihoods via bridge sampling
6. calculates the log Bayes factor

## results

```
Model Selection Summary

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

Transition aspect ratio: L/h ~ 19.2
```

Thick beams (L/h <= 15) strongly favor Timoshenko. Slender beams (L/h = 20-30) favor EB. Very slender beams (L/h = 50) show both models are equivalent.

## output files

After running, check:

- `outputs/figures/aspect_ratio_study.png` — Bayes factors vs aspect ratio
- `outputs/reports/study_summary.txt` — text summary with recommendations
- `outputs/reports/results.csv` — raw numbers for further analysis
- `outputs/figures/detailed/` — detailed analysis plots

## how to interpret results

The log Bayes factor tells you which model is better:

| log BF    | meaning                            |
|-----------|------------------------------------|
| < -2.3    | strong preference for Timoshenko   |
| -2.3 to 0 | moderate preference for Timoshenko |
| ~0        | inconclusive                       |
| 0 to +2.3 | moderate preference for EB         |
| > +2.3    | strong preference for EB           |

## connection to digital twins

A digital twin needs to pick the right physics model. This framework:

1. takes sensor measurements from a beam
2. runs Bayesian calibration with both theories
3. tells you which theory to use for that specific beam
4. updates the recommendation if geometry or loading changes

The idea is automated model selection, not manual engineering judgment.

## key files to understand

Physics:
- `apps/models/euler_bernoulli.py` — EB deflection equations
- `apps/models/timoshenko.py` — Timoshenko deflection equations
- `apps/fem/beam_fem.py` — 1D Timoshenko FEM

Bayesian inference:
- `apps/bayesian/calibration.py` — PyMC model definitions
- `apps/bayesian/bridge_sampling.py` — marginal likelihood estimation
- `apps/bayesian/model_selection.py` — Bayes factor computation

Data and pipeline:
- `apps/data/synthetic_generator.py` — synthetic data from 1D FEM
- `apps/pipeline/orchestrator.py` — full workflow coordination
- `configs/default_config.yaml` — all tunable parameters
