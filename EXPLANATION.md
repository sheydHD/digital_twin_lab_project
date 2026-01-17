# What This Project Does

This project answers one question: **When should you use Timoshenko beam theory instead of the simpler Euler-Bernoulli theory?**

The answer depends on how thick or slender your beam is. We use Bayesian statistics to figure out where the transition happens.


## The Two Beam Theories

**Euler-Bernoulli (EB)**: The simple one. Assumes the beam cross-section stays perpendicular to the neutral axis during bending. Works great for long, slender beams. Ignores shear deformation.

**Timoshenko**: The advanced one. Accounts for shear deformation and rotational inertia. More accurate for short, thick beams where shear effects matter.

The key parameter is the **aspect ratio L/h** (length divided by height):
- L/h > 20: Beam is slender, EB is fine
- L/h < 10: Beam is thick, Timoshenko is better
- L/h between 10-20: Gray zone, need to check


## What the Code Actually Does

1. **Generates fake sensor data** for beams with different aspect ratios (L/h = 5, 8, 10, 12, 15, 20, 30, 50)

2. **Fits both beam models** to each dataset using Bayesian inference (MCMC sampling with PyMC)

3. **Compares the models** using Bayes factors to determine which theory explains the data better

4. **Outputs a recommendation** for which theory to use at each aspect ratio


## Project Structure

```
apps/
  models/           # Beam theory implementations (EB and Timoshenko)
  fem/              # 2D finite element model for reference solutions
  bayesian/         # PyMC calibration and model comparison
  data/             # Synthetic data generation
  analysis/         # Plotting and reporting
  pipeline/         # Orchestrates the whole workflow
  utils/            # Config loading, logging

configs/            # YAML configuration files
outputs/            # Generated figures, reports, data
tests/              # Unit tests
```


## Main Commands

Install everything:
```bash
make install
```

Run the full pipeline:
```bash
make run
```

This runs all stages: data generation, Bayesian calibration, model selection analysis, and report generation. Takes about 5-10 minutes.

Run only data generation:
```bash
make run-data
```

Run only calibration (requires data first):
```bash
make run-calibration
```

Run tests:
```bash
make test
```

Clean outputs:
```bash
make clean
```


## What You See When It Runs

The pipeline prints progress for each aspect ratio. For each one, it:

1. Runs MCMC sampling for Euler-Bernoulli model (4 chains, 2000 samples each)
2. Runs MCMC sampling for Timoshenko model (4 chains, 2000 samples each)
3. Computes WAIC and LOO-CV for model comparison
4. Calculates Bayes factor

The warnings about "WAIC starting to fail" and "Pareto k > 0.7" are expected. They indicate the model might not fit perfectly, which is actually informative for model selection.


## Output Files

After running, check:

- `outputs/figures/summary_report.png`: Visual overview of all results
- `outputs/figures/aspect_ratio_study.png`: Bayes factors vs aspect ratio
- `outputs/reports/study_summary.txt`: Text summary with recommendations
- `outputs/reports/results.csv`: Raw numbers for further analysis


## How to Interpret Results

The **Log Bayes Factor** tells you which model is better:

| Log BF | Meaning |
|--------|---------|
| 0-1 | No real difference between models |
| 1-3 | Mild preference for one model |
| 3-5 | Strong preference |
| >5 | Very strong preference |

Positive values favor Timoshenko. Negative values favor Euler-Bernoulli.


## Current Limitations

The synthetic data generator creates analytical beam deflections plus noise. If the noise level is higher than the actual shear correction (a few percent), the Bayes factor cannot distinguish the theories.

To see clearer Timoshenko preference for thick beams:
- Lower `noise_fraction` in `configs/default_config.yaml` (try 0.005 instead of 0.02)
- Or use FEM-generated reference data which includes true shear effects


## Connection to Digital Twins

A digital twin needs to pick the right physics model. This framework:

1. Takes sensor measurements from a real beam
2. Runs Bayesian calibration with both theories
3. Tells you which theory to use for that specific beam
4. Updates the recommendation if geometry or loading changes

The idea is automated model selection, not manual engineering judgment.


## Key Files to Understand

If you want to modify the physics:
- `apps/models/euler_bernoulli.py`: EB deflection equations
- `apps/models/timoshenko.py`: Timoshenko deflection equations

If you want to modify the Bayesian inference:
- `apps/bayesian/calibration.py`: PyMC model definitions
- `apps/bayesian/model_selection.py`: Bayes factor computation

If you want to modify what gets generated:
- `apps/data/synthetic_generator.py`: How fake data is created
- `configs/default_config.yaml`: All tunable parameters


## Running Your Own Analysis

Edit `configs/default_config.yaml` to change:
- Beam geometry and material properties
- Aspect ratios to study
- Number of MCMC samples
- Noise levels
- Number of sensors

Then run `make run` again.
