# Complete Bayesian Statistics Glossary for This Project

This glossary explains every Bayesian concept used in the codebase, what it does, where it appears in the code, and whether it is **actually executed** during `make run`.

---

## 1. MCMC — Markov Chain Monte Carlo

**What it is**: A family of algorithms for sampling from probability distributions that cannot be evaluated analytically. Instead of computing $p(\theta | y)$ in closed form, MCMC generates samples $\theta_1, \theta_2, \ldots, \theta_N$ from the posterior.

**Why we need it**: The posterior $p(\theta | y, \mathcal{M})$ involves an integral in the denominator (the marginal likelihood) that is intractable for our beam models. MCMC avoids computing this integral by constructing a Markov chain whose stationary distribution *is* the posterior.

**Where in code**: `apps/bayesian/calibration.py` — the `calibrate()` method calls `pm.sample(...)`.

**Actually used during `make run`?**: **YES** — this is the core of the entire pipeline. Every calibration call runs MCMC.

**Config**: `configs/default_config.yaml` → `bayesian.n_samples: 1500`, `bayesian.n_tune: 800`, `bayesian.n_chains: 4`

---

## 2. NUTS — No-U-Turn Sampler

**What it is**: An adaptive variant of Hamiltonian Monte Carlo (HMC). HMC uses the gradient of the log-posterior to propose moves (like a particle sliding on the posterior surface). NUTS automatically tunes the trajectory length — it runs the particle forward and backward until it would "turn around" (the "U-turn" condition), preventing wasted computation.

**Why it matters**: NUTS is far more efficient than random-walk Metropolis-Hastings for continuous parameters. It explores the posterior with fewer samples and handles correlated parameters well.

**Where in code**: `apps/bayesian/calibration.py` — `pm.sample(...)` uses NUTS by default (PyMC's default sampler for continuous models). The initialization strategy is `init="adapt_diag"`, which estimates a diagonal mass matrix during warmup.

**Actually used during `make run`?**: **YES** — every `pm.sample()` call uses NUTS implicitly. PyMC selects it automatically because all our parameters are continuous.

**Config**: `target_accept=0.95` (higher than default 0.8 for better exploration of tight posteriors).

---

## 3. Warmup / Tuning

**What it is**: The initial phase of MCMC where the sampler adapts its internal parameters (step size, mass matrix) to the posterior geometry. These samples are **discarded** — they are not part of the posterior.

**Why it matters**: Without tuning, NUTS would use a generic step size that may be too large (causing rejections) or too small (causing slow exploration).

**Where in code**: `pm.sample(tune=self.n_tune, ...)` in `calibrate()`.

**Actually used during `make run`?**: **YES** — 800 tuning draws per chain, all discarded.

**Our values**: 800 warmup + 1500 kept = 2300 total draws per chain.

---

## 4. Posterior Distribution

**What it is**: The updated belief about parameters $\theta$ after observing data $y$:

$$p(\theta | y, \mathcal{M}) = \frac{p(y | \theta, \mathcal{M}) \cdot p(\theta | \mathcal{M})}{p(y | \mathcal{M})}$$

The posterior combines our prior knowledge with the information in the data via the likelihood.

**Where in code**: The MCMC trace object (`self._trace`) contains posterior samples. Accessed via `az.summary(self._trace)` for summary statistics, or directly as `self._trace.posterior`.

**Actually used during `make run`?**: **YES** — posterior samples are the primary output of calibration. Summary statistics (mean, HDI, R̂, ESS) are extracted and reported.

---

## 5. Prior Distribution

**What it is**: Our belief about parameters *before* seeing data. Encodes domain knowledge.

**Our priors** (in normalized space):

| Parameter | Prior | Physical meaning |
|-----------|-------|-----------------|
| E (normalized) | $\mathcal{N}(1.0, 0.05)$ | E ≈ 210 GPa ± 5% |
| ν (Timoshenko only) | $\mathcal{N}(0.3, 0.03)$ | Steel Poisson's ratio |
| σ (noise) | HalfNormal(1.0) | Observation noise, positive |

**Note**: The config file specifies `LogNormal(μ=26.07, σ=0.05)` for E, but the code normalizes E to O(1) and uses `Normal(1.0, 0.05)` in practice. This is equivalent in relative terms.

**Where in code**: `_build_pymc_model()` in `calibration.py` — prior definitions via `pm.Normal(...)`, `pm.HalfNormal(...)`. Prior configurations are created by `create_default_priors()` and `create_timoshenko_priors()`.

**Actually used during `make run`?**: **YES** — priors are fundamental to every MCMC run.

---

## 6. Likelihood Function

**What it is**: The probability of observing data $y$ given parameters $\theta$ and model $\mathcal{M}$:

$$p(y | \theta, \mathcal{M}) = \prod_{i=1}^{n} \mathcal{N}\big(y_i \;\big|\; w_{\mathcal{M}}(x_i; \theta),\; \sigma^2\big)$$

Each observation is assumed to be independently drawn from a Gaussian centered on the forward model prediction, with noise variance $\sigma^2$.

**Where in code**: `pm.Normal("y_obs", mu=y_pred, sigma=sigma, observed=y_obs_norm)` in `_build_pymc_model()`.

**Actually used during `make run`?**: **YES** — the likelihood is what MCMC evaluates at every step.

---

## 7. Forward Model

**What it is**: The physics function that maps parameters to observable predictions. Given beam parameters $\theta = (E, \nu)$ and positions $x_i$, it computes predicted deflections $w(x_i; \theta)$ or strains $\varepsilon(x_i; \theta)$.

**Our forward models**:

| Model | Deflection formula | Parameters |
|-------|-------------------|------------|
| Euler-Bernoulli | $w(x) = -\frac{Px^2}{6EI}(3L - x)$ | E |
| Timoshenko | $w(x) = -\frac{Px^2}{6EI}(3L - x) - \frac{Px}{\kappa GA}$ | E, ν |
| Strain (both) | $\varepsilon(x) = -\frac{h}{2} \cdot \frac{P(L-x)}{EI}$ | E |

**Where in code**: `_pytensor_forward_normalized()` (displacement), `_pytensor_strain_forward_normalized()` (strain) in `calibration.py`. Per-model overrides in `EulerBernoulliCalibrator` and `TimoshenkoCalibrator`.

**Actually used during `make run`?**: **YES** — called inside the PyMC model via PyTensor symbolic computation.

---

## 8. Normalization

**What it is**: Scaling all quantities to O(1) before MCMC. Raw beam mechanics quantities span 14+ orders of magnitude (E ~ 10¹¹ Pa, w ~ 10⁻⁵ m), which destroys MCMC convergence.

**Scaling**:
- $E_{\text{norm}} = E / E_{\text{scale}}$, where $E_{\text{scale}} = 210 \times 10^9$
- $w_{\text{norm}} = w / w_{\text{scale}}$, where $w_{\text{scale}} = \max|w_{\text{obs}}|$

**Where in code**: `apps/bayesian/normalization.py` — `compute_normalization_params()`, `NormalizationParams` dataclass. Applied in `_build_pymc_model()`.

**Actually used during `make run`?**: **YES** — all MCMC runs use normalized space.

---

## 9. R̂ (R-hat) — Gelman-Rubin Convergence Diagnostic

**What it is**: A convergence diagnostic that compares variance *within* chains to variance *between* chains. If chains have mixed properly, R̂ ≈ 1.0.

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

where $\hat{V}$ is pooled variance and $W$ is within-chain variance.

**Thresholds in our code**:
- R̂ < 1.01 → Converged ✓
- 1.01 < R̂ < 1.05 → Warning (marginal convergence)
- R̂ > 1.05 → Reject (chains not mixed)

**Where in code**: `az.rhat(self._trace)` in `calibrate()`. Validated in `_validate_convergence()`.

**Actually used during `make run`?**: **YES** — computed for every calibration, logged in convergence diagnostics.

**Our results**: R̂ = 1.002–1.003 across all configurations.

---

## 10. ESS — Effective Sample Size

**What it is**: The number of *independent* samples equivalent to the autocorrelated MCMC chain. If consecutive samples are correlated, the "effective" number of independent samples is less than the actual number drawn.

**Thresholds**:
- ESS_bulk > 400 → Reliable central estimates
- ESS_tail > 200 → Reliable quantile estimates

**Where in code**: `az.ess(self._trace)` in `calibrate()`.

**Actually used during `make run`?**: **YES** — computed and checked against thresholds.

**Our results**: ESS = 1250–1650 across all runs.

---

## 11. WAIC — Widely Applicable Information Criterion

**What it is**: An information criterion that estimates out-of-sample predictive accuracy:

$$\text{WAIC} = -2 \big( \widehat{\text{elpd}} - p_{\text{WAIC}} \big)$$

- $\widehat{\text{elpd}}$: expected log pointwise predictive density (measures fit)
- $p_{\text{WAIC}}$: effective number of parameters (complexity penalty)

WAIC naturally implements Occam's razor — a more complex model (Timoshenko with 3 params) is penalized relative to a simpler one (EB with 2 params) unless the extra parameters genuinely improve fit.

**Where in code**: `az.waic(self._trace)` in `calibrate()`. The result's `elpd_waic` is stored in `CalibrationResult.waic`.

**Actually used during `make run`?**: **YES** — WAIC is computed for each calibration as a diagnostic. However, model selection now uses bridge sampling marginal likelihoods as the primary method.

---

## 12. LOO-CV — Leave-One-Out Cross-Validation

**What it is**: Estimates predictive accuracy by approximating leave-one-out cross-validation using Pareto-smoothed importance sampling (PSIS). Each data point is held out in turn, and the model's ability to predict it is assessed.

**Where in code**: `az.loo(self._trace)` in `calibrate()`. Stored in `CalibrationResult.loo`.

**Actually used during `make run`?**: **YES** — computed alongside WAIC as an additional diagnostic. Model selection uses bridge sampling marginal likelihoods (`use_marginal_likelihood=True` in `analyze_aspect_ratio_study`).

---

## 13. Log Bayes Factor

**What it is**: The log ratio of marginal likelihoods (model evidences):

$$\ln BF_{12} = \ln p(y | \mathcal{M}_1) - \ln p(y | \mathcal{M}_2)$$

**Convention in our code**: M1 = Euler-Bernoulli, M2 = Timoshenko.
- $\ln BF > 0$ → favors EB
- $\ln BF < 0$ → favors Timoshenko
- $|\ln BF| < 0.5$ → inconclusive

**Computation**: We use bridge sampling marginal likelihoods: $\ln BF = \ln \hat{p}(y|\text{EB}) - \ln \hat{p}(y|\text{Timo})$, where each marginal likelihood is estimated via the Meng & Wong (1996) iterative bridge sampling algorithm.

**Where in code**: `BayesianModelSelector.compare_models()` in `model_selection.py`.

**Actually used during `make run`?**: **YES** — computed for all aspect ratios using bridge sampling. This is the main result of the pipeline.

---

## 14. Marginal Likelihood (Model Evidence)

**What it is**: The probability of the data under a model, integrating over all possible parameter values:

$$p(y | \mathcal{M}) = \int p(y | \theta, \mathcal{M}) \, p(\theta | \mathcal{M}) \, d\theta$$

This integral is the denominator in Bayes' theorem. It is high when the model is both well-fitting *and* parsimonious.

**Where in code**: `compute_marginal_likelihood()` in `calibration.py`. Uses bridge sampling by default; harmonic mean as fallback.

**Actually used during `make run`?**: **YES** — `compute_marginal_likelihood(method="bridge_sampling")` is called for each calibration result in the orchestrator. The bridge sampling estimate directly drives model selection via true Bayes factors.

---

## 15. Harmonic Mean Estimator

**What it is**: A simple estimator for the marginal likelihood:

$$\hat{p}(y | \mathcal{M}) = \left[ \frac{1}{N} \sum_{i=1}^{N} \frac{1}{p(y | \theta_i, \mathcal{M})} \right]^{-1}$$

where $\theta_i$ are posterior samples. This estimator is known to have **infinite variance** — it is dominated by the worst-fitting posterior sample. It is provided as a rough check, not as the primary method.

**Where in code**: `compute_marginal_likelihood(method="harmonic_mean")` in `calibration.py`. Uses log-sum-exp trick for numerical stability.

**Actually used during `make run`?**: **YES** — serves as the fallback if bridge sampling doesn't converge. The bridge sampling implementation falls back to harmonic mean when convergence fails.

---

## 16. Bridge Sampling

**What it is**: A more accurate method for estimating marginal likelihoods. It uses a proposal distribution $q(\theta)$ (typically a multivariate normal fitted to the posterior) and the identity:

$$p(y | \mathcal{M}) = \frac{E_q[p(y|\theta) p(\theta) h(\theta)]}{E_{\text{post}}[q(\theta) h(\theta)]}$$

where $h(\theta)$ is a bridge function. The optimal bridge function (Meng & Wong, 1996) minimizes variance.

**Where in code**: `apps/bayesian/bridge_sampling.py` — `BridgeSampler` class with full implementation. Also has `compute_bayes_factor_bridge()` convenience function.

**Actually used during `make run`?**: **YES** — Bridge sampling is the primary method for marginal likelihood estimation and model selection:
1. `compute_marginal_likelihood(method="bridge_sampling")` builds NumPy log-likelihood and log-prior callables from the calibrator's stored state
2. `BridgeSampler` fits an MVN proposal to the posterior, generates proposal samples, and iterates the optimal bridge function
3. `analyze_aspect_ratio_study()` uses `use_marginal_likelihood=True` — true Bayes factors from bridge sampling drive model selection
4. Falls back to harmonic mean if bridge sampling doesn't converge

**Status**: Fully implemented and wired as the primary evidence method.

---

## 17. Kass & Raftery Evidence Scale

**What it is**: A standard interpretation scale for Bayes factors (Kass & Raftery, 1995):

| |ln BF| | Evidence strength |
|---------|-------------------|
| < 0.5 | Not worth more than a bare mention (inconclusive) |
| 0.5 – 1.0 | Weak |
| 1.0 – 2.3 | Moderate |
| > 2.3 | Strong |

**Where in code**: `BayesianModelSelector._interpret_bayes_factor()` in `model_selection.py`. Also `BF_THRESHOLDS` dict and `INCONCLUSIVE_LOG_BF_THRESHOLD = 0.5`.

**Actually used during `make run`?**: **YES** — every comparison gets an evidence label (Strong/Moderate/Weak/Inconclusive).

---

## 18. Posterior Predictive Check

**What it is**: Simulating data from the posterior to check if the model can reproduce the observed data patterns. If the model is well-specified, simulated data should look like the real data.

**Where in code**: `posterior_predictive_check()` method in `BayesianCalibrator`.

**Actually used during `make run`?**: **NO** — the method exists but is not called by the orchestrator.

---

## 19. Occam's Razor (Bayesian)

**What it is**: In Bayesian model selection, Occam's razor emerges naturally. A model with more parameters spreads its prior probability over a larger parameter space. Unless those extra parameters genuinely improve fit, the marginal likelihood is diluted — the model is penalized for unnecessary complexity.

In our project: Timoshenko has 3 parameters (E, ν, σ) vs. EB's 2 (E, σ). For slender beams where shear is negligible, ν doesn't help fit the data → the marginal likelihood penalizes Timoshenko → EB is preferred.

**Why the penalty is small for slender beams**: The Occam penalty for an extra parameter depends on how much prior "volume" is wasted. Our ν prior is tight (Normal(0.3, 0.03)), so very little probability mass is wasted when ν is unused. This makes the Occam penalty for Timoshenko's extra parameter only ~0.05–0.2 in log scale — well below the inconclusive threshold of 0.5. That is why the log Bayes factor plateaus near zero for slender beams rather than growing strongly positive. A wider ν prior (e.g., σ=0.5) would produce a larger Occam penalty and stronger EB preference for slender beams.

**Where in code**: Implicit in the marginal likelihood integral — bridge sampling captures this automatically by integrating over the full prior volume.

**Actually used during `make run`?**: **YES** — this is why EB wins for slender beams despite Timoshenko technically fitting equally well.

---

## 20. HDI — Highest Density Interval

**What it is**: The narrowest interval containing a specified probability mass (e.g., 94%) of the posterior. Unlike equal-tailed intervals, HDI concentrates on the highest-density region.

**Where in code**: `az.summary(self._trace)` returns `hdi_3%` and `hdi_97%` columns (94% HDI by default in ArviZ).

**Actually used during `make run`?**: **YES** — reported in posterior summary for all parameters.

---

## 21. InferenceData

**What it is**: ArviZ's standard data structure for storing MCMC output. Contains groups for `posterior`, `log_likelihood`, `sample_stats`, etc. Stored as xarray Datasets.

**Where in code**: `self._trace` in `BayesianCalibrator` — returned by `pm.sample(return_inferencedata=True)`. Stored in `CalibrationResult.trace`.

**Actually used during `make run`?**: **YES** — all diagnostics, summaries, and model comparisons operate on InferenceData objects.

---

## 22. Log-Likelihood

**What it is**: The log of the likelihood function evaluated at each posterior sample for each data point. This matrix is needed by WAIC and LOO.

**Where in code**: `pm.compute_log_likelihood(self._trace)` in `calibrate()`. Accessed as `self._trace.log_likelihood["y_obs"]`.

**Actually used during `make run`?**: **YES** — computed after every MCMC run, required for WAIC and LOO.

---

## 23. Adapt-Diag Initialization

**What it is**: An initialization strategy for NUTS that estimates a diagonal mass matrix (one step size per parameter dimension) during an initial adaptation phase. This is more robust than the default `jitter+adapt_diag` for models with non-standard geometry.

**Where in code**: `pm.sample(init="adapt_diag", ...)` in `calibrate()`.

**Actually used during `make run`?**: **YES** — used for all MCMC runs.

---

## 24. Target Accept Rate

**What it is**: The desired acceptance probability for the NUTS sampler. Higher values (e.g., 0.95) make the sampler take smaller steps and explore the posterior more carefully, at the cost of higher autocorrelation.

**Where in code**: `pm.sample(target_accept=self.target_accept, ...)` where `self.target_accept = 0.95`.

**Actually used during `make run`?**: **YES** — set to 0.95 (higher than PyMC default of 0.8) for stability with our tightly constrained posteriors.

---

## 25. Frequency-Based Model Selection

**What it is**: Analytical comparison of natural frequencies predicted by EB and Timoshenko theories. For higher vibration modes, Timoshenko effects become significant even for slender beams because shear deformation and rotary inertia affect higher modes more strongly.

**Where in code**: `FrequencyBasedModelSelector` in `hyperparameter_optimization.py`. Called by `orchestrator.run_frequency_analysis()`.

**Actually used during `make run`?**: **YES** — analytical frequency comparison is run for all datasets. Results are saved in `outputs/reports/frequency_analysis.txt`.

**Note**: This is purely analytical, not Bayesian. No MCMC is run for frequency analysis.

---

## Summary Table: What Actually Runs During `make run`

| # | Method | Used? | Purpose |
|---|--------|-------|---------|
| 1 | MCMC (PyMC) | ✅ YES | Core posterior sampling |
| 2 | NUTS sampler | ✅ YES | Efficient gradient-based sampling |
| 3 | Warmup/tuning | ✅ YES | Sampler adaptation (800 draws, discarded) |
| 4 | Posterior distribution | ✅ YES | Output of calibration |
| 5 | Prior distributions | ✅ YES | Normal, HalfNormal for parameters |
| 6 | Likelihood | ✅ YES | Gaussian measurement noise model |
| 7 | Forward model | ✅ YES | EB and Timoshenko deflection formulas |
| 8 | Normalization | ✅ YES | Scale to O(1) for stable MCMC |
| 9 | R̂ diagnostic | ✅ YES | Convergence check |
| 10 | ESS diagnostic | ✅ YES | Sample quality check |
| 11 | WAIC | ✅ YES | Computed for diagnostics, no longer drives selection |
| 12 | LOO-CV | ✅ YES | Computed for diagnostics |
| 13 | Log Bayes factor | ✅ YES | Main result — bridge sampling marginal likelihoods |
| 14 | Marginal likelihood | ✅ YES | Bridge sampling estimate (primary), harmonic mean (fallback) |
| 15 | Harmonic mean estimator | ✅ YES | Fallback if bridge sampling doesn't converge |
| 16 | Bridge sampling | ✅ YES | Primary evidence method for model selection |
| 17 | Kass & Raftery scale | ✅ YES | Evidence interpretation labels |
| 18 | Posterior predictive check | ❌ NO | Method exists, not called in pipeline |
| 19 | Occam's razor | ✅ YES | Implicit in marginal likelihood (bridge sampling) |
| 20 | HDI | ✅ YES | 94% credible intervals in summary |
| 21 | InferenceData | ✅ YES | ArviZ storage format |
| 22 | Log-likelihood | ✅ YES | Required for WAIC/LOO |
| 23 | Adapt-diag init | ✅ YES | NUTS initialization strategy |
| 24 | Target accept = 0.95 | ✅ YES | Conservative NUTS tuning |
| 25 | Frequency analysis | ✅ YES | Analytical, not Bayesian |

---

## Pipeline Execution Flow (What `make run` Actually Does)

```
main.py --stage all
  └── orchestrator.run_full_pipeline()
        │
        ├── Stage 1: run_data_generation()
        │     └── SyntheticDataGenerator → 1D Timoshenko FEM → noisy data
        │
        ├── Stage 2: run_calibration()
        │     ├── For each L/h ratio (11 total):
        │     │     ├── EulerBernoulliCalibrator.calibrate()
        │     │     │     ├── _build_pymc_model() → priors + likelihood
        │     │     │     ├── pm.sample(NUTS, 4 chains, 800 tune, 1500 draws)
        │     │     │     ├── pm.compute_log_likelihood()
        │     │     │     ├── az.waic() → elpd_waic
        │     │     │     ├── az.loo() → elpd_loo
        │     │     │     └── az.rhat(), az.ess() → convergence check
        │     │     │
        │     │     ├── TimoshenkoCalibrator.calibrate()
        │     │     │     └── (same as above, but with ν parameter)
        │     │     │
        │     │     ├── eb_calibrator.compute_marginal_likelihood(method="bridge_sampling")
        │     │     │     └── BridgeSampler: MVN proposal → iterative bridge → log p(y|M)
        │     │     └── timo_calibrator.compute_marginal_likelihood(method="bridge_sampling")
        │     │           └── BridgeSampler: MVN proposal → iterative bridge → log p(y|M)
        │
        ├── Stage 3: run_analysis()
        │     └── BayesianModelSelector.analyze_aspect_ratio_study()
        │           ├── compare_models(use_marginal_likelihood=True)
        │           │     └── log_BF = log_ML_EB - log_ML_Timo  (bridge sampling)
        │           ├── _interpret_bayes_factor() → Kass & Raftery labels
        │           ├── _find_transition_point() → L/h ≈ 19.3
        │           └── _generate_guidelines()
        │
        ├── Stage 4: run_frequency_analysis()
        │     └── FrequencyBasedModelSelector.analyze_frequency_study()
        │           └── Analytical EB vs Timo natural frequencies
        │
        └── Stage 5: generate_report()
              ├── study_summary.txt
              ├── results.json, results.csv
              ├── calibration_eb_0..2.txt
              └── frequency_analysis.txt
```
