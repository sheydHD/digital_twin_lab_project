# Bayesian Glossary

| Field        | Value                                       |
|--------------|---------------------------------------------|
| **Author**   | Antoni Dudij, Maksim Feldmann — RWTH Aachen |
| **Status**   | Review                                      |
| **Last Updated** | 2026-03-01                              |

---

## TL;DR

This glossary covers all 22 Bayesian and statistical concepts used in the codebase, anchored to the exact file and method where each one is instantiated. Every entry answers three questions: what the concept is mathematically, where it appears in the code, and whether it executes during a standard `make run`.

---

## 1. MCMC — Markov Chain Monte Carlo

MCMC samples from the posterior distribution $p(\boldsymbol{\theta} \mid \mathbf{y}, \mathcal{M})$ when it cannot be computed in closed form. Rather than solving the integral analytically, it constructs a Markov chain whose stationary distribution is the posterior and draws dependent samples from it.

**Code location:** `calibration.py` — `calibrate()` calls `pm.sample(...)`. **Runs during `make run`:** yes.

---

## 2. NUTS — No-U-Turn Sampler

NUTS is an adaptive Hamiltonian Monte Carlo variant that uses gradients of the log-posterior to propose moves and automatically tunes the trajectory length to avoid the "U-turn" degeneracy of fixed-length HMC. It requires substantially fewer samples than random-walk Metropolis-Hastings to achieve the same effective sample size.

**Code location:** `pm.sample(...)` invokes NUTS by default for all continuous parameters. **Config:** `target_accept=0.95`. **Runs during `make run`:** yes.

---

## 3. Warmup / Tuning

The initial MCMC phase during which the sampler adapts its step size and mass matrix estimate. All draws from this phase are discarded before computing any posterior statistics.

**Code location:** `pm.sample(tune=self.n_tune, ...)` in `calibrate()`. Default: 400 tuning draws per chain. **Runs during `make run`:** yes.

---

## 4. Posterior Distribution

The updated belief about parameters after observing data. By Bayes' theorem:

$$p(\boldsymbol{\theta} \mid \mathbf{y}, \mathcal{M}) = \frac{p(\mathbf{y} \mid \boldsymbol{\theta}, \mathcal{M})\; p(\boldsymbol{\theta} \mid \mathcal{M})}{p(\mathbf{y} \mid \mathcal{M})}$$

**Code location:** stored in `self._trace` (an ArviZ `InferenceData` object) inside `BayesianCalibrator`; summarised via `az.summary(self._trace)`. **Runs during `make run`:** yes.

---

## 5. Prior Distribution

The belief about parameters before observing data. Priors are specified per-parameter and encode domain knowledge (e.g., steel has $E \approx 210$ GPa).

| Parameter | Prior | Interpretation |
|---|---|---|
| $E$ (normalised) | $\mathcal{N}(1.0,\ 0.05)$ | $E \approx 210$ GPa $\pm 5\%$ |
| $\nu$ (Timoshenko only) | $\mathcal{N}(0.3,\ 0.03)$ | Steel Poisson's ratio |
| $\sigma$ (noise, normalised) | $\text{HalfNormal}(1.0)$ | Positive observation noise |

**Code location:** `_build_pymc_model()` in `calibration.py`. **Runs during `make run`:** yes.

---

## 6. Likelihood Function

The probability of observing the data given the parameters. Each sensor measurement $y_i$ is modelled as the beam theory prediction at that location plus independent Gaussian noise:

$$p(\mathbf{y} \mid \boldsymbol{\theta}, \mathcal{M}) = \prod_{i=1}^{n} \mathcal{N}\!\left(y_i \mid \hat{w}_{\mathcal{M}}(x_i;\, \boldsymbol{\theta}),\; \sigma^2\right)$$

**Code location:** `pm.Normal("y_obs", mu=y_pred, sigma=sigma, observed=y_obs_norm)` in `_build_pymc_model()`. **Runs during `make run`:** yes.

---

## 7. Forward Model

The physics function mapping parameters $\boldsymbol{\theta}$ to predicted measurements. During MCMC this function must be expressed in PyTensor symbolic arithmetic so gradients can be computed by automatic differentiation.

| Model | Deflection | Parameters |
|---|---|---|
| Euler-Bernoulli | $w(x) = -\frac{Px^2}{6EI}(3L - x)$ | $E$ |
| Timoshenko | $w(x) = -\frac{Px^2}{6EI}(3L - x) - \frac{Px}{\kappa GA}$ | $E,\ \nu$ |
| Strain (both) | $\varepsilon(x) = -\frac{h}{2} \cdot \frac{P(L-x)}{EI}$ | $E$ |

**Code location:** `_pytensor_forward_normalized()` and `_pytensor_strain_forward_normalized()` in `calibration.py`. **Runs during `make run`:** yes.

---

## 8. Normalisation

Beam mechanics spans ~14 orders of magnitude ($E \sim 10^{11}$ Pa, $w \sim 10^{-5}$ m). NUTS step-size tuning collapses on problems with such large dynamic ranges. All quantities are scaled to $\mathcal{O}(1)$ before entering the PyMC graph:

$$E_{norm} = \frac{E}{E_{scale}}, \quad E_{scale} = 210 \times 10^9$$

$$w_{norm} = \frac{w}{w_{scale}}, \quad w_{scale} = \max|\mathbf{w}_{obs}|$$

Posterior samples are denormalised before being written to `CalibrationResult.posterior_summary`.

**Code location:** `normalization.py` — `compute_normalization_params()`, `NormalizationParams`. **Runs during `make run`:** yes.

---

## 9. $\hat{R}$ — Convergence Diagnostic

The potential scale reduction factor compares variance within individual chains to variance across chains. A well-mixed chain produces $\hat{R} \approx 1.0$.

| $\hat{R}$ | Decision |
|---|---|
| $< 1.01$ | Converged |
| $1.01$–$1.05$ | Issue; raise `ConvergenceWarning` |
| $> 1.05$ | Reject; raise `ConvergenceError` |

Empirical results across all aspect ratios: $\hat{R} = 1.002$–$1.003$.

**Code location:** `az.rhat(self._trace)` in `calibrate()`. **Runs during `make run`:** yes.

---

## 10. ESS — Effective Sample Size

Because MCMC draws are autocorrelated, $N$ raw samples are not equivalent to $N$ independent samples. ESS estimates the equivalent number of independent draws.

| Metric | Threshold |
|---|---|
| `ESS_bulk` | $> 400$ |
| `ESS_tail` | $> 200$ |

Empirical results: $\text{ESS} = 1\,250$–$1\,650$.

**Code location:** `az.ess(self._trace)` in `calibrate()`. **Runs during `make run`:** yes.

---

## 11. WAIC — Widely Applicable Information Criterion

An estimate of out-of-sample log predictive density:

$$\text{WAIC} = -2\!\left(\widehat{\text{elpd}} - p_{\text{WAIC}}\right)$$

where $\widehat{\text{elpd}}$ measures fit quality and $p_{\text{WAIC}}$ is the effective parameter count acting as a complexity penalty. **WAIC is computed as a convergence diagnostic only.** Model selection uses bridge sampling marginal likelihoods because WAIC does not approximate the marginal likelihood (see [entry 14](#14-bridge-sampling)).

**Code location:** `az.waic(self._trace)` in `calibrate()`. **Runs during `make run`:** yes.

---

## 12. Log Bayes Factor

The log ratio of marginal likelihoods under two competing models. By convention, $\mathcal{M}_1$ = Euler-Bernoulli and $\mathcal{M}_2$ = Timoshenko throughout this project:

$$\ln B_{12} = \ln p(\mathbf{y} \mid \mathcal{M}_1) - \ln p(\mathbf{y} \mid \mathcal{M}_2)$$

A positive value favours EB; a negative value favours Timoshenko; $|\ln B_{12}| < 0.5$ is classified as inconclusive.

**Code location:** `BayesianModelSelector.compare_models()` in `model_selection.py`. **Runs during `make run`:** yes.

---

## 13. Marginal Likelihood (Model Evidence)

The probability of the observed data under a model, averaged over all possible parameter values:

$$p(\mathbf{y} \mid \mathcal{M}) = \int p(\mathbf{y} \mid \boldsymbol{\theta}, \mathcal{M})\; p(\boldsymbol{\theta} \mid \mathcal{M})\; d\boldsymbol{\theta}$$

This integral is high when the model is both accurate and parsimonious — it automatically penalises unnecessary parameters (see [entry 16](#16-occams-razor-bayesian)).

**Code location:** `compute_marginal_likelihood()` in `calibration.py`, estimated via `BridgeSampler`. **Runs during `make run`:** yes.

---

## 14. Bridge Sampling

The gold-standard numerical method for estimating the marginal likelihood from MCMC output. It uses the identity (Meng & Wong, 1996):

$$\ln p(\mathbf{y} \mid \mathcal{M}) = \ln \frac{\mathbb{E}_{q}\!\left[\frac{p(\mathbf{y}\mid\boldsymbol{\theta})\,p(\boldsymbol{\theta}\mid\mathcal{M})}{q(\boldsymbol{\theta})}\right]}{\mathbb{E}_{\pi}\!\left[\frac{1}{q(\boldsymbol{\theta})}\right]}$$

where $q(\boldsymbol{\theta})$ is a proposal distribution (a multivariate normal fitted to the posterior) and $\pi$ denotes the posterior. The equation is iterated until convergence.

**Code location:** `BridgeSampler` in `bridge_sampling.py`. **Runs during `make run`:** yes.

---

## 15. Kass–Raftery Evidence Scale

The standard interpretation of Bayes factor magnitude (Kass & Raftery, 1995):

| $|\ln B_{12}|$ | Evidence |
|---|---|
| $< 0.5$ | Inconclusive |
| $0.5$–$1.0$ | Weak |
| $1.0$–$2.3$ | Moderate |
| $> 2.3$ | Strong |

**Code location:** `BayesianModelSelector._interpret_bayes_factor()` in `model_selection.py`. **Runs during `make run`:** yes.

---

## 16. Occam's Razor (Bayesian)

Bayesian model selection naturally penalises model complexity without any explicit penalty term. A model with more parameters spreads its prior probability mass over a larger parameter space; unless those additional parameters improve the likelihood, the marginal likelihood integral is smaller. In this project, Timoshenko has one extra parameter ($\nu$) relative to EB. For slender beams where shear is negligible, $\nu$ contributes no information to the fit, and the marginal likelihood penalises Timoshenko accordingly. Because the $\nu$ prior is tight ($\sigma = 0.03$), the penalty is small — which is why $\ln B_{12}$ plateaus near zero for high $L/h$ rather than growing strongly positive.

**Code location:** implicit in the marginal likelihood integral; `BridgeSampler` captures this automatically.

---

## 17. HDI — Highest Density Interval

The narrowest interval that contains a specified probability mass (94% by default in ArviZ) of the posterior. Unlike a symmetric credible interval, the HDI adapts to asymmetric or multimodal posteriors.

**Code location:** `az.summary(self._trace)` returns `hdi_3%` and `hdi_97%` columns in `CalibrationResult.posterior_summary`. **Runs during `make run`:** yes.

---

## 18. InferenceData

ArviZ's hierarchical data structure for storing MCMC output. It organises draws into named groups (`posterior`, `log_likelihood`, `sample_stats`, etc.) and provides built-in serialisation to NetCDF.

**Code location:** `self._trace` in `BayesianCalibrator`; accessed by bridge sampler and reporter. **Runs during `make run`:** yes.

---

## 19. Log-Likelihood (pointwise)

The log-likelihood evaluated at each posterior sample for each individual data point. Stored in the `log_likelihood` group of the `InferenceData` object; required for WAIC computation.

**Code location:** `pm.compute_log_likelihood(self._trace)` called in `calibrate()`. **Runs during `make run`:** yes.

---

## 20. `adapt_diag` Initialisation

A NUTS initialisation strategy that estimates a diagonal mass matrix during warmup by fitting the variance of each parameter independently. It is more robust than default initialisation for models whose geometry deviates from a standard normal.

**Code location:** `pm.sample(init="adapt_diag", ...)` in `calibrate()`. **Runs during `make run`:** yes.

---

## 21. Target Accept Rate

The desired Metropolis acceptance probability for NUTS. Higher values (0.95) cause the dual-averaging algorithm to choose smaller step sizes, making the sampler more conservative but reducing divergences on difficult posteriors.

**Code location:** `pm.sample(target_accept=self.target_accept, ...)`. Default: 0.95. **Runs during `make run`:** yes.

---

## 22. Frequency-Based Model Selection

An analytical comparison of natural frequencies predicted by EB and Timoshenko theories. This is not a Bayesian method — no MCMC is involved. It serves as a complementary physics-based validation of the Bayesian results.

EB natural frequency for mode $n$:

$$f_n^{EB} = \frac{\lambda_n^2}{2\pi L^2} \sqrt{\frac{EI}{\rho A}}$$

Timoshenko frequencies are lower than EB predictions, especially for thick beams and higher modes, because shear and rotary inertia corrections reduce effective stiffness.

**Code location:** `FrequencyBasedModelSelector` in `hyperparameter_optimization.py`. Results written to `outputs/reports/frequency_analysis.txt`. **Runs during `make run`:** yes.

---

## Summary Table

| # | Concept | Runs? | Purpose |
|---|---|---|---|
| 1 | MCMC | yes | Posterior sampling |
| 2 | NUTS sampler | yes | Gradient-based efficient sampling |
| 3 | Warmup / tuning | yes | Sampler adaptation (samples discarded) |
| 4 | Posterior distribution | yes | Primary calibration output |
| 5 | Prior distributions | yes | Normal, HalfNormal on E, ν, σ |
| 6 | Likelihood function | yes | Gaussian noise model |
| 7 | Forward model | yes | EB and Timoshenko analytical equations |
| 8 | Normalisation | yes | Scale all quantities to O(1) |
| 9 | $\hat{R}$ | yes | Convergence validation |
| 10 | ESS | yes | Sample quality check |
| 11 | WAIC | yes | Diagnostic only |
| 12 | Log Bayes factor | yes | Primary model selection statistic |
| 13 | Marginal likelihood | yes | Estimated via bridge sampling |
| 14 | Bridge sampling | yes | Gold-standard evidence estimator |
| 15 | Kass–Raftery scale | yes | Evidence interpretation labels |
| 16 | Occam's razor | yes | Implicit in marginal likelihood |
| 17 | HDI | yes | Credible interval for posteriors |
| 18 | InferenceData | yes | ArviZ MCMC storage format |
| 19 | Log-likelihood (pointwise) | yes | Required for WAIC |
| 20 | `adapt_diag` init | yes | Robust NUTS mass matrix init |
| 21 | Target accept rate | yes | Conservative NUTS step-size tuning |
| 22 | Frequency analysis | yes | Analytical cross-validation (not Bayesian) |

---

## References

1. Gelman, A., et al. *Bayesian Data Analysis*, 3rd ed.
2. Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *Journal of the American Statistical Association*, 90(430), 773–795.
3. Meng, X.-L., & Wong, W. H. (1996). Simulating ratios of normalizing constants via a simple identity. *Statistica Sinica*, 6(4), 831–860.
4. Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using LOO-CV and WAIC. *Statistics and Computing*, 27(5), 1413–1432.
