# bayesian glossary

Every Bayesian concept used in the codebase, what it does, where it appears, and whether it runs during `make run`.

## 1. MCMC — Markov chain Monte Carlo

Samples from the posterior distribution when it cannot be computed analytically. Instead of solving $p(\theta | y)$ in closed form, MCMC generates samples from it.

Where: `calibration.py` — `calibrate()` calls `pm.sample(...)`.

Used during `make run`: yes. This is the core of the pipeline.

## 2. NUTS — No-U-Turn Sampler

An adaptive Hamiltonian Monte Carlo variant. Uses gradients of the log-posterior to propose moves. Automatically tunes trajectory length.

Where: `pm.sample(...)` uses NUTS by default for continuous parameters.

Used during `make run`: yes, implicitly via PyMC.

Config: `target_accept=0.95`.

## 3. warmup / tuning

Initial MCMC phase where the sampler adapts step size and mass matrix. These samples are discarded.

Where: `pm.sample(tune=self.n_tune, ...)` in `calibrate()`.

Used during `make run`: yes. 400 tuning draws per chain, all discarded.

## 4. posterior distribution

The updated belief about parameters after observing data:

$$p(\theta | y, \mathcal{M}) = \frac{p(y | \theta, \mathcal{M}) \cdot p(\theta | \mathcal{M})}{p(y | \mathcal{M})}$$

Where: MCMC trace object (`self._trace`). Accessed via `az.summary(self._trace)`.

Used during `make run`: yes. Primary output of calibration.

## 5. prior distribution

Belief about parameters before seeing data.

| parameter | prior | meaning |
|-----------|-------|---------|
| E (normalized) | $\mathcal{N}(1.0, 0.05)$ | E ~ 210 GPa +/- 5% |
| nu (Timoshenko) | $\mathcal{N}(0.3, 0.03)$ | steel Poisson's ratio |
| sigma (noise) | HalfNormal(1.0) | observation noise, positive |

Where: `_build_pymc_model()` in `calibration.py`.

Used during `make run`: yes.

## 6. likelihood function

Probability of observing data given parameters:

$$p(y | \theta, \mathcal{M}) = \prod_{i=1}^{n} \mathcal{N}(y_i \mid w_{\mathcal{M}}(x_i; \theta), \sigma^2)$$

Where: `pm.Normal("y_obs", mu=y_pred, sigma=sigma, observed=y_obs_norm)` in `_build_pymc_model()`.

Used during `make run`: yes.

## 7. forward model

Physics function mapping parameters to predictions. Given beam parameters, computes deflections or strains.

| model | deflection | parameters |
|-------|-----------|------------|
| Euler-Bernoulli | $w(x) = -\frac{Px^2}{6EI}(3L - x)$ | E |
| Timoshenko | $w(x) = -\frac{Px^2}{6EI}(3L - x) - \frac{Px}{\kappa GA}$ | E, nu |
| strain (both) | $\varepsilon(x) = -\frac{h}{2} \cdot \frac{P(L-x)}{EI}$ | E |

Where: `_pytensor_forward_normalized()` and `_pytensor_strain_forward_normalized()` in `calibration.py`.

Used during `make run`: yes, via PyTensor symbolic computation inside PyMC.

## 8. normalization

Scaling all quantities to O(1) before MCMC. Raw beam mechanics spans 14+ orders of magnitude (E ~ 10^11 Pa, w ~ 10^-5 m), which destroys MCMC convergence.

- $E_{\text{norm}} = E / E_{\text{scale}}$, where $E_{\text{scale}} = 210 \times 10^9$
- $w_{\text{norm}} = w / w_{\text{scale}}$, where $w_{\text{scale}} = \max|w_{\text{obs}}|$

Where: `normalization.py` — `compute_normalization_params()`, `NormalizationParams`.

Used during `make run`: yes.

## 9. R-hat — convergence diagnostic

Compares variance within chains to variance between chains. If chains have mixed properly, R-hat ~ 1.0.

Thresholds: < 1.01 converged, 1.01-1.05 warning, > 1.05 reject.

Where: `az.rhat(self._trace)` in `calibrate()`.

Used during `make run`: yes. Our results: R-hat = 1.002-1.003 across all runs.

## 10. ESS — effective sample size

Number of independent samples equivalent to the autocorrelated MCMC chain.

Thresholds: ESS_bulk > 400, ESS_tail > 200.

Where: `az.ess(self._trace)` in `calibrate()`.

Used during `make run`: yes. Our results: ESS = 1250-1650.

## 11. WAIC — widely applicable information criterion

Estimates out-of-sample predictive accuracy:

$$\text{WAIC} = -2 (\widehat{\text{elpd}} - p_{\text{WAIC}})$$

where elpd measures fit quality and p_WAIC is the effective parameter count (complexity penalty).

Where: `az.waic(self._trace)` in `calibrate()`.

Used during `make run`: yes, computed as a diagnostic. Model selection uses bridge sampling marginal likelihoods, not WAIC.

## 12. log Bayes factor

Log ratio of marginal likelihoods:

$$\ln BF_{12} = \ln p(y | \mathcal{M}_1) - \ln p(y | \mathcal{M}_2)$$

Convention: M1 = EB, M2 = Timoshenko. Positive favors EB, negative favors Timoshenko. |ln BF| < 0.5 is inconclusive.

Computed from bridge sampling marginal likelihoods.

Where: `BayesianModelSelector.compare_models()` in `model_selection.py`.

Used during `make run`: yes. This is the main result of the pipeline.

## 13. marginal likelihood (model evidence)

Probability of the data under a model, integrating over all parameter values:

$$p(y | \mathcal{M}) = \int p(y | \theta, \mathcal{M}) \, p(\theta | \mathcal{M}) \, d\theta$$

High when the model is both well-fitting and parsimonious.

Where: `compute_marginal_likelihood()` in `calibration.py`. Uses bridge sampling.

Used during `make run`: yes.

## 14. bridge sampling

Estimates marginal likelihoods using a proposal distribution (multivariate normal fitted to the posterior) and the optimal bridge function (Meng & Wong, 1996).

Where: `apps/bayesian/bridge_sampling.py` — `BridgeSampler` class.

Used during `make run`: yes. This is the primary method for marginal likelihood estimation and model selection.

## 15. Kass & Raftery evidence scale

Standard interpretation for Bayes factors (Kass & Raftery, 1995):

| |ln BF| | evidence |
|---------|----------|
| < 0.5 | inconclusive |
| 0.5 - 1.0 | weak |
| 1.0 - 2.3 | moderate |
| > 2.3 | strong |

Where: `BayesianModelSelector._interpret_bayes_factor()` in `model_selection.py`.

Used during `make run`: yes.

## 16. Occam's razor (Bayesian)

Emerges naturally in Bayesian model selection. A model with more parameters spreads its prior probability over a larger space. Unless extra parameters improve fit, the marginal likelihood is penalized.

In this project: Timoshenko has 3 parameters (E, nu, sigma) vs EB's 2 (E, sigma). For slender beams where shear is negligible, nu doesn't help fit the data, so the marginal likelihood penalizes Timoshenko and EB is preferred.

The penalty is small because our nu prior is tight (sigma=0.03), so little probability mass is wasted. This is why log BF plateaus near zero for slender beams rather than growing strongly positive.

Where: implicit in the marginal likelihood integral — bridge sampling captures this automatically.

Used during `make run`: yes.

## 17. HDI — highest density interval

Narrowest interval containing a specified probability mass (94% by default) of the posterior.

Where: `az.summary(self._trace)` returns `hdi_3%` and `hdi_97%` columns.

Used during `make run`: yes.

## 18. InferenceData

ArviZ's data structure for MCMC output. Contains groups for posterior, log_likelihood, sample_stats, etc.

Where: `self._trace` in `BayesianCalibrator`.

Used during `make run`: yes.

## 19. log-likelihood

Log of the likelihood evaluated at each posterior sample for each data point. Needed for WAIC.

Where: `pm.compute_log_likelihood(self._trace)` in `calibrate()`.

Used during `make run`: yes.

## 20. adapt-diag initialization

NUTS initialization strategy that estimates a diagonal mass matrix during warmup. More robust than default for models with non-standard geometry.

Where: `pm.sample(init="adapt_diag", ...)` in `calibrate()`.

Used during `make run`: yes.

## 21. target accept rate

Desired acceptance probability for NUTS. Higher values (0.95) make the sampler take smaller, more careful steps.

Where: `pm.sample(target_accept=self.target_accept, ...)`, set to 0.95.

Used during `make run`: yes.

## 22. frequency-based model selection

Analytical comparison of natural frequencies predicted by EB and Timoshenko theories. Not Bayesian — no MCMC involved.

Where: `FrequencyBasedModelSelector` in `hyperparameter_optimization.py`.

Used during `make run`: yes, results saved to `outputs/reports/frequency_analysis.txt`.

## summary table

| # | method | used? | purpose |
|---|--------|-------|---------|
| 1 | MCMC (PyMC) | yes | posterior sampling |
| 2 | NUTS sampler | yes | gradient-based sampling |
| 3 | warmup/tuning | yes | sampler adaptation (discarded) |
| 4 | posterior distribution | yes | calibration output |
| 5 | prior distributions | yes | Normal, HalfNormal |
| 6 | likelihood | yes | Gaussian noise model |
| 7 | forward model | yes | EB and Timoshenko formulas |
| 8 | normalization | yes | scale to O(1) |
| 9 | R-hat | yes | convergence check |
| 10 | ESS | yes | sample quality check |
| 11 | WAIC | yes | diagnostic only |
| 12 | log Bayes factor | yes | main result (bridge sampling) |
| 13 | marginal likelihood | yes | bridge sampling estimate |
| 14 | bridge sampling | yes | primary evidence method |
| 15 | Kass & Raftery scale | yes | evidence labels |
| 16 | Occam's razor | yes | implicit in marginal likelihood |
| 17 | HDI | yes | credible intervals |
| 18 | InferenceData | yes | ArviZ storage format |
| 19 | log-likelihood | yes | needed for WAIC |
| 20 | adapt-diag | yes | NUTS initialization |
| 21 | target accept | yes | conservative NUTS tuning |
| 22 | frequency analysis | yes | analytical, not Bayesian |

## pipeline execution flow

```
main.py --stage all
  orchestrator.run_full_pipeline()
    |
    +-- stage 1: data generation
    |     SyntheticDataGenerator -> 1D Timoshenko FEM -> noisy data
    |
    +-- stage 2: calibration
    |     for each L/h:
    |       EB calibrator: PyMC model -> NUTS -> trace -> WAIC -> bridge sampling
    |       Timo calibrator: PyMC model -> NUTS -> trace -> WAIC -> bridge sampling
    |
    +-- stage 3: model selection
    |     log BF = log ML(EB) - log ML(Timo) -> evidence labels
    |     find transition point -> L/h ~ 19.2
    |
    +-- stage 4: frequency analysis
    |     analytical EB vs Timo natural frequencies
    |
    +-- stage 5: reporting
          study_summary.txt, results.json, results.csv, figures
```
