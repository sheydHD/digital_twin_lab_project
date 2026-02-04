# Bayesian Model Selection Audit Analysis

## Executive Summary

This document presents a comprehensive audit of the Bayesian model selection implementation for comparing Euler-Bernoulli vs. Timoshenko beam theories in a Digital Twin application.

**Audit Date:** February 3, 2026  
**Auditor:** Computational Mechanics & Bayesian Inference Expert  
**Codebase:** `digital_twin_lab_project`

---

## Risk Assessment Overview

| Area | Risk Level | Current Status | Impact on Results |
|------|------------|----------------|-------------------|
| MCMC Convergence | ⚠️ MEDIUM | Adequate but improvable | ~10-20% error in evidence |
| Marginal Likelihood | 🔴 HIGH | Using WAIC proxy | Bayes factors may be incorrect |
| Physics Likelihood | ✅ GOOD | σ is learned correctly | Stable model selection |
| Prior Distributions | ⚠️ MEDIUM | Asymmetric penalty | Timoshenko unfairly penalized |
| Data Normalization | 🔴 HIGH | Not implemented | MCMC instability possible |

---

## 1. MCMC Convergence & Sampler Settings

### 1.1 Current Configuration

**File:** `configs/default_config.yaml`

```yaml
bayesian:
  n_samples: 800      # MCMC samples per chain
  n_tune: 400         # Tuning/warmup samples
  n_chains: 2         # Number of chains
  target_accept: 0.95 # Target acceptance rate
```

### 1.2 Identified Issues

#### Issue 1.2.1: Insufficient Samples for Marginal Likelihood Estimation

- **Current:** 800 draws × 2 chains = 1,600 effective samples
- **Required:** For stable WAIC/LOO-CV estimates, ~4,000+ effective samples recommended
- **Evidence:** WAIC warnings observed: `"posterior variance of the log predictive densities exceeds 0.4"`

#### Issue 1.2.2: R-hat Checks Not Enforced

**Location:** `apps/bayesian/calibration.py`

The code computes R-hat diagnostics but does not:
- Reject results when R-hat > 1.01
- Warn users about marginal convergence
- Automatically retry with more samples

```python
# Current implementation (approximately lines 290-310)
def _compute_diagnostics(self, trace) -> Dict:
    summary = az.summary(trace)
    r_hat_max = summary["r_hat"].max()
    # WARNING: No rejection logic implemented!
    return {"r_hat_max": r_hat_max, ...}
```

#### Issue 1.2.3: Only 2 Chains

- **Problem:** With only 2 chains, R-hat estimation is less reliable
- **Recommendation:** Use 4 chains for robust convergence diagnostics

### 1.3 Impact Assessment

| Metric | Current Value | Recommended | Impact |
|--------|---------------|-------------|--------|
| Samples per chain | 800 | 1500 | WAIC stability |
| Tuning samples | 400 | 800 | Better adaptation |
| Number of chains | 2 | 4 | Reliable R-hat |
| R-hat threshold | None | 1.01 | Reject bad results |

---

## 2. Marginal Likelihood Estimation Method

### 2.1 Current Implementation

**File:** `apps/bayesian/calibration.py` (lines ~380-430)

```python
def compute_marginal_likelihood(self, method: str = "waic") -> float:
    """Compute marginal likelihood estimate."""
    if method == "waic":
        waic_data = az.waic(self._trace)
        return waic_data.elpd_waic  # This is NOT marginal likelihood!
    elif method == "harmonic_mean":
        log_lik = self._trace.log_likelihood[self._obs_name].values
        return -np.log(np.mean(np.exp(-log_lik)))
```

### 2.2 Critical Problem: WAIC ≠ Marginal Likelihood

#### Mathematical Distinction

| Quantity | Symbol | Definition | What Code Returns |
|----------|--------|------------|-------------------|
| **Marginal Likelihood** | $Z = P(D\|M)$ | $\int P(D\|\theta)P(\theta\|M)d\theta$ | ❌ Not computed |
| **ELPD (WAIC)** | - | $\sum_i \log P(y_i\|y_{-i})$ | ✅ This is returned |
| **True Bayes Factor** | $BF_{12}$ | $Z_1 / Z_2$ | ❌ Not computed |

#### What Your Code Actually Computes

```
Your "log_bayes_factor" = WAIC_EB - WAIC_Timoshenko
                        ≈ Δ(out-of-sample predictive accuracy)
                        ≠ log(P(D|M_EB) / P(D|M_Timo))
```

#### Why This Matters

WAIC measures **predictive accuracy**, not **evidence**. While often correlated with Bayes factors:

1. **No proper Occam's razor:** WAIC penalizes effective parameters, not prior volume
2. **Different scale:** WAIC is in log-predictive-density units, not log-probability
3. **Can disagree:** For non-nested models, WAIC and Bayes factors can favor different models

### 2.3 Alternative Methods Available

| Method | Accuracy | Computational Cost | Implementation Difficulty |
|--------|----------|-------------------|--------------------------|
| WAIC (current) | ⚠️ Proxy only | Low | Already done |
| Harmonic Mean | 🔴 High variance | Low | Easy |
| **Bridge Sampling** | ✅ Gold standard | Medium | Moderate |
| SMC (Sequential Monte Carlo) | ✅ Excellent | High | Complex |
| Laplace Approximation | ⚠️ Assumes Gaussian | Very Low | Easy |

**Recommendation:** Implement Bridge Sampling (Gronau et al., 2017)

---

## 3. Physics Likelihood Function

### 3.1 Current Implementation

**File:** `apps/bayesian/calibration.py`

```python
def _build_euler_bernoulli_model(self, data: SyntheticDataset) -> pm.Model:
    with pm.Model() as model:
        # Priors
        E = pm.LogNormal("elastic_modulus", mu=26.07, sigma=0.05)
        sigma = pm.HalfNormal("sigma", sigma=1e-6)  # ✅ σ IS learned!
        
        # Physics model
        predictions = self._compute_eb_deflection(E, data)
        
        # Likelihood
        pm.Normal(
            "observations",
            mu=predictions,
            sigma=sigma,  # ✅ Using learned σ
            observed=data.displacements
        )
    return model
```

### 3.2 Positive Finding: σ is Correctly Learned

✅ **Good Practice:** The observation noise σ is treated as a nuisance parameter that MCMC learns from data.

This is the **correct approach** because:
- Fixed σ makes model selection hyper-sensitive to small discrepancies
- Learned σ allows the model to adapt to actual noise levels
- Results in more stable Bayes factors

### 3.3 Potential Improvement: Heteroscedastic Errors

**Current Assumption:** Homoscedastic (constant variance) errors
$$y_i \sim \mathcal{N}(\mu_i, \sigma^2)$$

**Physical Reality:** FEM discretization errors are often heteroscedastic:
- Larger errors near boundaries (cantilever root)
- Smaller errors in beam middle
- Error magnitude scales with deflection

**Impact:** Low to Medium - current homoscedastic model is acceptable but not optimal

---

## 4. Prior Distributions

### 4.1 Current Priors

**Euler-Bernoulli (2 parameters):**

| Parameter | Distribution | Parameters |
|-----------|--------------|------------|
| E (elastic modulus) | LogNormal | μ=26.07, σ=0.05 |
| σ (noise) | HalfNormal | σ=1e-6 |

**Timoshenko (3 parameters):**

| Parameter | Distribution | Parameters |
|-----------|--------------|------------|
| E (elastic modulus) | LogNormal | μ=26.07, σ=0.05 |
| ν (Poisson ratio) | Normal | μ=0.3, σ=0.03 |
| σ (noise) | HalfNormal | σ=1e-6 |

### 4.2 Asymmetric Occam's Razor Penalty

#### The Problem

Bayesian model selection includes an implicit **Occam's razor** penalty:

$$\log P(D|M) = \underbrace{\log P(D|\hat{\theta}, M)}_{\text{Goodness of Fit}} - \underbrace{\text{Complexity Penalty}}_{\text{Prior Volume}}$$

The complexity penalty is proportional to the **prior volume** in parameter space.

#### Quantifying the Asymmetry

| Model | Parameters | Prior Volume Contribution |
|-------|------------|--------------------------|
| Euler-Bernoulli | E, σ | $V_{EB}$ |
| Timoshenko | E, ν, σ | $V_T = V_{EB} \times V_\nu$ |

With current priors for ν ~ Normal(0.3, 0.03):
- Effective range: [0.21, 0.39] (±3σ)
- Prior "volume": ~0.18
- **Occam penalty:** log(0.18) ≈ **-1.7 nats**

**Impact:** Timoshenko starts with a ~1.7 nat disadvantage purely from having an extra parameter, regardless of data fit.

### 4.3 Prior Width Recommendations

| Parameter | Current σ | Recommended σ | Rationale |
|-----------|-----------|---------------|-----------|
| E | 0.05 | 0.05 | Keep - appropriate uncertainty |
| ν | 0.03 | 0.05 | Widen - reduce Occam asymmetry |
| σ | 1e-6 | 1e-6 | Keep - matches expected noise |

---

## 5. Data Normalization

### 5.1 Current State

**No normalization implemented.** Data is used directly in physical units.

### 5.2 Scale Analysis

| Quantity | Typical Value | Order of Magnitude |
|----------|---------------|-------------------|
| Displacement (slender, L/h=50) | ~1e-3 m | $10^{-3}$ |
| Displacement (thick, L/h=5) | ~1e-6 m | $10^{-6}$ |
| Strain | ~1e-6 | $10^{-6}$ |
| Elastic modulus | 2.1e11 Pa | $10^{11}$ |

**Dynamic range:** 14+ orders of magnitude between smallest displacement and E!

### 5.3 Numerical Issues

This scale mismatch causes:

1. **Gradient Instability in NUTS**
   - Hamiltonian dynamics requires balanced scales
   - Large scale differences → poor mass matrix estimation

2. **Covariance Matrix Conditioning**
   - Posterior covariance has condition number ~$10^{14}$
   - Near numerical precision limits

3. **Step Size Adaptation Problems**
   - Optimal step size differs by orders of magnitude per parameter
   - Leads to inefficient sampling or divergences

### 5.4 Recommended Normalization Strategy

| Quantity | Normalization | Normalized Range |
|----------|---------------|------------------|
| Displacements | $y_{norm} = y / \max(\|y\|)$ | [-1, 1] |
| E | $E_{norm} = E / E_{nominal}$ | ~[0.9, 1.1] |
| σ | $\sigma_{norm} = \sigma / \max(\|y\|)$ | ~[0, 0.1] |

---

## 6. Additional Observations

### 6.1 Warning Messages Analysis

The following warnings appear during runs:

```
UserWarning: For one or more samples the posterior variance of the 
log predictive densities exceeds 0.4. This could be indication of 
WAIC starting to fail.
```

**Cause:** Insufficient samples or poor posterior approximation  
**Impact:** WAIC estimates unreliable  
**Fix:** Increase samples, implement bridge sampling

```
UserWarning: Estimated shape parameter of Pareto distribution is 
greater than 0.69 for one or more samples.
```

**Cause:** Influential observations affecting LOO-CV  
**Impact:** Some data points dominate the posterior  
**Fix:** Consider robust likelihood (Student-t) or remove outliers

### 6.2 Edge Case: L/h = 50

**Observed:** Log BF = -0.031 (nearly zero)  
**Expected:** Clear preference for Euler-Bernoulli (simpler model, both fit equally well)

**Root Cause:** When models fit equally well, the Occam penalty should favor E-B, but:
1. WAIC doesn't properly compute Occam penalty
2. Prior asymmetry partially compensates (wrongly)

**Current Fix:** Inconclusive threshold (|log BF| < 0.5 → prefer simpler model)  
**Proper Fix:** Implement true marginal likelihood via bridge sampling

---

## 7. Summary of Findings

### Critical Issues (Must Address)

1. **WAIC is not marginal likelihood** - Bayes factors are approximations at best
2. **No data normalization** - Risk of numerical instability
3. **R-hat not enforced** - May accept non-converged results

### Medium Issues (Should Address)

4. **Insufficient samples** (800) - Increase to 1500+
5. **Only 2 chains** - Increase to 4 for reliable diagnostics
6. **Asymmetric Occam penalty** - Widen ν prior

### Minor Issues (Nice to Have)

7. **Homoscedastic noise model** - Consider heteroscedastic option
8. **No automatic retry** - Add retry logic for failed convergence

---

## References

1. Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the American Statistical Association*, 90(430), 773-795.

2. Gronau, Q. F., et al. (2017). bridgesampling: An R Package for Estimating Normalizing Constants. *Journal of Statistical Software*, 82(13), 1-29.

3. Vehtari, A., et al. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.

4. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
