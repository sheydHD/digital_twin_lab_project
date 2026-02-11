# bayesian audit analysis

status: **all issues resolved**

audit date: February 3, 2026. This document records the original findings. All critical and medium issues have been addressed in v1.0.0+.

## original findings

### critical (fixed)

1. **WAIC used as marginal likelihood** — WAIC measures predictive accuracy, not model evidence. Bayes factors computed from WAIC differences are not true Bayes factors.
   - fix: implemented bridge sampling (Meng & Wong, 1996) for true marginal likelihood estimation. WAIC kept as diagnostic only.

2. **no data normalization** — raw quantities span 14+ orders of magnitude (E ~ 10^11, w ~ 10^-5), causing MCMC instability.
   - fix: added normalization module. E divided by E_scale (210e9), displacements divided by max|w_obs|. All sampling in O(1) space.

3. **harmonic mean estimator available as fallback** — known to have infinite variance and overestimates marginal likelihood.
   - fix: removed entirely. Bridge sampling is the only method.

### medium (fixed)

4. **R-hat not enforced** — convergence diagnostics computed but not checked.
   - fix: convergence validation added to calibration pipeline.

5. **insufficient samples** — 800 draws x 2 chains = 1600 total.
   - fix: config updated. current defaults produce reliable results for this problem size.

6. **asymmetric Occam penalty** — tight nu prior (sigma=0.03) penalizes Timoshenko unfairly.
   - fix: prior widened (sigma=0.05), reducing the asymmetry.

### minor (accepted)

7. **homoscedastic noise model** — assumes constant variance across all sensors. Physical errors may vary spatially. Accepted as adequate for this application.

## methods comparison

| method | status | purpose |
|--------|--------|---------|
| bridge sampling | active | marginal likelihood (model selection) |
| WAIC | active | diagnostic only |
| LOO-CV | removed | redundant with bridge sampling |
| harmonic mean | removed | unstable, replaced by bridge sampling |

## references

1. Kass, R.E., & Raftery, A.E. (1995). Bayes factors. *JASA*, 90(430), 773-795.
2. Meng, X.-L., & Wong, W.H. (1996). Simulating ratios of normalizing constants via a simple identity. *Statistica Sinica*, 6, 831-860.
3. Gronau, Q.F., et al. (2017). bridgesampling: An R Package for Estimating Normalizing Constants. *JSS*, 82(13), 1-29.
