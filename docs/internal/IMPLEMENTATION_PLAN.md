# implementation plan

status: **all tasks completed**

This document records the improvement plan from the February 2026 audit. All phases have been implemented.

## phases

### phase 1: config updates (completed)

- increased MCMC samples and tuning for stable posteriors
- widened Poisson ratio prior (sigma 0.03 -> 0.05) to reduce Occam asymmetry
- added convergence threshold settings

### phase 2: data normalization (completed)

- created `apps/bayesian/normalization.py`
- `compute_normalization_params()` computes E_scale and displacement_scale
- `NormalizationParams` dataclass holds scale factors
- all MCMC sampling in normalized O(1) space
- results denormalized automatically after sampling

### phase 3: convergence validation (completed)

- R-hat checked after sampling
- ESS thresholds enforced
- diagnostics included in calibration output

### phase 4: bridge sampling (completed)

- created `apps/bayesian/bridge_sampling.py`
- `BridgeSampler` class fits multivariate normal proposal to posterior
- iterates optimal bridge function (Meng & Wong, 1996)
- returns log marginal likelihood estimate
- integrated into `model_selection.py` for Bayes factor computation
- WAIC kept as diagnostic, no longer used for model selection

### phase 5: cleanup (completed)

- removed LOO-CV computation
- removed harmonic mean estimator
- removed dead code across 10+ files
- removed unused EulerBernoulliFEM class
- updated all documentation

## outcome

| metric | before | after |
|--------|--------|-------|
| marginal likelihood method | WAIC (proxy) | bridge sampling (exact) |
| normalization | none | E_scale, displacement_scale |
| convergence checking | unchecked | R-hat < 1.01 enforced |
| LOO-CV | computed | removed |
| harmonic mean | fallback | removed |
| L/h=50 result | inconsistent | correct (indistinguishable) |
