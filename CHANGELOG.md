# changelog

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Follows [Semantic Versioning](https://semver.org/).

## [unreleased]

### added
- bridge sampling for marginal likelihood estimation (Meng & Wong, 1996)
- Optuna-based hyperparameter optimization for priors
- frequency-based analytical model comparison

### removed
- LOO-CV computation (redundant with bridge sampling)
- harmonic mean estimator (unstable, replaced by bridge sampling)
- EulerBernoulliFEM class from beam_fem.py (unused)
- dead code across models, calibration, normalization, and utilities

### fixed
- bare expression bugs in timoshenko.py and test_normalization.py
- chained != comparison in model_selection.py

## [1.0.0] - 2026-01-17

### added
- 1D Timoshenko beam FEM for ground truth generation
- Bayesian model selection framework using PyMC
- WAIC model comparison (diagnostic)
- automated pipeline across multiple aspect ratios
- visualization and reporting system
- configuration-driven execution via YAML

### changed
- replaced 2D plane stress FEM with 1D beam FEM (100x faster, exact consistency)
- reduced default MCMC samples from 2000 to 800 per chain
- reduced noise level from 0.1% to 0.05%
- adaptive mesh sizing based on aspect ratio

### fixed
- L/h=8 now correctly favors Timoshenko (was incorrectly favoring EB with 2D FEM)
- model selection results physically accurate across all aspect ratios
- numerical instability at high aspect ratios eliminated
- memory issues on standard laptops fixed

### results
- transition aspect ratio: L/h ~ 19.2
- thick beams (L/h <= 15): strong Timoshenko preference (log BF down to -10.8)
- slender beams (L/h >= 20): EB preference
## [0.1.0] - 2025-12-15

### added
- initial project structure
- basic beam theory implementations (EB and Timoshenko)
- 2D plane stress FEM (deprecated in v1.0.0)
- PyMC integration for Bayesian calibration

### known issues
- 2D FEM has systematic stiffness bias vs beam theories
- non-monotonic model selection at intermediate aspect ratios
