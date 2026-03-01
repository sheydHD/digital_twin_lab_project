# Parameters

| Field        | Value                                       |
|--------------|---------------------------------------------|
| **Author**   | Antoni Dudij, Maksim Feldmann — RWTH Aachen |
| **Status**   | Review                                      |
| **Last Updated** | 2026-03-01                              |

---

## TL;DR

This document defines every parameter that enters the Bayesian inference loop: the two or three calibrated unknowns per model, their prior distributions in both YAML-config and normalised-MCMC form, the sign conventions for loads and deflections, and the full parametric study grid from $L/h = 5$ to $L/h = 100$ with empirical log Bayes factors.

---

## 1. Calibrated Parameters

The pipeline estimates material properties from synthetic displacement and strain sensor data. The two models differ by one parameter.

**Euler-Bernoulli** — 2 free parameters:

| Parameter | Symbol | Nominal Value | Role |
|---|---|---|---|
| Elastic modulus | $E$ | 210 GPa | Controls bending stiffness; deflection $\propto 1/E$ |
| Observation noise | $\sigma$ | ~0.05% of signal | Gaussian measurement uncertainty |

**Timoshenko** — 3 free parameters:

| Parameter | Symbol | Nominal Value | Role |
|---|---|---|---|
| Elastic modulus | $E$ | 210 GPa | Bending stiffness |
| Poisson's ratio | $\nu$ | 0.3 | Determines shear modulus $G = E / (2(1+\nu))$ |
| Observation noise | $\sigma$ | ~0.05% of signal | Gaussian measurement uncertainty |

Timoshenko carries one extra parameter ($\nu$). The bridge sampling marginal likelihood naturally penalises this extra complexity when $\nu$ does not improve the fit to data — Bayesian Occam's razor in action (see [bayesian-glossary.md](bayesian-glossary.md#16-occams-razor-bayesian)).

### Calibration Observable Types

Two types of sensor data can drive calibration, selectable via `data_type` in `calibrate()`:

| Data Type | Observable | Forward Model | Typical Use |
|---|---|---|---|
| `displacement` (default) | $w(x_i)$ | Analytical deflection formula | Primary calibration |
| `strain` | $\varepsilon(x_i)$ | $\varepsilon = -(h/2) \cdot P(L-x)/(EI)$ | Surface strain gauge setups |

Both EB and Timoshenko share the same bending-strain formula because shear deformation does not produce axial strain in the Timoshenko model.

---

## 2. Prior Distributions

Priors are defined in `configs/default_config.yaml` (physical units) but all MCMC sampling occurs in a normalised coordinate system where every quantity is $\mathcal{O}(1)$.

| Parameter | Config Definition | Normalised MCMC Prior | Scale Factor |
|---|---|---|---|
| $E$ | LogNormal($\mu=26.07$, $\sigma=0.05$) | $\mathcal{N}(1.0,\ 0.05)$ | $E_{scale} = 210 \times 10^9$ Pa |
| $\nu$ | $\mathcal{N}(0.3,\ 0.03)$ | $\mathcal{N}(0.3,\ 0.03)$ | (dimensionless, no scaling) |
| $\sigma$ | HalfNormal($\sigma=10^{-6}$) | HalfNormal($1.0$) | $w_{scale} = \max\lvert\mathbf{w}_{obs}\rvert$ |

The $\sigma$ prior is tight relative to the signal-to-noise ratio of the synthetic data, which reflects the known noise level of the simulator (`noise_fraction = 5 \times 10^{-4}$). Using a looser prior would not change the MAP estimate materially but would increase bridge sampling variance.

### Posterior Recovery

Across all aspect ratios studied, the posterior for $E$ consistently recovers the true value with:
- $\hat{R} = 1.002$–$1.003$ (well converged)
- $\text{ESS} = 1{,}250$–$1{,}650$ (ample independent samples)
- Posterior mean within 0.5% of 210 GPa

---

## 3. Sign Conventions

These conventions are enforced consistently across `base_beam.py`, `calibration.py`, and all test fixtures.

| Quantity | Convention |
|---|---|
| Applied load $P$ | Positive = downward |
| Deflection $w$ | Negative = downward (positive $P$ → negative $w$) |
| $y$-coordinate | Positive = downward from neutral axis |
| Bending strain | $\varepsilon = -y \cdot M(x)/(EI)$; tension at bottom face for positive moment |
| $\ln B_{EB/Timo}$ | $\ln p(\mathbf{y}\mid M_{EB}) - \ln p(\mathbf{y}\mid M_{Timo})$ |
| Negative $\ln B$ | Favours Timoshenko |
| Positive $\ln B$ | Favours Euler-Bernoulli |
| $\lvert\ln B\rvert < 0.5$ | Inconclusive |

---

## 4. Prior Distribution Types

| Distribution | Applied To | Rationale |
|---|---|---|
| $\mathcal{N}(1.0,\ 0.05)$ | $E$ (normalised) | Symmetric; 5% relative uncertainty around steel nominal |
| $\mathcal{N}(0.3,\ 0.03)$ | $\nu$ (Timoshenko) | Steel $\nu \in [0.25, 0.35]$; tight prior limits Occam penalty |
| HalfNormal$(1.0)$ | $\sigma$ (normalised) | Forces positive noise; concentrates mass near zero |
| $\mathcal{N}(\hat{w},\ \sigma^2)$ | Likelihood | Gaussian sensor noise model |

---

## 5. Parametric Study Grid

Fixed parameters: $L = 1.0$ m, $b = 0.1$ m, $P = 1{,}000$ N, $E = 210$ GPa, $\nu = 0.3$, $\kappa = 5/6$.
Varied: $L/h$, which fixes beam height as $h = L/(L/h)$.

| $L/h$ | $h$ [m] | Beam Type | $\ln B_{EB/Timo}$ | Recommendation |
|---|---|---|---|---|
| 5 | 0.200 | Very thick | −11.11 | Timoshenko |
| 8 | 0.125 | Thick | −7.68 | Timoshenko |
| 10 | 0.100 | Moderate | −4.32 | Timoshenko |
| 12 | 0.083 | Moderate | −3.91 | Timoshenko |
| 15 | 0.067 | Transition zone | −2.45 | Timoshenko |
| 20 | 0.050 | Slender | +0.39 | Euler-Bernoulli |
| 30 | 0.033 | Slender | +0.08 | Euler-Bernoulli |
| 50 | 0.020 | Very slender | +0.06 | Euler-Bernoulli |
| 60 | 0.017 | Very slender | −0.06 | Inconclusive |
| 70 | 0.014 | Very slender | +0.18 | Euler-Bernoulli |
| 100 | 0.010 | Very slender | −0.02 | Inconclusive |

**Transition point: $L/h \approx 19.2$** (linear interpolation of the $\ln B = 0$ zero-crossing between $L/h = 15$ and $L/h = 20$).

The study grid is denser in the $L/h = 10$–$20$ region to resolve the transition accurately. For very slender beams ($L/h \geq 50$) the shear term $PL/(\kappa GA)$ is negligible relative to the bending term $PL^3/(3EI)$, and both models predict essentially identical deflections. The resulting $|\ln B| < 0.1$ correctly classifies this region as inconclusive rather than strongly favouring either theory.

---

## 6. Frequency Analysis Parameters

The pipeline also computes analytical natural frequencies for both theories as a physics-based cross-check. The EB natural frequency for cantilever mode $n$ is:

$$f_n^{EB} = \frac{\lambda_n^2}{2\pi L^2} \sqrt{\frac{EI}{\rho A}}$$

where $\lambda_n$ are the eigenvalues of the clamped-free boundary condition ($\lambda_1 = 1.875$, $\lambda_2 = 4.694$, …).

Timoshenko frequencies include shear and rotary inertia corrections that reduce predicted values compared to EB, with the discrepancy growing for thick beams and higher modes. Results are written to `outputs/reports/frequency_analysis.txt` by `FrequencyBasedModelSelector` in `hyperparameter_optimization.py`.

---

## References

1. Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *Journal of the American Statistical Association*, 90(430), 773–795.
2. Timoshenko, S. P. (1921). On the correction factor for shear of the differential equation for transverse vibrations of bars. *Philosophical Magazine*, 41, 744–746.
