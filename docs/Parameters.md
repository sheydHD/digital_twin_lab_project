# parameters

## 1. calibrated parameters

**Euler-Bernoulli** — 2 parameters:

| parameter | symbol | true value | role |
|-----------|--------|------------|------|
| elastic modulus | E | 210 GPa | stiffness, deflection proportional to 1/E |
| observation noise | sigma | ~0.05% of signal | measurement uncertainty |

**Timoshenko** — 3 parameters:

| parameter | symbol | true value | role |
|-----------|--------|------------|------|
| elastic modulus | E | 210 GPa | stiffness |
| Poisson's ratio | nu | 0.3 | controls shear modulus G = E/(2(1+nu)) |
| observation noise | sigma | ~0.05% of signal | measurement uncertainty |

Timoshenko has 1 extra parameter (nu). Bridge sampling marginal likelihoods naturally penalize this extra complexity when it doesn't improve fit (Occam's razor).

### calibration data types

| data type | observable | forward model | use case |
|-----------|-----------|---------------|----------|
| displacement (default) | w(x_i) | analytical deflection formula | primary calibration |
| strain | epsilon(x_i) | epsilon = -(h/2) * P(L-x) / (EI) | surface strain gauges |

Both EB and Timoshenko share the same bending-strain formula because shear deformation does not affect axial strain.

## 2. priors and posteriors

### priors

Config defines LogNormal for E, but code normalizes to O(1) before sampling:

| parameter | config definition | actual MCMC prior (normalized) |
|-----------|-------------------|-------------------------------|
| E | LogNormal(mu=26.07, sigma=0.05) | Normal(mu=1.0, sigma=0.05) |
| nu | Normal(mu=0.3, sigma=0.03) | Normal(mu=0.3, sigma=0.03) |
| sigma | HalfNormal(sigma=1e-6) | HalfNormal(sigma=1.0) |

Why the difference: E is divided by E_scale=210e9, sigma is divided by displacement_scale. This puts all values near 1.0 for stable MCMC.

### posteriors

From calibration runs (all L/h ratios):
- E recovers to ~210 GPa consistently
- R-hat = 1.002-1.003 (converged)
- ESS = 1250-1650 (sufficient)

## 3. sign convention

| quantity | convention |
|----------|-----------|
| load P | positive = downward |
| deflection w | negative = downward (positive P -> negative w) |
| y-coordinate | positive = downward from neutral axis |
| strain epsilon | epsilon = -y * M/(EI). Tension at bottom for positive moment |
| log Bayes factor | log BF = log p(y|EB) - log p(y|Timo) |
| negative log BF | favors Timoshenko |
| positive log BF | favors Euler-Bernoulli |
| |log BF| < 0.5 | inconclusive |

## 4. distributions used

| distribution | where | why |
|---|---|---|
| Normal(1.0, 0.05) | E (normalized) | symmetric around nominal, 5% relative uncertainty |
| Normal(0.3, 0.03) | nu (Timoshenko) | steel nu in [0.25, 0.35] |
| HalfNormal(1.0) | sigma (normalized) | forces sigma > 0, concentrates near zero |
| Normal (likelihood) | y ~ N(w_model, sigma^2) | Gaussian measurement noise |

## 5. parametric study grid

Fixed parameters: L = 1.0 m, b = 0.1 m, P = 1000 N, E = 210 GPa, nu = 0.3, kappa = 5/6.

Varied: aspect ratio L/h, which sets h = L/(L/h).

| L/h | h [m] | beam type | log BF | recommendation |
|-----|-------|-----------|--------|----------------|
| 5 | 0.200 | very thick | -11.11 | Timoshenko |
| 8 | 0.125 | thick | -7.68 | Timoshenko |
| 10 | 0.100 | moderate | -4.32 | Timoshenko |
| 12 | 0.083 | moderate | -3.91 | Timoshenko |
| 15 | 0.067 | transition | -2.45 | Timoshenko |
| 20 | 0.050 | slender | +0.39 | Euler-Bernoulli |
| 30 | 0.033 | slender | +0.08 | Euler-Bernoulli |
| 50 | 0.020 | very slender | +0.06 | Euler-Bernoulli |
| 60 | 0.017 | very slender | -0.06 | Euler-Bernoulli |
| 70 | 0.014 | very slender | +0.18 | Euler-Bernoulli |
| 100 | 0.010 | very slender | -0.02 | Euler-Bernoulli |

Transition point: L/h ~ 19.3 (interpolated zero-crossing).

Grid is denser around L/h = 10-20 to resolve the transition region.

## 6. frequency analysis

The pipeline also runs analytical natural frequency analysis for both beam theories.

EB natural frequency (mode n):

$$f_n^{EB} = \frac{\lambda_n^2}{2\pi L^2} \sqrt{\frac{EI}{\rho A}}$$

Timoshenko natural frequencies include shear and rotary inertia corrections that lower frequencies compared to EB, especially for thick beams and higher modes.

Results saved to `outputs/reports/frequency_analysis.txt`.
