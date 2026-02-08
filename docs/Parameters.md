# Parameter explanation

## 1. Calibrated Parameters

**Euler-Bernoulli** — 2 parameters:

| Parameter | Symbol | True value | Role |
|-----------|--------|------------|------|
| Elastic modulus | E | 210 GPa | Stiffness. Deflection ∝ 1/E |
| Observation noise | σ | ~0.05% of signal | Measurement uncertainty |

**Timoshenko** — 3 parameters:

| Parameter | Symbol | True value | Role |
|-----------|--------|------------|------|
| Elastic modulus | E | 210 GPa | Stiffness |
| Poisson's ratio | ν | 0.3 | Controls shear modulus G = E/(2(1+ν)) |
| Observation noise | σ | ~0.05% of signal | Measurement uncertainty |

Timoshenko has 1 extra parameter (ν). WAIC penalizes this extra complexity (Occam's razor).

### Calibration Data Types

The framework supports calibration on two data types:

| Data type | Observable | Forward model | Use case |
|-----------|-----------|---------------|----------|
| **Displacement** (default) | $w(x_i)$ | Analytical deflection formula | Primary calibration — most informative |
| **Strain** | $\varepsilon(x_i)$ | $\varepsilon = -\frac{h}{2} \cdot \frac{P(L - x)}{EI}$ | Surface strain gauges — validates E independently |

Both EB and Timoshenko share the same bending-strain formula because shear deformation does not affect axial strain (it only adds rigid-body translation, not curvature).

---

## 2. Priors and Posteriors

### Priors

Config file defines LogNormal for E, but code normalizes everything to O(1) before sampling:

| Parameter | Config definition | Actual MCMC prior (normalized space) |
|-----------|-------------------|--------------------------------------|
| E | LogNormal(μ=26.07, σ=0.05) | Normal(μ=1.0, σ=0.05) |
| ν | Normal(μ=0.3, σ=0.03) | Normal(μ=0.3, σ=0.03) |
| σ | HalfNormal(σ=1e-6) | HalfNormal(σ=1.0) |

Why the difference: E is divided by E_scale=210e9, σ is divided by displacement_scale. This puts all values near 1.0 for stable MCMC.

### Posteriors

From calibration runs (all 11 L/h ratios):
- E recovers to ~210 GPa consistently
- R̂ = 1.002–1.003 (converged)
- ESS = 1250–1650 (sufficient)

---

## 3. Sign Convention

| Quantity | Convention |
|----------|-----------|
| Load P | Positive = downward |
| Deflection w | Negative = downward (positive P → negative w) |
| y-coordinate | Positive = downward from neutral axis |
| Strain ε | ε = −y·M/(EI). Tension at bottom for positive moment |
| Log Bayes factor | log BF = log p(y\|EB) − log p(y\|Timo) |
| Negative log BF | → Favors Timoshenko |
| Positive log BF | → Favors Euler-Bernoulli |
| \|log BF\| < 0.5 | → Inconclusive |

---

## 4. Distributions Used

| Distribution | Where | Why |
|---|---|---|
| Normal(1.0, 0.05) | E (normalized) | Symmetric around nominal, 5% relative uncertainty |
| Normal(0.3, 0.03) | ν (Timoshenko) | Steel ν ∈ [0.25, 0.35], tight around known value |
| HalfNormal(1.0) | σ (normalized) | Forces σ > 0, concentrates near zero (noise is small) |
| Normal (likelihood) | y ~ N(w_model, σ²) | Standard Gaussian measurement noise |

---

## 5. Grid Sampling (Parametric Study)

Not MCMC grid sampling. It's a sweep over beam geometries.

**Fixed parameters**:
- Length L = 1.0 m
- Width b = 0.1 m
- Point load P = 1000 N
- E = 210 GPa, ν = 0.3, κ = 5/6

**Varied**: aspect ratio L/h → height h = L/(L/h)

| L/h | h [m] | Beam type | Log BF | Recommendation |
|-----|-------|-----------|--------|----------------|
| 5 | 0.200 | Very thick | −11.11 | Timoshenko |
| 8 | 0.125 | Thick | −7.68 | Timoshenko |
| 10 | 0.100 | Moderate | −4.32 | Timoshenko |
| 12 | 0.083 | Moderate | −3.91 | Timoshenko |
| 15 | 0.067 | Transition | −2.45 | Timoshenko |
| 20 | 0.050 | Slender | +0.39 | Euler-Bernoulli |
| 30 | 0.033 | Slender | +0.08 | Euler-Bernoulli |
| 50 | 0.020 | Very slender | +0.06 | Euler-Bernoulli |
| 60 | 0.017 | Very slender | −0.06 | Euler-Bernoulli |
| 70 | 0.014 | Very slender | +0.18 | Euler-Bernoulli |
| 100 | 0.010 | Very slender | −0.02 | Euler-Bernoulli |

**Transition point**: L/h ≈ 19.3 (interpolated zero-crossing)

Grid is denser around L/h = 10–20 to resolve the transition region precisely.

---

## 6. Frequency Analysis

The pipeline also performs analytical natural frequency analysis for both beam theories.

**EB natural frequency** (mode n):

$$f_n^{EB} = \frac{\lambda_n^2}{2\pi L^2} \sqrt{\frac{EI}{\rho A}}$$

**Timoshenko natural frequency** includes shear and rotary inertia corrections, which lower the natural frequencies compared to EB — especially for thick beams and higher modes.

The frequency analysis report is generated automatically and saved to `outputs/reports/frequency_analysis.txt`.

**Key takeaway**: The divergence between EB and Timoshenko natural frequencies increases with mode number and decreases with L/h, consistent with the static deflection transition at L/h ≈ 19.3.
