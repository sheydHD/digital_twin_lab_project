# Presentation Guide: Bayesian Model Selection for Beam Theory in SHM Digital Twins

**Duration**: 20 minutes (~1 min per slide)
**Total slides**: 20

---

## Slide 1 — Title Slide

**Title**: Bayesian Model Selection for Beam Theory in Structural Health Monitoring Digital Twins

**Subtitle**: Digital Twins Lab Project — Winter Semester 2026

**Authors**: Antoni Dudij, Maksim Feldmann

**Affiliation**: RWTH Aachen University

**Visual**: University logo + a simple cantilever beam schematic.

**Notes**: No speaking needed beyond "Good morning, our topic is..."

---

## Slide 2 — Motivation: Why This Matters

**Title**: The Model Selection Problem in Digital Twins

**Content**:

- Digital twins for structural health monitoring combine physics models with real-time sensor data
- Applications: cantilever structures (bridge segments, building frames, elevated platforms)
- The beam theory used fundamentally shapes every prediction: deflections, stresses, remaining life
- Current practice: engineers pick one theory by intuition, no rigorous justification
- **Problem statement**: Can we use data to decide *which* beam theory a digital twin should use?

**Visual**: Diagram showing a cantilever beam structure → sensors → digital twin loop. Mark the "physics model" block with a question mark.

**Speaker text**:
"Digital twins for structural health monitoring rely on beam models. Our focus is cantilever-type structures — common in bridges and elevated infrastructure. Engineers typically choose Euler-Bernoulli or Timoshenko by rule of thumb. We ask: can we let the data decide, using Bayesian inference?"

---

## Slide 3 — Two Beam Theories

**Title**: Euler-Bernoulli vs. Timoshenko Beam Theory

**Content** (two-column layout):

| | Euler-Bernoulli | Timoshenko |
|---|---|---|
| Assumption | Plane sections remain perpendicular to neutral axis | Plane sections remain plane, but rotate independently |
| Shear deformation | Neglected | Included via correction factor κ |
| Deflection formula | $w(x) = -\frac{Px^2}{6EI}(3L - x)$ | $w(x) = -\frac{Px^2}{6EI}(3L - x) - \frac{Px}{\kappa GA}$ |
| Best for | Slender beams (L/h > 20) | Thick beams (L/h < 10), higher modes |
| Parameters | E | E, ν (through G and κ) |

**Visual**: Side-by-side beam cross-section diagrams showing deformed configuration — EB with sections perpendicular, Timoshenko with shear angle γ.

**Speaker text**:
"Euler-Bernoulli neglects shear — sections stay perpendicular. Timoshenko adds a shear correction term Px over κGA. For thick beams this extra term is substantial. The question is: at what geometry does it become negligible?"

---

## Slide 4 — The Shear Deformation Effect

**Title**: When Does Shear Deformation Matter?

**Image**: `outputs/figures/detailed/shear_contribution_analysis.png`

**Why this image**: It directly shows the shear-to-total deflection ratio as a function of L/h. This quantifies the physical effect we are trying to detect with Bayesian inference.

**Speaker text**:
"This plot shows the shear contribution to total tip deflection. At L/h = 5, shear accounts for roughly 15% of the deflection. By L/h = 20, it drops below 1%. This is the signal our Bayesian framework must detect from noisy data."

---

## Slide 5 — Methodology Overview

**Title**: Methodology Pipeline

**Image**: `outputs/figures/detailed/methodology_flowchart.png`

**Why this image**: Shows the complete pipeline from data generation to model selection in one diagram.

**Content** (if the image is not detailed enough, use a numbered list):

1. Generate synthetic data from 1D Timoshenko FEM ground truth (displacements + strains)
2. Calibrate Euler-Bernoulli model (infer E) via MCMC
3. Calibrate Timoshenko model (infer E, ν) via MCMC
4. Compute WAIC for each model
5. Compute log Bayes factor → select model
6. Repeat for 11 aspect ratios L/h ∈ {5, 8, 10, 12, 15, 20, 30, 50, 60, 70, 100}
7. Frequency-based natural frequency analysis for each aspect ratio

**Speaker text**:
"Our pipeline has four stages. We generate ground truth data — both displacements and surface strains — from a Timoshenko FEM, calibrate both beam theories independently via MCMC, then compare them using the Widely Applicable Information Criterion to compute Bayes factors. We repeat this across 11 aspect ratios. We also perform a frequency analysis to characterize where shear effects modify natural frequencies."

---

## Slide 6 — Ground Truth: 1D Timoshenko FEM

**Title**: Synthetic Data Generation

**Content**:

- 1D Timoshenko beam finite elements with exact stiffness matrix
- Element stiffness includes shear parameter $\Phi = \frac{12EI}{\kappa G A L_e^2}$
- DOFs per node: deflection $w$ and rotation $\theta$
- Cantilever boundary condition: $w(0) = 0$, $\theta(0) = 0$
- Point load $P = 1000$ N at the free end
- Gaussian noise added: 0.05% relative noise (σ/signal)
- Material: structural steel, E = 210 GPa, ν = 0.3, κ = 5/6
- **Data generated**: Displacements at 5 sensor locations + surface strains at element centroids
- Strain formula: $\varepsilon(x) = -\frac{h}{2} \cdot \frac{P(L-x)}{EI}$ (top‐surface axial strain)

**Visual**: Show the FEM element stiffness matrix (from `beam_fem.py`):

```
K_e = (EI / (L³(1+Φ))) × [  12      6L      -12      6L    ]
                           [  6L   (4+Φ)L²   -6L   (2-Φ)L²  ]
                           [ -12     -6L       12     -6L     ]
                           [  6L   (2-Φ)L²   -6L   (4+Φ)L²  ]
```

**Speaker text**:
"The ground truth comes from 1D Timoshenko beam elements for a cantilever configuration — fixed at x=0, point load at the free end. We extract both displacement profiles and surface strains. Using a 1D beam FEM rather than 2D plane stress FEM ensures the reference solution is exactly consistent with our analytical beam theories. The only difference between theory and ground truth is whether shear deformation is included."

---

## Slide 7 — Deflection Profiles: Thick vs. Slender

**Title**: FEM Ground Truth vs. Analytical Theories

**Image**: `outputs/figures/detailed/deflection_profiles_grid.png`

**Why this image**: Visually shows how EB and Timoshenko predictions diverge for thick beams and converge for slender beams. The audience can see the problem before you introduce the Bayesian solution.

**Speaker text**:
"Here we plot deflection profiles for several aspect ratios. At L/h = 5, the Euler-Bernoulli prediction misses the FEM ground truth significantly — the shear term is large. At L/h = 30, both theories are essentially identical. Our Bayesian framework must discriminate these cases using only 5 noisy displacement sensors."

---

## Slide 8 — Bayesian Inference Framework

**Title**: Bayesian Calibration Formulation

**Content**:

$$p(\theta | y, \mathcal{M}) = \frac{p(y | \theta, \mathcal{M}) \cdot p(\theta | \mathcal{M})}{p(y | \mathcal{M})}$$

Where:
- $\theta$: model parameters (E for EB; E, ν for Timoshenko)
- $y$: observed data (displacements or strains at sensor locations)
- $\mathcal{M}$: beam theory model
- $p(y | \mathcal{M}) = \int p(y|\theta, \mathcal{M}) \, p(\theta|\mathcal{M}) \, d\theta$ — **marginal likelihood** (model evidence)

**Likelihood** (displacement mode):

$$y_i \sim \mathcal{N}\big(w_{\mathcal{M}}(x_i; \theta),\; \sigma^2\big)$$

**Likelihood** (strain mode):

$$\varepsilon_i \sim \mathcal{N}\big(\varepsilon_{\mathcal{M}}(x_i; \theta),\; \sigma^2_\varepsilon\big)$$

**Priors**:

| Parameter | Prior |
|-----------|-------|
| E (normalized) | $\mathcal{N}(1.0, \; 0.05)$ |
| ν (Timoshenko only) | $\mathcal{N}(0.3, \; 0.05)$ |
| σ (noise) | HalfNormal(1.0) |

**Speaker text**:
"We formulate inference through Bayes' theorem. The likelihood assumes Gaussian measurement noise around the forward model prediction. We placed a Normal prior on the normalized elastic modulus centered at 1.0 with 5% relative uncertainty. The key quantity for model selection is the denominator — the marginal likelihood, or model evidence."

---

## Slide 9 — Numerical Normalization

**Title**: Normalization for Stable MCMC

**Content**:

The raw problem spans 14+ orders of magnitude:
- Deflections: ~10⁻⁵ m
- Elastic modulus: ~10¹¹ Pa
- This causes gradient instability in the NUTS sampler

**Solution**: Normalize all quantities to O(1):

$$E_{\text{norm}} = E / E_{\text{scale}}, \quad w_{\text{norm}} = w / w_{\text{scale}}$$

- $E_{\text{scale}} = 210 \times 10^9$ Pa (nominal steel)
- $w_{\text{scale}} = \max|w_{\text{observed}}|$

All sampling occurs in normalized space → denormalize posteriors after.

**Speaker text**:
"A crucial implementation detail: raw beam mechanics quantities span 14 orders of magnitude — micrometers to hundreds of gigapascals. This destroys MCMC convergence. We normalize everything to order 1 before sampling. The sampler works in normalized space, and we transform back to physical units after."

---

## Slide 10 — MCMC Sampling Configuration

**Title**: MCMC Sampling: NUTS Algorithm

**Content**:

| Setting | Value |
|---------|-------|
| Sampler | No-U-Turn Sampler (NUTS) |
| Chains | 4 |
| Warmup / tuning | 800 draws (discarded) |
| Post-warmup draws | 1500 draws per chain (kept) |
| Total draws per chain | 2300 (800 + 1500) |
| Target accept rate | 0.95 |
| Total posterior samples | 6000 (4 × 1500) |

**Convergence criteria** (Vehtari et al., 2021):
- $\hat{R} < 1.01$ (all parameters)
- ESS_bulk > 400
- ESS_tail > 200

**Achieved**: R̂ = 1.002–1.003 for all runs ✓

**Speaker text**:
"We run 4 independent MCMC chains with the NUTS sampler. Each chain draws 1500 samples after 800 warmup iterations. Convergence is verified via R-hat below 1.01 and effective sample sizes above 400. All 11 calibrations converged successfully."

---

## Slide 11 — Convergence Diagnostics

**Title**: MCMC Convergence Verification

**Image**: `outputs/figures/detailed/convergence_diagnostics.png`

**Why this image**: Demonstrates that the MCMC results are trustworthy. Shows trace plots and R-hat values. Without convergence, all downstream model selection would be meaningless.

**Speaker text**:
"This figure shows convergence diagnostics. The trace plots show well-mixed chains — no drift, no stuck regions. R-hat values are below 1.01 for all parameters across all configurations. This confirms our posterior estimates are reliable."

---

## Slide 12 — Parameter Recovery

**Title**: Parameter Recovery Analysis

**Image**: `outputs/figures/detailed/parameter_recovery_analysis.png`

**Why this image**: Validates that the Bayesian framework correctly recovers the true parameter values used to generate the data. This is a sanity check — if parameters are not recovered, model selection is unreliable.

**Speaker text**:
"Before trusting model selection, we verify parameter recovery. The posterior for E consistently covers the true value of 210 GPa across all aspect ratios. The posterior concentrates tightly, confirming the data is informative and the forward model is correctly implemented."

---

## Slide 13 — Model Comparison: WAIC

**Title**: Model Selection via WAIC

**Content**:

**Widely Applicable Information Criterion** (Watanabe, 2010):

$$\text{WAIC} = -2 \left( \widehat{\text{elpd}} - p_{\text{WAIC}} \right)$$

- $\widehat{\text{elpd}}$: expected log pointwise predictive density (fit quality)
- $p_{\text{WAIC}}$: effective number of parameters (complexity penalty)

**Log Bayes Factor approximation**:

$$\ln BF_{12} \approx \frac{1}{2}(\text{WAIC}_2 - \text{WAIC}_1)$$

**Interpretation** (Kass & Raftery, 1995):
| |ln BF| | Evidence |
|---------|----------|
| < 0.5 | Inconclusive |
| 0.5 – 1.0 | Weak |
| 1.0 – 2.3 | Moderate |
| > 2.3 | Strong |

**Speaker text**:
"We use WAIC to approximate the log Bayes factor. WAIC balances goodness-of-fit against model complexity — it naturally penalizes Timoshenko for having an extra parameter. A log Bayes factor below -2.3 means strong evidence for Timoshenko; above +2.3 means strong evidence for Euler-Bernoulli."

---

## Slide 14 — Jeffreys' Scale

**Title**: Bayes Factor Interpretation Scale

**Image**: `outputs/figures/detailed/jeffreys_scale_diagram.png`

**Why this image**: Provides the audience with a visual reference for interpreting all subsequent Bayes factor results. Anchors the quantitative results to qualitative labels.

**Speaker text**:
"This diagram shows the Jeffreys/Kass-Raftery interpretation scale we use. Our log Bayes factors range from -11.1 to +0.4, so the thick-beam cases fall in 'strong evidence for Timoshenko' while the slender-beam cases are inconclusive or weakly favor Euler-Bernoulli."

---

## Slide 15 — Main Result: Log Bayes Factor vs. L/h

**Title**: Model Selection Results Across Aspect Ratios

**Image**: `outputs/figures/detailed/bayes_factor_scale.png` or `outputs/figures/aspect_ratio_study.png`

**Why this image**: This is the central result of the entire project. It shows the log Bayes factor as a function of L/h, with the transition point clearly visible.

**Key data points to highlight**:

| L/h | Log BF | Evidence |
|-----|--------|----------|
| 5 | −11.11 | **Strong** Timoshenko |
| 8 | −7.68 | **Strong** Timoshenko |
| 10 | −4.32 | **Strong** Timoshenko |
| 15 | −2.45 | **Moderate** Timoshenko |
| 20 | +0.39 | Inconclusive / weak EB |
| 50 | +0.06 | Inconclusive |
| 100 | −0.02 | Inconclusive |

**Transition point**: L/h ≈ 19.3 (interpolated zero-crossing)

**Speaker text**:
"This is our main result. The log Bayes factor is strongly negative for thick beams — the data has overwhelming evidence for Timoshenko. The curve rises monotonically and crosses zero near L/h = 19.3. Beyond that, the Bayes factor is near zero — both models fit equally well, so Occam's razor weakly favors the simpler Euler-Bernoulli. The transition at L/h ≈ 19 is consistent with the engineering rule of thumb that shear becomes negligible around L/h = 20."

---

## Slide 16 — Model Probabilities

**Title**: Posterior Model Probabilities

**Image**: `outputs/figures/detailed/model_probability_analysis.png`

**Why this image**: Translates the log Bayes factor into probabilities. More intuitive for the audience — "Timoshenko has 99.99% probability at L/h = 5" is more impactful than "log BF = -11.1".

**Content**:

Assuming equal prior model probabilities:

$$P(\mathcal{M}_1 | y) = \frac{BF_{12}}{1 + BF_{12}}, \quad P(\mathcal{M}_2 | y) = \frac{1}{1 + BF_{12}}$$

**Speaker text**:
"Converting to posterior probabilities: at L/h = 5, Timoshenko has essentially 100% probability. The transition is sharp — around L/h = 19, probabilities are roughly 50/50. For slender beams, both models share probability equally, but Euler-Bernoulli is slightly favored because it is simpler."

---

## Slide 17 — Evidence Strength Summary

**Title**: Evidence Strength Across All Configurations

**Image**: `outputs/figures/detailed/evidence_strength_bars.png`

**Why this image**: Bar chart format makes the evidence categories immediately visible — strong/moderate/weak/inconclusive — for all 11 aspect ratios at a glance.

**Speaker text**:
"This bar chart summarizes evidence strength. Five cases show definitive Timoshenko preference. The remaining six cases are inconclusive — meaning Euler-Bernoulli is acceptable there because the simpler model explains the data just as well."

---

## Slide 18 — Transition Analysis

**Title**: Transition Region Analysis

**Image**: `outputs/figures/detailed/transition_analysis.png`

**Why this image**: Zooms into the transition region to show the exact crossover behavior and confidence around the threshold.

**Content**:

- Interpolated transition: **L/h ≈ 19.3**
- For L/h = 15: log BF = −2.45 (last case with clear Timoshenko preference)
- For L/h = 20: log BF = +0.39 (first case with EB preference)
- Transition width: approximately L/h ∈ [15, 20]

**Speaker text**:
"Zooming into the transition region: between L/h = 15 and L/h = 20, the preference switches. The interpolated crossover is at L/h ≈ 19.3. This narrow transition band means the decision boundary is well-defined — not a gradual, ambiguous zone."

---

## Slide 19 — Practical Guidelines for Digital Twins

**Title**: Decision Rules for Digital Twin Implementation

**Content** (decision flowchart or table):

```
                        Compute L/h
                           |
                    ┌──────┴──────┐
                    │             │
               L/h < 15     L/h ≥ 20
                    │             │
              Timoshenko    Euler-Bernoulli
                    │
            15 ≤ L/h < 20
                    │
         Run Bayesian model
         selection or default
         to Timoshenko (safe)
```

**Recommendations**:

1. **L/h < 15**: Use Timoshenko. Shear deformation is statistically detectable and physically significant.
2. **15 ≤ L/h < 20**: Transition zone. Run model selection or default to Timoshenko for safety.
3. **L/h ≥ 20**: Use Euler-Bernoulli. Simpler model is equally accurate. Saves computation.
4. **During operation**: Re-evaluate if geometry changes (e.g., corrosion reducing cross-section height).

**Speaker text**:
"For practitioners: if the aspect ratio is below 15, always use Timoshenko. Above 20, Euler-Bernoulli is sufficient and computationally cheaper. In the transition zone, run Bayesian model selection or default to Timoshenko as the safer choice. During digital twin operation, re-evaluate if the geometry changes — for example, corrosion reducing the effective height would lower L/h toward the Timoshenko regime."

---

## Slide 20 — Conclusions & Future Work

**Title**: Conclusions

**Content**:

**Achieved**:
- Implemented complete Bayesian model selection pipeline for EB vs. Timoshenko beam theories
- Identified transition point at **L/h ≈ 19.3** from data-driven evidence
- Result is consistent with engineering rule of thumb (L/h ~ 20) but now has quantitative, probabilistic justification
- All MCMC chains converged (R̂ < 1.01, ESS > 400) across all 11 configurations
- Provided actionable decision rules for digital twin initialization
- Supports both **displacement-based** and **strain-based** calibration
- Analytical frequency analysis characterizing natural frequency divergence across L/h

**Key scientific contribution**:
- Replaced heuristic model choice with rigorous, evidence-based selection
- Occam's razor naturally penalizes Timoshenko's extra parameter when shear is negligible
- Framework handles multiple observable types (displacements, strains) in a unified probabilistic setting

**Limitations & future work**:
- Current study: static cantilever loading, single material (steel), single boundary condition
- Extend to dynamic loading / higher vibration modes with Bayesian frequency-domain calibration
- Extend to composite materials where shear effects are larger
- Apply to real sensor data from monitored structures
- Implement online model switching during digital twin operation

**Speaker text**:
"To conclude: we built a Bayesian model selection framework that replaces heuristic beam theory choice with data-driven evidence. The transition occurs at L/h ≈ 19.3 — confirming the engineering intuition of L/h ~ 20, but now with probabilistic justification. The framework supports both displacement and strain observations. Future work should extend to dynamic loading, composite materials, and real sensor data from monitored structures. Thank you."

---

## Appendix — Figure File Reference

| Slide | Figure file | Purpose |
|-------|-------------|---------|
| 4 | `outputs/figures/detailed/shear_contribution_analysis.png` | Shear-to-total ratio vs L/h |
| 5 | `outputs/figures/detailed/methodology_flowchart.png` | Pipeline overview |
| 7 | `outputs/figures/detailed/deflection_profiles_grid.png` | EB vs Timo vs FEM deflections |
| 11 | `outputs/figures/detailed/convergence_diagnostics.png` | Trace plots, R-hat |
| 12 | `outputs/figures/detailed/parameter_recovery_analysis.png` | Posterior vs true E |
| 14 | `outputs/figures/detailed/jeffreys_scale_diagram.png` | Bayes factor scale |
| 15 | `outputs/figures/detailed/bayes_factor_scale.png` | Main result: log BF vs L/h |
| 16 | `outputs/figures/detailed/model_probability_analysis.png` | P(model | data) vs L/h |
| 17 | `outputs/figures/detailed/evidence_strength_bars.png` | Evidence categories per L/h |
| 18 | `outputs/figures/detailed/transition_analysis.png` | Transition region zoom |

## Appendix — Timing Guide

| Slides | Topic | Time |
|--------|-------|------|
| 1–2 | Title + Motivation | 2 min |
| 3–4 | Beam theories + shear effect | 2 min |
| 5–7 | Methodology + data generation | 3 min |
| 8–10 | Bayesian framework + normalization + MCMC | 3 min |
| 11–12 | Convergence + parameter recovery | 2 min |
| 13–14 | WAIC + Jeffreys scale | 2 min |
| 15–18 | Results (main result + probabilities + transition) | 4 min |
| 19–20 | Guidelines + conclusions | 2 min |
| **Total** | | **20 min** |
