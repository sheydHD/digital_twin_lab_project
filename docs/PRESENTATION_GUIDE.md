# presentation guide

20 minutes, ~1 min per slide, 20 slides.

Authors: Antoni Dudij, Maksim Feldmann. RWTH Aachen University, Digital Twins Lab, WS 2026.

## slide 1 — title

Bayesian Model Selection for Beam Theory in Structural Health Monitoring Digital Twins.

Visual: university logo + cantilever beam schematic.

## slide 2 — motivation

The model selection problem in digital twins.

Content:
- digital twins for SHM combine physics models with sensor data
- the beam theory used shapes all predictions: deflections, stresses, remaining life
- current practice: engineers pick one theory by intuition
- question: can we use data to decide which beam theory a digital twin should use?

Visual: cantilever beam -> sensors -> digital twin loop. Mark the physics model block with a question mark.

Speaker notes: "Digital twins for structural health monitoring rely on beam models. Engineers typically choose Euler-Bernoulli or Timoshenko by rule of thumb. We ask: can we let the data decide, using Bayesian inference?"

## slide 3 — two beam theories

Euler-Bernoulli vs Timoshenko.

| | Euler-Bernoulli | Timoshenko |
|---|---|---|
| assumption | sections stay perpendicular | sections remain plane, rotate independently |
| shear deformation | neglected | included via correction factor kappa |
| deflection | $w(x) = -\frac{Px^2}{6EI}(3L - x)$ | $w(x) = -\frac{Px^2}{6EI}(3L - x) - \frac{Px}{\kappa GA}$ |
| best for | slender beams (L/h > 20) | thick beams (L/h < 10) |
| parameters | E | E, nu |

Visual: side-by-side deformed cross-sections showing EB perpendicular vs Timoshenko with shear angle.

Speaker notes: "Euler-Bernoulli neglects shear — sections stay perpendicular. Timoshenko adds a shear correction term. For thick beams this extra term is substantial. The question is: at what geometry does it become negligible?"

## slide 4 — shear deformation effect

When does shear deformation matter?

Image: `outputs/figures/detailed/shear_contribution_analysis.png`

Shows shear-to-total deflection ratio as a function of L/h.

Speaker notes: "At L/h = 5, shear accounts for roughly 15% of deflection. By L/h = 20, it drops below 1%. This is the signal our Bayesian framework must detect from noisy data."

## slide 5 — methodology overview

Image: `outputs/figures/detailed/methodology_flowchart.png`

Pipeline steps:
1. generate synthetic data from 1D Timoshenko FEM (displacements + strains)
2. calibrate EB model (infer E) via MCMC
3. calibrate Timoshenko model (infer E, nu) via MCMC
4. compute marginal likelihoods via bridge sampling
5. compute log Bayes factor -> select model
6. repeat for 11 aspect ratios
7. frequency-based natural frequency analysis

Speaker notes: "We generate ground truth from Timoshenko FEM, calibrate both theories via MCMC, then compare using bridge sampling marginal likelihoods to compute Bayes factors. We repeat across 11 aspect ratios."

## slide 6 — ground truth: 1D Timoshenko FEM

Synthetic data generation.

Content:
- 1D Timoshenko beam elements with exact stiffness matrix
- shear parameter $\Phi = \frac{12EI}{\kappa G A L_e^2}$
- DOFs per node: deflection w, rotation theta
- cantilever: w(0) = 0, theta(0) = 0, point load P = 1000 N at free end
- 0.05% relative Gaussian noise
- material: steel, E = 210 GPa, nu = 0.3, kappa = 5/6
- data: deflections at 5 sensors + surface strains at element centroids

Speaker notes: "Ground truth comes from 1D Timoshenko beam elements. Using 1D beam FEM rather than 2D plane stress FEM ensures exact consistency with our analytical beam theories."

## slide 7 — deflection profiles

FEM ground truth vs analytical theories.

Image: `outputs/figures/detailed/deflection_profiles_grid.png`

Speaker notes: "At L/h = 5, Euler-Bernoulli misses the FEM ground truth significantly. At L/h = 30, both theories are essentially identical. Our framework must discriminate these cases using only 5 noisy sensors."

## slide 8 — Bayesian inference framework

$$p(\theta | y, \mathcal{M}) = \frac{p(y | \theta, \mathcal{M}) \cdot p(\theta | \mathcal{M})}{p(y | \mathcal{M})}$$

- theta: model parameters (E for EB; E, nu for Timoshenko)
- y: observed data
- M: beam theory model
- p(y|M): marginal likelihood (model evidence)

Likelihood: $y_i \sim \mathcal{N}(w_{\mathcal{M}}(x_i; \theta), \sigma^2)$

Priors: E_norm ~ N(1.0, 0.05), nu ~ N(0.3, 0.05), sigma ~ HalfNormal(1.0)

Speaker notes: "The key quantity for model selection is the marginal likelihood in the denominator. We estimate it via bridge sampling."

## slide 9 — normalization

Raw problem spans 14+ orders of magnitude. Deflections ~10^-5 m, elastic modulus ~10^11 Pa. This destroys MCMC convergence.

Solution: normalize to O(1). E_norm = E / E_scale, w_norm = w / w_scale.

Speaker notes: "A crucial implementation detail. We normalize everything to order 1 before sampling and transform back after."

## slide 10 — MCMC configuration

| setting | value |
|---------|-------|
| sampler | NUTS |
| chains | 4 |
| warmup | 800 (discarded) |
| post-warmup draws | 1500 per chain |
| target accept | 0.95 |
| total posterior samples | 6000 |

Convergence: R-hat < 1.01, ESS_bulk > 400, ESS_tail > 200. Achieved: R-hat = 1.002-1.003 for all runs.

Speaker notes: "We run 4 MCMC chains with NUTS. All 11 calibrations converged successfully."

## slide 11 — convergence diagnostics

Image: `outputs/figures/detailed/convergence_diagnostics.png`

Speaker notes: "Trace plots show well-mixed chains. R-hat values below 1.01 for all parameters. Posterior estimates are reliable."

## slide 12 — parameter recovery

Image: `outputs/figures/detailed/parameter_recovery_analysis.png`

Speaker notes: "The posterior for E consistently covers the true value of 210 GPa. The framework correctly recovers parameters."

## slide 13 — model comparison via bridge sampling

Marginal likelihood estimated via bridge sampling (Meng & Wong, 1996):

$$p(y | \mathcal{M}) = \int p(y | \theta, \mathcal{M}) \, p(\theta | \mathcal{M}) \, d\theta$$

Log Bayes factor: $\ln BF = \ln p(y|\text{EB}) - \ln p(y|\text{Timo})$

Interpretation (Kass & Raftery, 1995):

| |ln BF| | evidence |
|---------|----------|
| < 0.5 | inconclusive |
| 0.5 - 1.0 | weak |
| 1.0 - 2.3 | moderate |
| > 2.3 | strong |

Speaker notes: "We use bridge sampling to estimate marginal likelihoods, then compute the log Bayes factor. Negative values favor Timoshenko, positive favor EB."

## slide 14 — Jeffreys' scale

Image: `outputs/figures/detailed/jeffreys_scale_diagram.png`

Speaker notes: "Our log Bayes factors range from -11.1 to +0.4, so thick-beam cases fall in 'strong evidence for Timoshenko' while slender cases are inconclusive."

## slide 15 — main result: log Bayes factor vs L/h

Image: `outputs/figures/detailed/bayes_factor_scale.png` or `outputs/figures/aspect_ratio_study.png`

| L/h | log BF | evidence |
|-----|--------|----------|
| 5 | -11.11 | strong Timoshenko |
| 8 | -7.68 | strong Timoshenko |
| 10 | -4.32 | strong Timoshenko |
| 15 | -2.45 | moderate Timoshenko |
| 20 | +0.39 | inconclusive / weak EB |
| 50 | +0.06 | inconclusive |
| 100 | -0.02 | inconclusive |

Transition: L/h ~ 19.3.

Speaker notes: "This is our main result. The log Bayes factor crosses zero near L/h = 19.3. The transition at L/h ~ 19 is consistent with the engineering rule of thumb that shear becomes negligible around L/h = 20."

## slide 16 — model probabilities

Image: `outputs/figures/detailed/model_probability_analysis.png`

Assuming equal prior model probabilities:

$$P(\mathcal{M}_1 | y) = \frac{BF_{12}}{1 + BF_{12}}$$

Speaker notes: "At L/h = 5, Timoshenko has essentially 100% probability. Around L/h = 19, probabilities are roughly 50/50."

## slide 17 — evidence strength summary

Image: `outputs/figures/detailed/evidence_strength_bars.png`

Speaker notes: "Five cases show definitive Timoshenko preference. The remaining six are inconclusive — EB is acceptable there."

## slide 18 — transition analysis

Image: `outputs/figures/detailed/transition_analysis.png`

- interpolated transition: L/h ~ 19.3
- L/h = 15: log BF = -2.45 (last case with clear Timoshenko preference)
- L/h = 20: log BF = +0.39 (first case with EB preference)
- transition width: approximately L/h in [15, 20]

Speaker notes: "The transition is sharp, not a gradual zone. The decision boundary is well-defined."

## slide 19 — practical guidelines

Decision rules:

```
L/h < 15      -> Timoshenko (shear detectable and significant)
15 <= L/h < 20 -> transition zone (run model selection or default to Timoshenko)
L/h >= 20      -> Euler-Bernoulli (simpler, equally accurate)
```

During operation: re-evaluate if geometry changes (e.g., corrosion reducing cross-section height).

Speaker notes: "Below 15, always use Timoshenko. Above 20, EB is sufficient. In the transition zone, run model selection or default to Timoshenko."

## slide 20 — conclusions

Achieved:
- Bayesian model selection pipeline for EB vs Timoshenko
- transition point at L/h ~ 19.3, consistent with engineering rule of thumb
- all MCMC chains converged (R-hat < 1.01, ESS > 400)
- supports displacement-based and strain-based calibration
- actionable decision rules for digital twin initialization

Key contribution: replaced heuristic model choice with evidence-based selection. Occam's razor naturally penalizes Timoshenko's extra parameter when shear is negligible.

Future work:
- dynamic loading / higher vibration modes
- composite materials (larger shear effects)
- real sensor data from monitored structures
- online model switching during digital twin operation

Speaker notes: "We built a framework that replaces heuristic beam theory choice with data-driven evidence. The transition at L/h ~ 19.3 confirms the engineering intuition of L/h ~ 20, now with probabilistic justification."

## figure reference

| slide | figure file | purpose |
|-------|-------------|---------|
| 4 | `outputs/figures/detailed/shear_contribution_analysis.png` | shear-to-total ratio vs L/h |
| 5 | `outputs/figures/detailed/methodology_flowchart.png` | pipeline overview |
| 7 | `outputs/figures/detailed/deflection_profiles_grid.png` | EB vs Timo vs FEM |
| 11 | `outputs/figures/detailed/convergence_diagnostics.png` | trace plots, R-hat |
| 12 | `outputs/figures/detailed/parameter_recovery_analysis.png` | posterior vs true E |
| 14 | `outputs/figures/detailed/jeffreys_scale_diagram.png` | Bayes factor scale |
| 15 | `outputs/figures/detailed/bayes_factor_scale.png` | main result |
| 16 | `outputs/figures/detailed/model_probability_analysis.png` | model probabilities |
| 17 | `outputs/figures/detailed/evidence_strength_bars.png` | evidence categories |
| 18 | `outputs/figures/detailed/transition_analysis.png` | transition region |

## timing guide

| slides | topic | time |
|--------|-------|------|
| 1-2 | title + motivation | 2 min |
| 3-4 | beam theories + shear effect | 2 min |
| 5-7 | methodology + data generation | 3 min |
| 8-10 | Bayesian framework + normalization + MCMC | 3 min |
| 11-12 | convergence + parameter recovery | 2 min |
| 13-14 | bridge sampling + Jeffreys scale | 2 min |
| 15-18 | results | 4 min |
| 19-20 | guidelines + conclusions | 2 min |
| total | | 20 min |
