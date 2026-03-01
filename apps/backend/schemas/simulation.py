"""Pydantic schemas for API request / response validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ── Request models ──────────────────────────────────────────────────────────


class BeamParametersIn(BaseModel):
    length: float = 1.0
    width: float = 0.1
    aspect_ratios: list[float] = Field(default=[5, 8, 10, 12, 15, 19, 20, 30, 50, 60])


class MaterialIn(BaseModel):
    elastic_modulus: float = 210.0e9
    poisson_ratio: float = 0.3


class BayesianIn(BaseModel):
    n_samples: int = 1500
    n_tune: int = 800
    n_chains: int = 4


class DataIn(BaseModel):
    noise_fraction: float = 0.0005


class SimulationConfigIn(BaseModel):
    beam_parameters: BeamParametersIn = BeamParametersIn()
    material: MaterialIn = MaterialIn()
    bayesian: BayesianIn = BayesianIn()
    data: DataIn = DataIn()


# ── Response models ─────────────────────────────────────────────────────────


class HealthOut(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ProgressOut(BaseModel):
    """Real-time progress of a running simulation."""

    stage: str = "idle"
    step: int = 0
    total: int = 0
    message: str = ""
    running: bool = False


class SimulationResultOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str
    status: str
    log_bayes_factors: dict[str, float] = Field(serialization_alias="logBayesFactors")
    recommended_model: str = Field(serialization_alias="recommendedModel")
    transition_point: float | None = Field(default=None, serialization_alias="transitionPoint")
    plots: list[str] = Field(default_factory=list)
