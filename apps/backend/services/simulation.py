"""Simulation service – thin wrapper around the pipeline orchestrator.

Translates between the API schema (``SimulationConfigIn``) and the
:class:`~apps.backend.core.pipeline.orchestrator.PipelineOrchestrator`
that does all the heavy lifting.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from apps.backend.schemas import SimulationConfigIn, SimulationResultOut

logger = logging.getLogger(__name__)


def _build_pipeline_config(cfg: SimulationConfigIn) -> dict:
    """Convert the Pydantic model into the dict the orchestrator expects."""
    return {
        "beam_parameters": {
            "length": cfg.beam_parameters.length,
            "width": cfg.beam_parameters.width,
            "aspect_ratios": cfg.beam_parameters.aspect_ratios,
        },
        "material": {
            "elastic_modulus": cfg.material.elastic_modulus,
            "poisson_ratio": cfg.material.poisson_ratio,
        },
        "load": {
            "point_load": 1000.0,
            "distributed_load": 0.0,
        },
        "bayesian": {
            "n_samples": cfg.bayesian.n_samples,
            "n_tune": cfg.bayesian.n_tune,
            "n_chains": cfg.bayesian.n_chains,
        },
        "data_generation": {
            "noise_fraction": cfg.data.noise_fraction,
            "n_displacement_sensors": 5,
            "n_strain_gauges": 4,
        },
        "output_dir": "outputs",
    }


def _clear_figures_dir() -> None:
    """Delete all PNG files in outputs/figures before a new run.

    Prevents plots from previous runs (potentially with different aspect ratios)
    from appearing in the current run's results.
    """
    fig_dir = Path("outputs/figures")
    if fig_dir.exists():
        for png in fig_dir.glob("*.png"):
            try:
                png.unlink()
            except OSError:
                pass  # Non-fatal; worst case an old file stays around


def run_simulation(cfg: SimulationConfigIn, progress: dict | None = None) -> SimulationResultOut:
    """Execute the Bayesian model-selection pipeline and return results.

    Designed to be called from a thread-pool worker (e.g. via
    ``asyncio.to_thread``) so the FastAPI event loop is never blocked.
    Internally the pipeline already parallelises calibration steps via
    ``ProcessPoolExecutor`` and FEM data-generation via ``ThreadPoolExecutor``.

    Args:
        cfg: Simulation configuration.
        progress: Optional shared dict written at each stage so the
            ``GET /api/progress`` endpoint can report live status to the UI.
    """
    from apps.backend.core.pipeline.orchestrator import PipelineOrchestrator

    n_ratios = len(cfg.beam_parameters.aspect_ratios)
    # total steps: 1 data-gen + N calibrations + 1 analysis
    total_steps = n_ratios + 2

    def _upd(stage: str, step: int, message: str) -> None:
        if progress is not None:
            progress.update(
                stage=stage,
                step=step,
                total=total_steps,
                message=message,
                running=True,
            )
            logger.debug("Progress: [%s] %d/%d – %s", stage, step, total_steps, message)

    job_id = uuid.uuid4().hex[:12]
    pipeline_cfg = _build_pipeline_config(cfg)

    # Clear figures from any previous run so stale plots are never mixed into
    # the results of this run when the figures directory is scanned at the end.
    _clear_figures_dir()

    logger.info("Starting simulation job %s", job_id)
    orch = PipelineOrchestrator(pipeline_cfg)

    # 1. Generate data  (step 1)
    _upd("data_generation", 1, "Generating synthetic FEM datasets\u2026")
    orch.run_data_generation()

    # 2. Bayesian calibration  (steps 2 .. n_ratios+1)
    _upd("calibration", 1, f"Starting Bayesian calibration (0/{n_ratios})\u2026")

    def _calib_cb(step_i: int, total_n: int, lh: float) -> None:
        # step_i is 1-based from orchestrator; map to global step 1+step_i
        _upd(
            "calibration",
            1 + step_i,
            f"Calibrated L/h\u2009=\u2009{lh:.0f}  ({step_i}/{total_n})",
        )

    orch.run_calibration(progress_callback=_calib_cb)

    # 3. Model selection analysis  (step n_ratios+2)
    _upd("analysis", n_ratios + 2, "Running model-selection analysis\u2026")
    orch.run_analysis()

    study = orch.study_results or {}

    # Build the response payload the frontend expects
    log_bfs: dict[str, float] = {}
    recommended = "No result"
    transition: float | None = None

    if study:
        ratios = study.get("aspect_ratios", [])
        bfs = study.get("log_bayes_factors", [])
        recs = study.get("recommendations", [])
        for r, bf in zip(ratios, bfs, strict=True):
            log_bfs[str(int(r))] = round(float(bf), 4)

        transition = study.get("transition_aspect_ratio")
        if transition is not None:
            recommended = f"L/h > {transition:.1f} favors Euler-Bernoulli"
        elif recs:
            recommended = f"Majority: {max(set(recs), key=recs.count)}"

    # Collect any generated plot images as URL paths served by /static
    plots: list[str] = []
    fig_dir = Path("outputs/figures")
    if fig_dir.exists():
        for p in sorted(fig_dir.glob("*.png")):
            plots.append(f"/static/figures/{p.name}")

    if progress is not None:
        progress.update(
            stage="done",
            step=total_steps,  # == n_ratios + 2
            total=total_steps,
            message="Simulation complete",
            running=False,
        )

    return SimulationResultOut(
        job_id=job_id,
        status="completed",
        log_bayes_factors=log_bfs,
        recommended_model=recommended,
        transition_point=transition,
        plots=plots,
    )
