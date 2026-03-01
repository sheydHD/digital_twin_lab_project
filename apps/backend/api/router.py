"""API route handlers.

All four endpoints are gathered in a single router so they can be
included into the FastAPI application in ``app.py``.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from apps.backend.schemas import (
    HealthOut,
    ProgressOut,
    SimulationConfigIn,
    SimulationResultOut,
)
from apps.backend.services import run_simulation

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Routes ─────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthOut, include_in_schema=False)
async def health_check() -> HealthOut:
    """Liveness probe (suppressed from access log via log filter)."""
    return HealthOut()


@router.get("/progress", response_model=ProgressOut)
async def get_progress(request: Request) -> ProgressOut:
    """Return current simulation progress (poll while loading)."""
    return ProgressOut(**request.app.state.progress)


@router.get("/config", response_model=SimulationConfigIn)
async def get_config(request: Request) -> SimulationConfigIn:
    """Return the current simulation configuration."""
    return request.app.state.sim_config


@router.post("/config", response_model=SimulationConfigIn)
async def set_config(request: Request, cfg: SimulationConfigIn) -> SimulationConfigIn:
    """Update the current simulation configuration."""
    request.app.state.sim_config = cfg
    return cfg


@router.get("/results/latest", response_model=SimulationResultOut | None)
async def get_latest_result(request: Request) -> SimulationResultOut | None:
    """Return the last completed simulation result."""
    return request.app.state.last_result


@router.post("/simulate", response_model=SimulationResultOut)
async def simulate(request: Request) -> JSONResponse:
    """Run the Bayesian model-selection pipeline with the current config.

    The simulation is CPU-bound (MCMC sampling takes O(minutes)).  Running it
    directly inside an ``async def`` handler would block the event loop for
    every in-flight request.  ``asyncio.to_thread`` moves the work onto a
    thread-pool worker so the event loop stays free to serve other requests
    (health checks, config updates, …) while the simulation is in progress.

    Returns camelCase JSON so the frontend receives consistent key names.
    """
    # Reset progress state for this run; the dict is shared with run_simulation
    # via a direct reference so thread updates are immediately visible here.
    progress = {"stage": "starting", "step": 0, "total": 0, "message": "Starting simulation…", "running": True}
    request.app.state.progress = progress
    request.app.state.last_result = None  # Clear previous result on new run
    try:
        result = await asyncio.to_thread(run_simulation, request.app.state.sim_config, progress)
        request.app.state.last_result = result
        progress.update({"stage": "done", "message": "Simulation complete", "running": False})
    except Exception as exc:
        logger.exception("Simulation failed")
        progress.update({"stage": "error", "message": str(exc), "running": False})
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=result.model_dump(by_alias=True))
