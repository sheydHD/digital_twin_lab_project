"""FastAPI application factory.

Instantiate with:
    uvicorn apps.backend.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from apps.backend.api.router import router
from apps.backend.schemas import SimulationConfigIn

logger = logging.getLogger(__name__)

# Configurable via environment — safe default allows the Vite dev-server only.
# Override in production:  CORS_ORIGINS="https://your-domain.com"
_CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise per-process state and ensure output directories exist."""
    app.state.sim_config = SimulationConfigIn()  # default config, mutable via /api/config
    app.state.progress = {"stage": "idle", "step": 0, "total": 0, "message": "", "running": False}
    app.state.last_result = None
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    logger.info("Backend ready – serving on port 8000")
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Digital Twin Lab API",
        version="0.1.0",
        description="Bayesian beam-theory model selection backend",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Serve generated figures as static files ────────────────────────────
    _figures_dir = Path("outputs/figures")
    _figures_dir.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/static/figures",
        StaticFiles(directory=str(_figures_dir)),
        name="figures",
    )

    # ── Register all API routes under /api ─────────────────────────────────
    application.include_router(router, prefix="/api")

    # ── Suppress noisy access-log lines for health & progress polls ────────
    _install_access_log_filter()

    return application


class _QuietPollFilter(logging.Filter):
    """Drop uvicorn access-log records for /api/health and /api/progress."""

    _NOISY = ("/api/health", "/api/progress")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._NOISY)


def _install_access_log_filter() -> None:
    """Attach the filter to uvicorn's access logger (if it exists)."""
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.addFilter(_QuietPollFilter())


# Module-level instance used by uvicorn and tests.
app = create_app()
