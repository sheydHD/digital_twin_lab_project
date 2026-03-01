"""Services package – re-exports all public service functions."""

from __future__ import annotations

from apps.backend.services.simulation import run_simulation

__all__ = ["run_simulation"]
