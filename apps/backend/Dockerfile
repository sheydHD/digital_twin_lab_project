# ---------- Stage 1: build deps ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build-time dependencies for scientific libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .

COPY . .
RUN . /opt/venv/bin/activate && uv pip install --no-cache --no-deps .

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim

# Runtime dependencies:
# - g++ is REQUIRED at runtime for PyTensor to compile C-optimized MCMC kernels
# - libopenblas-dev & liblapack-dev provide the linear algebra speedups (BLAS/LAPACK)
# - gfortran is often needed for some LAPACK routines
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# PYTENSOR_FLAGS: Force PyTensor to use the installed OpenBLAS library.
# This eliminates the "PyTensor could not link to a BLAS installation" warning.
# base_compiledir is set to /root/.pytensor which is mounted as a named volume
# in docker-compose so the compiled C-kernels survive container restarts.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTENSOR_FLAGS="blas__ldflags=-lopenblas,base_compiledir=/root/.pytensor" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Pre-compile PyTensor C-kernels at image-build time.
# PyTensor JIT-compiles computational graphs to C on first use, which can
# take several minutes per model.  Running a minimal PyMC model here bakes the
# compiled artefacts into the image layer so the first container start is fast.
# The named volume mounted at /root/.pytensor at runtime will be initialised
# from these pre-compiled files by Docker on first use (empty-volume seeding).
RUN python - <<'EOF'
import os, warnings
warnings.filterwarnings("ignore")
# Minimal PyMC model that exercises the same Ops used by the calibrators
# (Normal likelihood, LogNormal prior, HalfNormal, pm.sample with NUTS).
import numpy as np
import pymc as pm

rng = np.random.default_rng(0)
y_obs = rng.normal(0.0, 1.0, size=10)

with pm.Model():
    mu    = pm.Normal("mu", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    _     = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
    pm.sample(
        draws=10, tune=10, chains=1,
        progressbar=False,
        random_seed=0,
        # Use only 1 chain so the subprocess overhead is minimal during build.
        compute_convergence_checks=False,
    )
print("PyTensor warmup complete – C-kernels compiled and cached.")
EOF

EXPOSE 8000
CMD ["uvicorn", "apps.backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
