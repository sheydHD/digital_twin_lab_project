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
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTENSOR_FLAGS='blas__ldflags=-lopenblas'

EXPOSE 8000
CMD ["uvicorn", "apps.backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
