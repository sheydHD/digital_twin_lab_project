# ---------- Stage 1: build deps ----------
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .

COPY . .
RUN . /opt/venv/bin/activate && uv pip install --no-cache --no-deps .

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim

# PyTensor needs g++ at runtime to compile C-optimized MCMC kernels.
# Without it, sampling falls back to pure Python (~17x slower).
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "apps.backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
