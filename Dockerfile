# ---------- Stage 1: build dependencies ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir uv

# Copy only dependency metadata first (cache-friendly)
COPY pyproject.toml ./

# Install runtime deps into a virtual-env
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .

# Copy source code
COPY . .

# Install the project itself
RUN . /opt/venv/bin/activate && uv pip install --no-cache --no-deps .

# ---------- Stage 2: lean runtime ----------
FROM python:3.12-slim AS runtime

LABEL maintainer="sheydHD" \
      description="Bayesian beam-theory model selection for digital twins"

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Make the venv the default Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "configs/default_config.yaml"]
