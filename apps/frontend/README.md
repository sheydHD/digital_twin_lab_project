# Frontend — React Dashboard

| Field | Value |
|---|---|
| **Author** | Antoni Dudij, Maksim Feldmann — RWTH Aachen |
| **Status** | Review |
| **Last Updated** | 2026-03-01 |

> **TL;DR** — A React 19 + TypeScript SPA that provides interactive configuration and live-progress visualisation for the Bayesian model-selection pipeline. It follows MVVM: views own rendering, view-models own API state via React hooks, and models are plain TypeScript interfaces. In development, Vite proxies all `/api/*` traffic to the FastAPI backend; in production the compiled assets are served directly by FastAPI's `StaticFiles` mount.

## Tech Stack

| Layer | Technology |
|---|---|
| UI framework | React 19 + TypeScript 5.9 |
| Build / dev server | Vite 8 |
| Package manager | Bun |
| Architecture | MVVM (views → view-models → models) |

## Directory Layout

```
apps/frontend/src/
├── models/
│   └── types.ts                  TypeScript interfaces: SimulationConfig,
│                                 SimulationResult, SimulationProgress
├── viewmodels/
│   └── useSimulationViewModel.ts  React hook — all API calls and UI state
├── views/
│   └── Dashboard.tsx              Main dashboard (config sidebar + results panel)
├── App.tsx                       Root component
└── main.tsx                      Vite entry point
```

`vite.config.ts` proxies `/api/*` and `/static/*` to the backend at `http://localhost:8000` during development. No CORS configuration is needed for local dev.

## Quick Start

```bash
# from project root (recommended)
make frontend-dev      # → http://localhost:5173

# or manually
cd apps/frontend
bun install
bun run dev
```

## Connecting to the Backend

Start both servers in separate terminals, then open `http://localhost:5173`:

```bash
# terminal 1
make backend-dev    # FastAPI :8000

# terminal 2
make frontend-dev   # Vite :5173
```

Alternatively, `make up` starts both via Docker Compose.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `http://localhost:8000` | Backend API base URL |

## Production Build

```bash
cd apps/frontend
bun run build       # outputs to dist/
```

The Dockerfile runs a multi-stage build: `bun run build` in a `builder` stage, then copies only `dist/` into an nginx image. In production, FastAPI's `StaticFiles` mount serves `dist/` directly, so no separate nginx container is required.

## Testing and Linting

```bash
make frontend-test    # vitest unit tests
make frontend-lint    # eslint
```
