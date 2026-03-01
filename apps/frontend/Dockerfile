FROM oven/bun:1 AS base
WORKDIR /app

# Install deps  (cache-friendly: copy lockfiles first)
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .

# --- dev target (default) ---
FROM base AS dev
EXPOSE 5173
CMD ["bun", "run", "dev", "--host", "0.0.0.0"]

# --- production build ---
FROM base AS build
RUN bun run build

FROM nginx:alpine AS prod
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
