# Stage 1: Build admin portal
FROM node:20-slim AS admin-builder
WORKDIR /app
COPY admin/package.json admin/package-lock.json ./
RUN npm ci
COPY admin/ ./
RUN npm run build

# Stage 2: Build the Python wheel
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/

# Copy pre-built admin portal into the package
COPY --from=admin-builder /app/dist src/cliver/gateway/admin_dist

RUN uv export --no-dev --locked --no-hashes --no-emit-project -o requirements.txt
RUN uv build --wheel

# Stage 3: Runtime
FROM python:3.13-slim

COPY --from=builder /app/requirements.txt /tmp/
COPY --from=builder /app/dist/*.whl /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.txt /tmp/*.whl && \
    rm -f /tmp/requirements.txt /tmp/*.whl

RUN useradd -u 1001 -g 0 -m -d /home/cliver -s /bin/bash cliver && \
    chmod -R g=u /home/cliver && \
    chmod g=u /etc/passwd

USER 1001
WORKDIR /home/cliver
ENV HOME=/home/cliver \
    CLIVER_CONF_DIR=/home/cliver/.cliver

ENTRYPOINT ["cliver"]
