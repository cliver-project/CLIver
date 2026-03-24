# Stage 1: Build the wheel
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/

# Export locked requirements and build the wheel
RUN uv export --no-dev --locked --no-hashes --no-emit-project -o requirements.txt
RUN uv build --wheel

# Stage 2: Install with pip using locked versions
FROM python:3.13-slim

COPY --from=builder /app/requirements.txt /tmp/
COPY --from=builder /app/dist/*.whl /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.txt /tmp/*.whl && \
    rm -f /tmp/requirements.txt /tmp/*.whl

# Create non-root user (UID 1001 for local dev).
# OpenShift compatibility: GID 0 (root group) + group-writable dirs,
# so arbitrary UIDs assigned by OpenShift can write to /home/cliver.
RUN useradd -u 1001 -g 0 -m -d /home/cliver -s /bin/bash cliver && \
    chmod -R g=u /home/cliver && \
    chmod g=u /etc/passwd

USER 1001
WORKDIR /home/cliver
ENV HOME=/home/cliver \
    CLIVER_CONF_DIR=/home/cliver/.cliver

ENTRYPOINT ["cliver"]
