FROM python:3.12-slim@sha256:3d5ed973e45820f5ba5e46bd065bd88b3a504ff0724d85980dcd05eab361fcf4
COPY --from=ghcr.io/astral-sh/uv:0.11.21@sha256:ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946 /uv /uvx /bin/

ENV PYTENSOR_FLAGS="cxx=,mode=FAST_COMPILE" \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /requirements.lock
RUN uv pip install --system --no-cache --no-deps --require-hashes -r /requirements.lock

RUN python -c "import pymc, pytensor, numpy, arviz, pytest"

RUN mkdir -p /workspace /tests /solution /app /logs/verifier /logs/agent /logs/artifacts

WORKDIR /workspace
