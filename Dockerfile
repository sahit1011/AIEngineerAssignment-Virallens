# Multi-agent RAG over the filings in data/ (Q1).
#
#   docker build -t virallens-rag .
#   docker run --rm --env-file .env virallens-rag ask "Are pure omissions actionable under Rule 10b-5(b)?"
#   docker run --rm virallens-rag eval-retrieval        # needs no API key
#
# The index and embedding model are built into the image, so a container needs network access
# only for the LLM APIs. Secrets are passed at run time (--env-file) and never enter a layer.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first, so source edits do not invalidate this layer. The Q2 research extra
# (torch, sentence-transformers, notebooks) is not installed.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

COPY main.py ./
COPY data ./data
COPY eval ./eval
# Downloads the embedding model over plain HTTPS; the hf-xet transfer path can stall on some networks.
RUN HF_HUB_DISABLE_XET=1 /app/.venv/bin/python main.py ingest

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
