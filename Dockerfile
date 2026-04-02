FROM python:3.12-slim

WORKDIR /app

# Install build backend so pip can read pyproject.toml
RUN pip install --no-cache-dir poetry-core

# Copy project files and install
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[tracing]"

# Create data directory for SQLite, checkpoints, audit logs
RUN mkdir -p data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
