# P2-2: Add Dockerfile + docker-compose for Both Projects

## Context
Neither project has a Dockerfile. Adding containerization shows production mindset and makes it trivial for interviewers to run the projects.

## What to do

### Part A: content-intelligence-engine

Read `pyproject.toml` (or `requirements.txt`) and `src/api/main.py` to understand the app entry point.

Create `Dockerfile`:
```dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml ./
# If using pip install -e ., copy setup files too
COPY setup.py setup.cfg README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Create data directory for SQLite/JSONL
RUN mkdir -p data

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

IMPORTANT: Read the actual project files to determine the correct install method (`pip install -e .` vs `poetry install` vs `pip install -r requirements.txt`). Adapt the Dockerfile accordingly.

Create `docker-compose.yml`:
```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-}
      - LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-}
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data  # Persist SQLite, checkpoints, audit logs
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 3

  # Optional: Redis for production-like caching
  # Uncomment when migrating from SQLite cache to Redis
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"
```

Create `.dockerignore`:
```
.git
.venv
venv
__pycache__
*.pyc
.pytest_cache
.mypy_cache
.ruff_cache
data/
eval_results/
.env
```

### Part B: agentic-rag

Same pattern but at `/Users/joelle/Projects/side-projects/agentic-rag/`.

Read that project's dependency setup first (`pyproject.toml` or `requirements.txt`) and `src/api/main.py`.

Create `Dockerfile`, `docker-compose.yml`, and `.dockerignore` with the same structure, adapted for:
- Correct install method
- Correct app entry point
- Correct port (check if it uses a different port)
- Add pgvector and neo4j services to docker-compose if the project uses them (check README mentions "PostgreSQL + pgvector + Neo4j"):

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - DATABASE_URL=postgresql://rag:rag@postgres:5432/ragdb
      - NEO4J_URI=bolt://neo4j:7687
    depends_on:
      postgres:
        condition: service_healthy
      neo4j:
        condition: service_started

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: ragdb
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4jdata:/data

volumes:
  pgdata:
  neo4jdata:
```

### Step 3: Test Docker builds
```bash
# content-intelligence-engine
cd /Users/joelle/Projects/side-projects/content-intelligence-engine
docker build -t content-intel .

# agentic-rag
cd /Users/joelle/Projects/side-projects/agentic-rag
docker build -t agentic-rag .
```

If Docker is not available, at minimum ensure the Dockerfile syntax is valid and the install commands match the project's actual setup.

### IMPORTANT NOTES
- Read each project's actual dependency management (pyproject.toml vs requirements.txt vs setup.py) BEFORE writing the Dockerfile
- The Dockerfiles should build successfully with `docker build .`
- Keep it simple — no multi-stage builds unless the image is >500MB
- The docker-compose.yml should work with `docker compose up` out of the box (with optional API keys)
- Do NOT add Kubernetes manifests or Terraform — that's over-engineering for a portfolio
