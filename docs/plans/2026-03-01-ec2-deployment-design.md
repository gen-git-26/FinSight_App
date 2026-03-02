# EC2 Deployment Design — FastAPI + Streamlit on AWS

**Date:** 2026-03-01
**Status:** Approved
**Scope:** Demo / portfolio showcase

---

## Goal

Deploy FinSight to AWS EC2 so external users can access the Streamlit UI and a programmatic FastAPI endpoint from anywhere, without Codespaces running.

---

## Architecture

```
Internet
    │
    ▼
EC2 t2.micro (free tier)
┌─────────────────────────────┐
│  Docker Compose             │
│   ├── FastAPI   :8000 ──────┼──► public-ip:8000/api/query
│   └── Streamlit :8502 ──────┼──► public-ip:8502
│                             │
│  Security Group:            │
│   ports 8000, 8502 open     │
└─────────────────────────────┘
```

- Single EC2 instance, single Docker image, two processes managed by supervisord
- No ALB, no ECS, no ECR required
- Secrets injected via `.env` at runtime — never baked into the image

---

## FastAPI Component

**File:** `api.py` (project root)

Thin wrapper around the existing `run_query` function. No changes to `agent/`, `datasources/`, or `app.py`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `POST` | `/api/query` | Accepts a query, returns agent response as JSON |

### Request / Response

```json
// POST /api/query — request
{
  "query": "Should I buy AAPL?",
  "user_id": "demo-user"
}

// Response
{
  "response": "Based on the analysis...",
  "sources": ["yfinance"],
  "flow": "trading"
}
```

### Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent.graph import run_query

app = FastAPI(title="FinSight API")

class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/query")
async def query(body: QueryRequest):
    try:
        result = await run_query(body.query, user_id=body.user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Docker & Process Management

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn supervisor

COPY . .

EXPOSE 8000 8502
CMD ["supervisord", "-c", "supervisord.conf"]
```

### supervisord.conf

```ini
[supervisord]
nodaemon=true

[program:fastapi]
command=uvicorn api:app --host 0.0.0.0 --port 8000
autostart=true
autorestart=true

[program:streamlit]
command=streamlit run app.py --server.port 8502 --server.headless true --server.address 0.0.0.0
autostart=true
autorestart=true
```

### docker-compose.yml

```yaml
services:
  finsight:
    build: .
    ports:
      - "8000:8000"
      - "8502:8502"
    env_file:
      - .env
    restart: unless-stopped
```

---

## AWS EC2 Setup

### Instance Configuration

| Setting | Value |
|---------|-------|
| Instance type | t2.micro (free tier) |
| AMI | Amazon Linux 2023 |
| Storage | 20 GB gp3 |
| Region | us-east-1 |

### Security Group — Inbound Rules

| Port | Source | Purpose |
|------|--------|---------|
| 22 | Your IP only | SSH access |
| 8000 | 0.0.0.0/0 | FastAPI |
| 8502 | 0.0.0.0/0 | Streamlit |

### One-Time EC2 Setup

```bash
sudo yum install -y docker git
sudo systemctl enable docker && sudo systemctl start docker
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo usermod -aG docker ec2-user
```

### Deployment Workflow

```bash
# Clone repo and add .env on first deploy
git clone <repo-url>
cd FinSight_App
cp .env.example .env  # fill in secrets

# Start
docker compose up -d --build

# Every subsequent update
git pull origin main
docker compose down && docker compose up -d --build
```

### Accessing the App

| Service | URL |
|---------|-----|
| Streamlit UI | `http://<ec2-public-ip>:8502` |
| FastAPI docs | `http://<ec2-public-ip>:8000/docs` |
| Health check | `http://<ec2-public-ip>:8000/health` |

**Recommended:** Assign an Elastic IP (free when attached to a running instance) for a stable address that survives EC2 restarts.

---

## Error Handling

- FastAPI catches all agent exceptions and returns a clean `HTTP 500` JSON response
- `restart: unless-stopped` in Docker Compose ensures both processes recover from crashes and survive EC2 reboots

---

## Testing

Smoke tests to run after deployment:

```bash
# Health check
curl http://<ec2-ip>:8000/health

# Standard query
curl -X POST http://<ec2-ip>:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tesla stock price"}'

# Trading query
curl -X POST http://<ec2-ip>:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Should I buy AAPL?"}'

# Streamlit UI
open http://<ec2-ip>:8502
```

No new automated tests required — existing `pytest tests/` covers agent logic. The FastAPI layer is thin enough that smoke tests are sufficient for demo scope.

---

## Out of Scope

- HTTPS / TLS (can add via Certbot + nginx later)
- Authentication / API keys
- CI/CD pipeline (manual `git pull` deploy is sufficient for demo)
- ALB, ECS, ECR
- Integration of disconnected memory components (separate track)
