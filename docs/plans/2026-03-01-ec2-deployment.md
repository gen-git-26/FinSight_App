# EC2 Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a FastAPI layer to FinSight and deploy both services (FastAPI + Streamlit) to an AWS EC2 t2.micro instance so external users can access the app.

**Architecture:** Single Docker image running two processes via supervisord — uvicorn (FastAPI on :8000) and Streamlit on :8502. Docker Compose manages the container on EC2. FastAPI wraps the existing `run_query` graph via a thin `api.py` at the project root.

**Tech Stack:** FastAPI, uvicorn, supervisord, Docker, Docker Compose, AWS EC2 (Amazon Linux 2023, t2.micro free tier)

---

## Task 1: Write and pass tests for `api.py`

**Files:**
- Create: `api.py`
- Create: `tests/test_api.py`

### Step 1: Write the failing tests

Create `tests/test_api.py`:

```python
# tests/test_api.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


def test_health():
    from api import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_response():
    from api import app
    fake_state = {
        "response": "AAPL is trading at $190.",
        "sources": ["yfinance"],
        "is_trading_query": False,
    }
    with patch("api.get_graph") as mock_get_graph:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = fake_state
        mock_get_graph.return_value = mock_graph

        client = TestClient(app)
        response = client.post("/api/query", json={"query": "AAPL price"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "AAPL is trading at $190."
    assert body["sources"] == ["yfinance"]
    assert body["flow"] == "standard"


def test_query_trading_flow():
    from api import app
    fake_state = {
        "response": "Recommendation: BUY AAPL.",
        "sources": ["yfinance", "finnhub"],
        "is_trading_query": True,
    }
    with patch("api.get_graph") as mock_get_graph:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = fake_state
        mock_get_graph.return_value = mock_graph

        client = TestClient(app)
        response = client.post(
            "/api/query",
            json={"query": "Should I buy AAPL?", "user_id": "test-user"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["flow"] == "trading"
    assert "BUY" in body["response"]


def test_query_agent_error_returns_500():
    from api import app
    with patch("api.get_graph") as mock_get_graph:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = RuntimeError("OpenAI timeout")
        mock_get_graph.return_value = mock_graph

        client = TestClient(app)
        response = client.post("/api/query", json={"query": "AAPL price"})

    assert response.status_code == 500
    assert "OpenAI timeout" in response.json()["detail"]
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'api'` (or similar — `api.py` doesn't exist yet)

### Step 3: Create `api.py`

```python
# api.py
"""
FinSight FastAPI layer.

Endpoints:
  GET  /health       — liveness check for AWS monitoring
  POST /api/query    — run a query through the agent graph
"""
from __future__ import annotations

import dotenv
dotenv.load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import get_graph


app = FastAPI(title="FinSight API", version="1.0.0")


class QueryRequest(BaseModel):
    query: str
    user_id: str = "default"


class QueryResponse(BaseModel):
    response: str
    sources: list[str]
    flow: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    try:
        graph = get_graph()
        result = await graph.ainvoke({
            "query": body.query,
            "user_id": body.user_id,
        })
        return QueryResponse(
            response=result.get("response", "No response generated"),
            sources=result.get("sources", []),
            flow="trading" if result.get("is_trading_query") else "standard",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_api.py -v
```

Expected: All 4 tests PASS

### Step 5: Manual smoke test (optional, needs valid `.env`)

```bash
uvicorn api:app --port 8000 --reload
# In another terminal:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tesla stock price"}'
```

### Step 6: Commit

```bash
git add api.py tests/test_api.py
git commit -m "feat: add FastAPI layer with /health and /api/query endpoints"
```

---

## Task 2: Create supervisord configuration

**Files:**
- Create: `supervisord.conf`

### Step 1: Create `supervisord.conf`

```ini
; supervisord.conf
[supervisord]
nodaemon=true
logfile=/dev/null
logfile_maxbytes=0

[program:fastapi]
command=uvicorn api:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:streamlit]
command=streamlit run app.py --server.port 8502 --server.headless true --server.address 0.0.0.0
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
```

No tests for config files — verified in Task 3 (Docker build).

### Step 2: Commit

```bash
git add supervisord.conf
git commit -m "chore: add supervisord config to manage FastAPI and Streamlit processes"
```

---

## Task 3: Create Dockerfile

**Files:**
- Create: `Dockerfile`

### Step 1: Create `Dockerfile`

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install supervisor
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose both service ports
EXPOSE 8000 8502

CMD ["supervisord", "-c", "supervisord.conf"]
```

### Step 2: Verify the image builds

```bash
docker build -t finsight:local .
```

Expected: Build completes with no errors. Note: first build takes ~3-5 minutes due to pip installs.

### Step 3: Commit

```bash
git add Dockerfile
git commit -m "chore: add Dockerfile (python:3.12-slim, supervisord, ports 8000+8502)"
```

---

## Task 4: Create docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

### Step 1: Create `docker-compose.yml`

```yaml
# docker-compose.yml
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

### Step 2: Run and smoke test locally

```bash
docker compose up -d --build

# Wait ~10 seconds for both processes to start, then:
curl http://localhost:8000/health
# Expected: {"status":"ok"}

curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tesla stock price"}'
# Expected: JSON with "response" key

# Open Streamlit in browser:
# http://localhost:8502

# Stop when done:
docker compose down
```

### Step 3: Commit

```bash
git add docker-compose.yml
git commit -m "chore: add docker-compose.yml for local and EC2 deployment"
```

---

## Task 5: Provision AWS EC2 instance

This task is manual AWS console steps — no code to write.

### Step 1: Launch EC2 instance

1. Go to [AWS Console → EC2 → Launch Instance](https://console.aws.amazon.com/ec2/)
2. Settings:
   - **Name:** `finsight-demo`
   - **AMI:** Amazon Linux 2023 (64-bit x86)
   - **Instance type:** `t2.micro` (free tier eligible)
   - **Key pair:** Create new → download `.pem` file → save securely
   - **Storage:** 20 GiB gp3
3. Click **Launch instance**

### Step 2: Configure Security Group inbound rules

In the EC2 console → Security Groups → find the group attached to your instance → Edit inbound rules:

| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | My IP | SSH access |
| Custom TCP | 8000 | 0.0.0.0/0 | FastAPI |
| Custom TCP | 8502 | 0.0.0.0/0 | Streamlit |

### Step 3: Assign an Elastic IP (optional but recommended)

EC2 Console → Elastic IPs → Allocate → Associate with your instance.
This gives a stable IP that survives instance restarts.

### Step 4: Note the public IP

Copy the **Public IPv4 address** (or Elastic IP) — you'll need it for the next task.

---

## Task 6: Deploy to EC2

### Step 1: SSH into the instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ec2-user@<your-ec2-public-ip>
```

### Step 2: Install Docker and Docker Compose (run once)

```bash
sudo yum install -y docker git
sudo systemctl enable docker
sudo systemctl start docker
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo usermod -aG docker ec2-user
# Log out and back in for group change to take effect:
exit
ssh -i your-key.pem ec2-user@<your-ec2-public-ip>
```

### Step 3: Clone the repo and configure secrets

```bash
git clone <your-repo-url> FinSight_App
cd FinSight_App

# Create .env with your secrets (minimum required):
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EOF
```

### Step 4: Start the app

```bash
docker-compose up -d --build
```

Expected: Docker builds the image (~3-5 min first time), then starts the container. Both processes launch via supervisord.

### Step 5: Smoke test the live deployment

```bash
# From your local machine:
curl http://<ec2-public-ip>:8000/health
# Expected: {"status":"ok"}

curl -X POST http://<ec2-public-ip>:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tesla stock price"}'
# Expected: JSON response from the agent

# Open in browser:
# http://<ec2-public-ip>:8502   ← Streamlit UI
# http://<ec2-public-ip>:8000/docs  ← FastAPI auto-docs
```

### Step 6: Redeployment workflow (for future updates)

```bash
# On EC2, from ~/FinSight_App:
git pull origin main
docker-compose down && docker-compose up -d --build
```

---

## Checklist

- [ ] Task 1: `api.py` + `tests/test_api.py` — all 4 tests pass
- [ ] Task 2: `supervisord.conf` committed
- [ ] Task 3: `Dockerfile` builds successfully
- [ ] Task 4: `docker-compose.yml` — local smoke test passes
- [ ] Task 5: EC2 instance running, security group configured
- [ ] Task 6: App live at `http://<ec2-ip>:8502` and `http://<ec2-ip>:8000/docs`
