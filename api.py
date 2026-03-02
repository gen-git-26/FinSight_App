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

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import get_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from infrastructure.postgres_summaries import get_summaries
        get_summaries()
        print("[Startup] Postgres initialized")
    except Exception as e:
        print(f"[Startup] Postgres not available: {e}")
    yield


app = FastAPI(title="FinSight API", version="1.0.0", lifespan=lifespan)


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
