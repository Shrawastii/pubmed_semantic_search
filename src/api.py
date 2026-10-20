import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from src.search import SemanticSearcher

state = {}


@asynccontextmanager
async def lifespan(app):
    state["searcher"] = SemanticSearcher()   # load model and index once
    yield


app = FastAPI(title="PubMed Encephalitis Search", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(state["searcher"].df)}


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50)):
    t = time.perf_counter()
    results = state["searcher"].search(q, k)
    ms = (time.perf_counter() - t) * 1000
    return {"query": q, "latency_ms": round(ms, 1), "results": results}