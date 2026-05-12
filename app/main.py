"""FastAPI application entry point"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.api.routes import router
from app.config import settings
from app.core.rag import get_rag_pipeline
from app.metrics import http_request_duration_seconds, http_requests_total

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG application...")
    rag_pipeline = get_rag_pipeline()
    rag_pipeline.initialize()
    logger.info("RAG pipeline initialized")

    yield

    logger.info("Shutting down RAG application...")
    if rag_pipeline.use_cache and rag_pipeline.query_cache:
        logger.info("Saving query-response cache...")
        rag_pipeline.query_cache.save()
    rag_pipeline.close()

    # Shut down module-level thread pool executors to release OS threads cleanly
    from app.core.rag import _rag_executor
    from app.core.search.hybrid_search import _retrieval_executor

    _rag_executor.shutdown(wait=False)
    _retrieval_executor.shutdown(wait=False)
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="RAG system with hybrid search (BM25 + Vector)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    endpoint = request.url.path
    # Skip recording metrics for the /metrics endpoint itself
    if endpoint != "/metrics":
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code),
        ).inc()
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

    return response


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)  # nosec B104
