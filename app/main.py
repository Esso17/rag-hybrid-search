"""FastAPI application entry point"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.core.rag import get_rag_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    # Startup
    logger.info("Starting RAG application...")
    rag_pipeline = get_rag_pipeline()
    rag_pipeline.initialize()
    logger.info("RAG pipeline initialized")

    yield

    # Shutdown
    logger.info("Shutting down RAG application...")

    # Save cache to disk before shutdown
    if rag_pipeline.use_cache and rag_pipeline.query_cache:
        logger.info("Saving query-response cache...")
        rag_pipeline.query_cache.save()

    rag_pipeline.close()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="RAG system with hybrid search (BM25 + Vector)",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)  # nosec B104
