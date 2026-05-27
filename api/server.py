import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routes.chat_routes import ChatRequest, chat as chat_handler, router as chat_router
from api.routes.recruitment_routes import router as recruitment_router
from api.routes.onboarding_routes import router as onboarding_router
from api.routes.policy_routes import router as policy_router
from core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app startup and shutdown.
    
    STARTUP:
    - Initialize embedding models
    - Initialize vector store
    - Initialize LLM provider
    - Warm up ranker model
    
    SHUTDOWN:
    - Cleanup resources
    """
    # Startup
    print("\n" + "="*70)
    print("🚀 HRBOT BACKEND STARTUP - Initializing AI Models".center(70))
    print("="*70 + "\n")
    
    try:
        logger.info("SERVER: Initializing embeddings provider...")
        print("[INIT] 1/4 Loading embedding model...")
        from core.embedding_provider import get_embedding_function
        get_embedding_function()
        print("[INIT] ✓ Embeddings ready")
        
        logger.info("SERVER: Initializing vector store...")
        print("[INIT] 2/4 Loading vector store...")
        from core.vector_store import get_vector_store
        get_vector_store(collection_name="hr_knowledge_base")
        print("[INIT] ✓ Vector store ready")
        
        logger.info("SERVER: Verifying LLM provider...")
        print("[INIT] 3/4 Verifying LLM provider...")
        from core.llm_provider import llm
        if llm is None:
            print("[INIT] ⚠ LLM not configured - check GROQ_API_KEY")
            logger.warning("SERVER: LLM not initialized")
        else:
            print("[INIT] ✓ LLM provider ready")
        
        logger.info("SERVER: Pre-warming ranker model...")
        print("[INIT] 4/4 Pre-warming ranker model...")
        from modules.rag_module import get_ranker
        get_ranker()
        print("[INIT] ✓ Ranker ready")
        
        print("\n" + "="*70)
        print("✅ HRBOT BACKEND INITIALIZED SUCCESSFULLY".center(70))
        print("="*70 + "\n")
        logger.info("SERVER: Startup complete")
        
    except Exception as e:
        logger.error(f"SERVER: Startup error: {str(e)[:200]}")
        print(f"\n[ERROR] Startup failed: {e}")
        print("="*70 + "\n")
        raise
    
    yield
    
    # Shutdown
    print("\n" + "="*70)
    print("🛑 HRBOT BACKEND SHUTDOWN".center(70))
    print("="*70 + "\n")
    logger.info("SERVER: Shutting down")


app = FastAPI(
    title="SVYIA HR AI Backend API",
    description="Scalable API endpoints for SVYIA HR AI Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend/backend integration
_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:8000,http://127.0.0.1:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in _cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Include sub-routers under /api
app.include_router(chat_router, prefix="/api")
app.include_router(recruitment_router, prefix="/api")
app.include_router(onboarding_router, prefix="/api")
app.include_router(policy_router, prefix="/api")


@app.get("/health")
async def health_check():
    """
    Simple health check route.
    """
    return {"status": "healthy", "service": "SVYIA HR AI Backend"}


@app.get("/")
async def serve_chat_ui():
    """Serve built-in chat UI for local testing."""
    index_path = _static_dir / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return {"status": "healthy", "service": "SVYIA HR AI Backend", "chat_ui": False}


@app.post("/chat")
async def chat_compat(request: ChatRequest):
    return await chat_handler(request)


if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
