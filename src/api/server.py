from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Create FastAPI application instance
app = FastAPI(
    title="Multi-Agent AI System",
    description="""
    Production-ready multi-agent orchestration system.
    
    Features:
    - 4 specialized AI agents (Planner, Executor, Validator, Corrector)
    - LangGraph state machine orchestration
    - FREE LLM support (Groq/Ollama)
    - Short-term + long-term memory
    - Extensible tool registry
    - RESTful API
    
    Workflow:
    1. Planner: Breaks down complex tasks
    2. Executor: Runs tasks with tools
    3. Validator: Checks output quality
    4. Corrector: Fixes issues if needed
    """,
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc UI
)

# Add CORS middleware for cross-origin requests
# Allows frontend applications to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    
    Returns:
        Basic information about the API.
    
    Example:
        >>> curl http://localhost:8000/
    """
    return {
        "message": "Multi-Agent AI System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "memory": "GET /memory/recent"
        }
    }
