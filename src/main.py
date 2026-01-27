import os
import logging
from dotenv import load_dotenv
from src.graph import MultiAgentSystem
from src.api.server import app
from src.api import routes

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Validate environment configuration
def validate_environment():
    """
    Validate that required environment variables are set.
    
    Raises:
        ValueError: If required configuration is missing.
    """
    provider = os.getenv("LLM_PROVIDER", "groq")
    
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY not set. Get free key from: https://console.groq.com"
        )
    elif provider == "together" and not os.getenv("TOGETHER_API_KEY"):
        raise ValueError(
            "TOGETHER_API_KEY not set. Get from: https://api.together.xyz"
        )
    elif provider == "huggingface" and not os.getenv("HUGGINGFACE_API_KEY"):
        raise ValueError(
            "HUGGINGFACE_API_KEY not set. Get from: https://huggingface.co"
        )
    
    logger.info(f"Using LLM provider: {provider}")

# Validate configuration
validate_environment()

# Initialize the multi-agent system
logger.info("Initializing multi-agent system...")
try:
    system = MultiAgentSystem()
    logger.info("✓ Multi-agent system initialized successfully")
except Exception as e:
    logger.error(f"✗ Failed to initialize system: {e}")
    raise

# Link system to routes so API can access it
routes.system = system

# Include router in the app
app.include_router(routes.router)

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Run on application startup.
    
    Logs system information and configuration.
    """
    logger.info("="*80)
    logger.info("Multi-Agent AI System Starting...")
    logger.info("="*80)
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'groq')}")
    logger.info(f"LLM Model: {os.getenv('LLM_MODEL', 'default')}")
    logger.info(f"API Docs: http://localhost:8000/docs")
    logger.info("="*80)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Run on application shutdown.
    
    Cleanup resources if needed.
    """
    logger.info("Multi-Agent AI System shutting down...")

# Main entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    # Run the application
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level=os.getenv("LOG_LEVEL", "info").lower()