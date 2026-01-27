from fastapi import APIRouter, HTTPException, status
from src.api.schemas import QueryRequest, QueryResponse
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Create router for all agent-related endpoints
router = APIRouter(
    prefix="",
    tags=["Multi-Agent System"]
)

# This will be set in main.py
# Allows routes to access the initialized system
system = None


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process a query through the multi-agent system",
    description="""
    Submit a query to be processed by the multi-agent system.
    
    The query will be:
    1. Analyzed and broken down (Planner)
    2. Executed with available tools (Executor)
    3. Validated for quality (Validator)
    4. Corrected if needed (Corrector)
    
    Returns the final output along with execution details.
    """,
    responses={
        200: {
            "description": "Query processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "output": "Analysis complete: Sales increased by 15%...",
                        "plan": [
                            "Load Q4 sales data",
                            "Analyze trends",
                            "Generate forecast"
                        ],
                        "execution_trace": [
                            {"role": "planner", "content": "Created plan"},
                            {"role": "executor", "content": "Completed 3 subtasks"}
                        ],
                        "metadata": {
                            "required_tools": ["data_analyzer"],
                            "estimated_complexity": "medium"
                        }
                    }
                }
            }
        },
        500: {"description": "Internal server error"}
    }
)
async def process_query(request: QueryRequest):
    """
    Process a user query through the multi-agent system.
    
    Args:
        request: QueryRequest containing the query and optional context.
    
    Returns:
        QueryResponse with output, plan, trace, and metadata.
    
    Raises:
        HTTPException: If system is not initialized or processing fails.
    
    Example:
        ```python
        import requests
        
        response = requests.post(
            "http://localhost:8000/query",
            json={
                "query": "Analyze Q4 sales data",
                "context": {"year": 2024}
            }
        )
        print(response.json()["output"])
        ```
    """
    # Check if system is initialized
    if system is None:
        logger.error("System not initialized")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multi-agent system not initialized"
        )
    
    try:
        # Log incoming request
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Run the multi-agent system
        result = system.run(request.query)
        
        # Build response
        response = QueryResponse(
            output=result["output"],
            plan=result["plan"],
            execution_trace=result["messages"],
            metadata=result["metadata"]
        )
        
        # Log success
        logger.info(f"Query processed successfully. Plan had {len(result['plan'])} steps.")
        
        return response
        
    except Exception as e:
        # Log error
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        
        # Return error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check endpoint",
    description="Check if the system is running and all agents are initialized.",
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "agents": ["planner", "executor", "validator", "corrector"],
                        "llm_provider": "groq",
                        "tools_available": 4
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Check system health and status.
    
    Returns:
        Status information about the system and its components.
    
    Example:
        >>> curl http://localhost:8000/health
    """
    if system is None:
        return {
            "status": "unhealthy",
            "reason": "System not initialized"
        }
    
    return {
        "status": "healthy",
        "agents": ["planner", "executor", "validator", "corrector"],
        "memory": {
            "short_term": len(system.memory.get_short_term_history()),
            "long_term": "enabled"
        },
        "tools_available": len(system.tool_registry.get_all_tools())
    }


@router.get(
    "/memory/recent",
    summary="Get recent conversation history",
    description="Retrieve recent messages from short-term memory.",
    responses={
        200: {
            "description": "Recent conversation history",
            "content": {
                "application/json": {
                    "example": {
                        "history": [
                            {"role": "user", "content": "Analyze sales"},
                            {"role": "assistant", "content": "Analysis complete"}
                        ],
                        "count": 2
                    }
                }
            }
        }
    }
)
async def get_recent_memory():
    """
    Retrieve recent conversation history.
    
    Returns:
        Recent messages from short-term memory.
    
    Example:
        >>> curl http://localhost:8000/memory/recent
    """
    if system is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System not initialized"
        )
    
    history = system.memory.get_short_term_history()
    
    return {
        "history": history,
        "count": len(history)
    }


@router.delete(
    "/memory/clear",
    summary="Clear short-term memory",
    description="Clear all messages from conversation history.",
    responses={
        200: {"description": "Memory cleared successfully"}
    }
)
async def clear_memory():
    """
    Clear short-term conversation memory.
    
    Returns:
        Confirmation message.
    
    Example:
        >>> curl -X DELETE http://localhost:8000/memory/clear
    """
    if system is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System not initialized"
        )
    
    system.memory.short_term.clear()
    
    return {
        "status": "success",
        "message": "Short-term memory cleared"
    }