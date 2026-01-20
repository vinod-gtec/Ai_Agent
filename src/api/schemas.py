from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
import operator


class AgentState(TypedDict):
    """
    Shared state that gets passed between all agents in the workflow.
    
    This TypedDict serves as the central state container for the LangGraph
    state machine. Each agent reads from and writes to this state.
    
    Attributes:
        messages: List of conversation messages. Uses Annotated with operator.add
                 to automatically concatenate new messages instead of replacing.
        current_plan: The execution plan created by the Planner agent.
                     Contains ordered list of subtasks.
        execution_results: Dictionary mapping subtask IDs to their results.
                          Populated by the Executor agent.
        validation_status: Current validation state ("passed" or "failed").
                          Set by the Validator agent.
        corrections_needed: List of issues found during validation.
                           Used by Corrector to know what to fix.
        final_output: The final processed output after all agents have run.
        metadata: Additional information like required tools, complexity, etc.
    
    Example:
        >>> state: AgentState = {
        ...     "messages": [{"role": "user", "content": "Analyze sales"}],
        ...     "current_plan": None,
        ...     "execution_results": {},
        ...     "validation_status": None,
        ...     "corrections_needed": [],
        ...     "final_output": None,
        ...     "metadata": {}
        ... }
    """
    messages: Annotated[List[Dict[str, Any]], operator.add]
    current_plan: Optional[List[str]]
    execution_results: Dict[str, Any]
    validation_status: Optional[str]
    corrections_needed: List[str]
    final_output: Optional[str]
    metadata: Dict[str, Any]


class TaskPlan(BaseModel):
    """
    Structured output from the Planner agent.
    
    Represents a decomposed task with subtasks, dependencies, and metadata.
    Used for parsing LLM output into a structured format.
    
    Attributes:
        subtasks: Ordered list of subtasks to execute.
        dependencies: Map of subtask to its dependencies (for parallel execution).
        estimated_complexity: Rough estimate of task difficulty.
        required_tools: List of tools needed to complete the task.
    
    Example:
        >>> plan = TaskPlan(
        ...     subtasks=["Load data", "Analyze trends", "Generate report"],
        ...     dependencies={"Generate report": ["Load data", "Analyze trends"]},
        ...     estimated_complexity="medium",
        ...     required_tools=["data_analyzer", "report_generator"]
        ... )
    """
    subtasks: List[str] = Field(
        description="Ordered list of subtasks to complete the main task"
    )
    dependencies: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of subtasks to their prerequisites"
    )
    estimated_complexity: str = Field(
        default="medium",
        description="Complexity estimate: low, medium, or high"
    )
    required_tools: List[str] = Field(
        default_factory=list,
        description="List of tool names required for execution"
    )


class ValidationResult(BaseModel):
    """
    Structured output from the Validator agent.
    
    Contains validation status and detailed feedback about output quality.
    
    Attributes:
        is_valid: Whether the output passes validation criteria.
        confidence: Confidence score of the validation (0.0 to 1.0).
        issues: List of problems found in the output.
        suggestions: List of recommendations for improvement.
    
    Example:
        >>> validation = ValidationResult(
        ...     is_valid=False,
        ...     confidence=0.7,
        ...     issues=["Missing data source", "Incomplete analysis"],
        ...     suggestions=["Add data sources", "Include statistical tests"]
        ... )
    """
    is_valid: bool = Field(
        description="True if output meets quality standards"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level in the validation (0-1)"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of problems found in the output"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving the output"
    )


class QueryRequest(BaseModel):
    """
    API request schema for user queries.
    
    Attributes:
        query: The user's question or task.
        context: Optional additional context or parameters.
    
    Example:
        >>> request = QueryRequest(
        ...     query="Analyze Q4 sales data",
        ...     context={"year": 2024, "region": "US"}
        ... )
    """
    query: str = Field(
        description="User's query or task to process"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for the query"
    )


class QueryResponse(BaseModel):
    """
    API response schema for query results.
    
    Attributes:
        output: Final processed output.
        plan: The execution plan that was followed.
        execution_trace: Log of agent actions.
        metadata: Additional information about the execution.
    
    Example:
        >>> response = QueryResponse(
        ...     output="Sales increased by 15%...",
        ...     plan=["Load data", "Analyze", "Report"],
        ...     execution_trace=[{"role": "planner", "content": "Created plan"}],
        ...     metadata={"tools_used": ["analyzer"]}
        ... )
    """
    output: str = Field(
        description="Final output from the multi-agent system"
    )
    plan: List[str] = Field(
        description="The execution plan that was followed"
    )
    execution_trace: List[Dict[str, str]] = Field(
        description="Chronological log of agent actions"
    )
    metadata: Dict[str, Any] = Field(
        description="Additional execution metadata"
    )