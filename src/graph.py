from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from src.agents.planner import PlannerAgent
from src.agents.executor import ExecutorAgent
from src.agents.validator import ValidatorAgent
from src.agents.corrector import CorrectorAgent
from src.memory.manager import MemoryManager
from src.tools.registry import ToolRegistry
from src.api.schemas import AgentState
from src.utils.llm_factory import FreeLLMFactory


class MultiAgentSystem:
    """
    Multi-agent orchestration system using LangGraph.
    
    This is the main system that coordinates all agents using a state machine.
    The workflow is: Planner → Executor → Validator → Corrector (if needed)
    
    Key Features:
        - State-based workflow (LangGraph)
        - Automatic routing between agents
        - Conditional execution (corrector only if needed)
        - Memory management (short + long term)
        - Tool registry for extensibility
    
    Architecture:
        ┌─────────┐
        │  User   │
        │  Query  │
        └────┬────┘
             ▼
        ┌─────────┐
        │ PLANNER │ ← Creates execution plan
        └────┬────┘
             ▼
        ┌─────────┐
        │EXECUTOR │ ← Runs plan with tools
        └────┬────┘
             ▼
        ┌──────────┐
        │VALIDATOR │ ← Checks quality
        └────┬─────┘
             ▼
        ┌──────────┐
        │CORRECTOR?│ ← Fixes if needed
        └────┬─────┘
             ▼
        ┌──────────┐
        │  Output  │
        └──────────┘
    
    Attributes:
        llm: The language model used by all agents.
        memory: MemoryManager for conversation and semantic memory.
        tool_registry: ToolRegistry with available tools.
        planner: PlannerAgent instance.
        executor: ExecutorAgent instance.
        validator: ValidatorAgent instance.
        corrector: CorrectorAgent instance.
        graph: Compiled LangGraph state machine.
    
    Usage:
        >>> system = MultiAgentSystem()
        >>> result = system.run("Analyze sales data and forecast Q1")
        >>> print(result["output"])
    """
    
    def __init__(self):
        """
        Initialize the multi-agent system with all components.
        
        This sets up:
        1. FREE LLM (Groq/Ollama based on config)
        2. Memory systems (short + long term)
        3. Tool registry with available tools
        4. All four agents
        5. LangGraph state machine
        
        Note:
            First run may take a few seconds to download
            the embedding model for long-term memory.
        """
        # Create FREE LLM based on environment configuration
        self.llm = FreeLLMFactory.create_llm()
        
        # Initialize memory and tool systems
        self.memory = MemoryManager()
        self.tool_registry = ToolRegistry()
        
        # Initialize all agents with the same LLM
        # This ensures consistent behavior across agents
        self.planner = PlannerAgent(self.llm)
        self.executor = ExecutorAgent(self.llm, self.tool_registry)
        self.validator = ValidatorAgent(self.llm)
        self.corrector = CorrectorAgent(self.llm)
        
        # Build and compile the LangGraph workflow
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        
        Creates a directed graph that routes between agents based on state.
        The graph handles:
        - Linear flow: Planner → Executor → Validator
        - Conditional flow: Validator → Corrector (if validation fails)
        - State management: Passes AgentState between nodes
        
        Returns:
            Compiled StateGraph ready for execution.
        
        Graph Structure:
            START
              ↓
            planner (always)
              ↓
            executor (always)
              ↓
            validator (always)
              ↓
            corrector? (only if validation failed)
              ↓
            END
        
        Note:
            The graph is compiled, so changes require rebuilding.
        """
        # Create graph with AgentState type
        workflow = StateGraph(AgentState)
        
        # Add all agent nodes
        # Each node is a function that receives and returns AgentState
        workflow.add_node("planner", self.planner.plan)
        workflow.add_node("executor", self.executor.execute)
        workflow.add_node("validator", self.validator.validate)
        workflow.add_node("corrector", self.corrector.correct)
        
        def should_correct(state: AgentState) -> Literal["corrector", "end"]:
            """
            Routing function: decides if correction is needed.
            
            Args:
                state: Current agent state after validation.
            
            Returns:
                "corrector" if validation failed, "end" if passed.
            
            This is the key conditional in the workflow.
            Only runs corrector if there are issues to fix.
            """
            if state.get("validation_status") == "failed":
                return "corrector"
            return "end"
        
        # Define the workflow edges
        # SET ENTRY POINT: Where execution starts
        workflow.set_entry_point("planner")
        
        # LINEAR EDGES: Always follow these paths
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "validator")
        
        # CONDITIONAL EDGE: Route based on validation result
        workflow.add_conditional_edges(
            "validator",  # Source node
            should_correct,  # Routing function
            {
                "corrector": "corrector",  # If failed → corrector
                "end": END  # If passed → end
            }
        )
        
        # FINAL EDGE: Corrector always goes to end
        workflow.add_edge("corrector", END)
        
        # Compile the graph
        # This creates an executable workflow
        return workflow.compile()
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Run the complete multi-agent workflow.
        
        This is the main entry point for processing user queries.
        It orchestrates all agents through the state machine.
        
        Workflow:
        1. Create initial state with user query
        2. Execute graph (agents run automatically)
        3. Extract final output
        4. Store in memory
        5. Return results
        
        Args:
            user_input: The user's question or task.
        
        Returns:
            Dictionary containing:
            - output: Final processed result
            - plan: Execution plan that was followed
            - messages: Trace of agent actions
            - metadata: Additional info (tools used, etc.)
        
        Example:
            >>> system = MultiAgentSystem()
            >>> result = system.run("Analyze Q4 sales and forecast Q1")
            >>> 
            >>> print("Plan:", result["plan"])
            >>> print("Output:", result["output"])
            >>> print("Agents:", [m["role"] for m in result["messages"]])
        
        Side Effects:
            - Adds to short-term memory (conversation history)
            - May add to long-term memory (semantic search)
            - May call external tools/APIs
        """
        # Create initial state for the workflow
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": user_input}],
            "current_plan": None,
            "execution_results": {},
            "validation_status": None,
            "corrections_needed": [],
            "final_output": None,
            "metadata": {}
        }
        
        # Execute the graph
        # This runs all agents in sequence following the workflow
        final_state = self.graph.invoke(initial_state)
        
        # Store in memory for context in future queries
        self.memory.add_to_short_term("user", user_input)
        self.memory.add_to_short_term("assistant", final_state["final_output"])
        
        # Optionally store in long-term memory for semantic retrieval
        # Uncomment if you want to remember all interactions
        # self.memory.add_to_long_term(
        #     key=f"interaction_{len(final_state['messages'])}",
        #     value=final_state["final_output"],
        #     metadata={"query": user_input}
        # )
        
        # Return structured results
        return {
            "output": final_state["final_output"],
            "plan": final_state["current_plan"],
            "messages": final_state["messages"],
            "metadata": final_state["metadata"]
        }
