from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor


class ExecutorAgent:
    """
    Executor Agent - Executes subtasks using available tools.
    
    The Executor is the second agent in the workflow. It takes the plan
    created by the Planner and executes each subtask using the appropriate tools.
    
    Key Responsibilities:
        - Execute each subtask in the plan
        - Choose and invoke the right tools
        - Handle tool errors and retries
        - Collect and organize results
    
    Workflow Position:
        User Query → Planner → [EXECUTOR] → Validator → Corrector
    
    Attributes:
        llm: Language model for reasoning and tool selection.
        tools: List of available tools from ToolRegistry.
        agent: LangChain agent configured for tool calling.
        executor: AgentExecutor that runs the agent.
    
    Example:
        >>> executor = ExecutorAgent(llm, tool_registry)
        >>> state = {"current_plan": ["Analyze data", "Generate report"]}
        >>> updated_state = executor.execute(state)
        >>> print(updated_state["execution_results"])
    """
    
    def __init__(self, llm, tool_registry):
        """
        Initialize the Executor agent with tools.
        
        Args:
            llm: Language model instance.
            tool_registry: ToolRegistry instance providing available tools.
        
        Note:
            The LLM must support tool/function calling for this agent
            to work properly. Most modern LLMs (GPT-4, Claude, Llama 3+) do.
        """
        self.llm = llm
        self.tools = tool_registry.get_all_tools()
        
        # Prompt template for the executor
        # Instructs the agent on how to use tools effectively
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an execution agent with access to various tools.

YOUR JOB:
- Execute each subtask using the appropriate tools
- Be thorough and accurate
- If a tool fails, try an alternative approach
- Return clear, complete results

AVAILABLE TOOLS:
{tool_names}

EXECUTION TIPS:
1. Read the subtask carefully
2. Choose the most appropriate tool
3. Provide all required parameters
4. Verify the tool output makes sense
5. If uncertain, use multiple tools to cross-check

Always aim for accurate, complete execution."""),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create the tool-calling agent
        # This agent can reason about which tools to use and when
        self.agent = create_tool_calling_agent(
            self.llm,
            self.tools,
            self.prompt
        )
        
        # Wrap in AgentExecutor for execution
        # Handles the agent loop: think → act → observe → repeat
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,  # Set to False in production
            max_iterations=10,  # Prevent infinite loops
            handle_parsing_errors=True  # Gracefully handle errors
        )
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all subtasks in the plan.
        
        This method:
        1. Retrieves the plan from state
        2. Executes each subtask sequentially
        3. Collects results from each execution
        4. Updates state with all results
        5. Adds execution trace message
        
        Args:
            state: Current agent state with current_plan populated.
        
        Returns:
            Updated state with execution_results populated.
        
        Side Effects:
            - Modifies state["execution_results"]
            - Adds message to state["messages"]
            - May call external tools/APIs
        
        Example:
            >>> state = {"current_plan": ["Load data", "Analyze"]}
            >>> new_state = executor.execute(state)
            >>> print(new_state["execution_results"]["subtask_0"])
        
        Note:
            If a subtask fails, it will be recorded in the results
            but execution continues. The Validator will catch failures.
        """
        plan = state.get("current_plan", [])
        results = {}
        
        # Execute each subtask in order
        for i, subtask in enumerate(plan):
            try:
                # Invoke the agent executor for this subtask
                # The agent will:
                # 1. Reason about what tools to use
                # 2. Call the tools with appropriate parameters
                # 3. Process the tool outputs
                # 4. Return a final result
                result = self.executor.invoke({
                    "input": subtask,
                    "tool_names": ", ".join([t.name for t in self.tools])
                })
                
                # Store the result
                results[f"subtask_{i}"] = result["output"]
                
            except Exception as e:
                # If execution fails, record the error
                # Don't crash - let validator handle it
                results[f"subtask_{i}"] = f"ERROR: {str(e)}"
        
        # Update state with all results
        state["execution_results"] = results
        
        # Add trace message
        successful = sum(1 for v in results.values() if not v.startswith("ERROR"))
        state["messages"].append({
            "role": "executor",
            "content": (
                f"Completed {len(results)} subtasks "
                f"({successful} successful, {len(results)-successful} errors)"
            )
        })
        
        return state