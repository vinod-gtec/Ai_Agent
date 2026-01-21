from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from src.api.schemas import TaskPlan


class PlannerAgent:
    """
    Planner Agent - Decomposes complex tasks into executable subtasks.
    
    The Planner is the first agent in the workflow. It receives a user query
    and breaks it down into a sequence of subtasks that other agents can execute.
    
    Key Responsibilities:
        - Understand the user's high-level goal
        - Break it into logical, ordered subtasks
        - Identify dependencies between subtasks
        - Determine required tools/resources
        - Estimate complexity
    
    Workflow Position:
        User Query → [PLANNER] → Executor → Validator → Corrector
    
    Attributes:
        llm: The language model for generating plans.
        parser: Pydantic parser for structured output.
        prompt: The prompt template for the planner.
    
    Example:
        >>> planner = PlannerAgent(llm)
        >>> state = {"messages": [{"role": "user", "content": "Analyze sales"}]}
        >>> updated_state = planner.plan(state)
        >>> print(updated_state["current_plan"])
        ['Load sales data', 'Calculate metrics', 'Generate report']
    """
    
    def __init__(self, llm):
        """
        Initialize the Planner agent.
        
        Args:
            llm: Language model instance (ChatGroq, ChatOllama, etc.).
        
        Note:
            The LLM should support structured output or tool calling
            for best results with the PydanticOutputParser.
        """
        self.llm = llm
        
        # Parser converts LLM output to TaskPlan object
        self.parser = PydanticOutputParser(pydantic_object=TaskPlan)
        
        # Prompt template for the planner
        # Instructs the LLM on how to create good plans
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert planning agent. Your job is to break down complex tasks into clear, actionable subtasks.

GUIDELINES:
1. Create 3-7 subtasks (fewer is better if possible)
2. Make each subtask specific and actionable
3. Order subtasks logically (dependencies first)
4. Identify required tools for each subtask
5. Estimate overall complexity (low/medium/high)

GOOD PLAN EXAMPLE:
Task: "Analyze Q4 sales and forecast Q1"
Subtasks:
1. Load Q4 sales data from database
2. Calculate key metrics (revenue, growth, trends)
3. Identify patterns and seasonality
4. Generate Q1 forecast based on trends
5. Create summary report with recommendations

BAD PLAN EXAMPLE:
1. Do analysis (too vague)
2. Make report (missing steps)

{format_instructions}

Remember: Be specific, logical, and complete."""),
            ("user", "Task: {task}\n\nCreate a detailed execution plan.")
        ])
    
    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an execution plan from the user's task.
        
        This method:
        1. Extracts the user's task from state
        2. Sends it to the LLM with planning instructions
        3. Parses the LLM output into a TaskPlan object
        4. Updates the state with the plan
        5. Adds a message to the execution trace
        
        Args:
            state: Current agent state containing the user's message.
        
        Returns:
            Updated state with current_plan and required_tools populated.
        
        Side Effects:
            - Modifies state["current_plan"]
            - Modifies state["metadata"]["required_tools"]
            - Adds message to state["messages"]
        
        Example:
            >>> state = {
            ...     "messages": [{"role": "user", "content": "Analyze trends"}],
            ...     "current_plan": None,
            ...     "metadata": {}
            ... }
            >>> new_state = planner.plan(state)
            >>> print(new_state["current_plan"])
            ['Load data', 'Analyze patterns', 'Generate insights']
        """
        # Extract the user's task from the last message
        task = state["messages"][-1]["content"]
        
        # Create the chain: prompt → LLM → parser
        chain = self.prompt | self.llm | self.parser
        
        # Invoke the chain to generate the plan
        plan = chain.invoke({
            "task": task,
            "format_instructions": self.parser.get_format_instructions()
        })
        
        # Update state with the generated plan
        state["current_plan"] = plan.subtasks
        state["metadata"]["required_tools"] = plan.required_tools
        state["metadata"]["estimated_complexity"] = plan.estimated_complexity
        
        # Add trace message for debugging/logging
        state["messages"].append({
            "role": "planner",
            "content": (
                f"Created execution plan with {len(plan.subtasks)} subtasks. "
                f"Complexity: {plan.estimated_complexity}"
            )
        })
        
        return state