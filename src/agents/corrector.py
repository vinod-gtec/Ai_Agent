from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate


class CorrectorAgent:
    """
    Corrector Agent - Fixes validation issues and refines output.
    
    The Corrector is the optional fourth agent, activated only if validation
    fails. It takes the validator's feedback and produces an improved version.
    
    Key Responsibilities:
        - Address all validation issues
        - Refine and improve output quality
        - Fill in missing information
        - Reorganize for better clarity
        - Ensure final output meets standards
    
    Workflow Position:
        User Query → Planner → Executor → Validator → [CORRECTOR] (if needed)
    
    Correction Strategy:
        1. Review validation feedback
        2. Identify root causes of issues
        3. Apply targeted fixes
        4. Enhance overall quality
        5. Produce final polished output
    
    Attributes:
        llm: Language model for correction and refinement.
        prompt: Prompt template for corrections.
    
    Example:
        >>> corrector = CorrectorAgent(llm)
        >>> state = {
        ...     "corrections_needed": ["Missing data source", "Incomplete"],
        ...     "execution_results": {"subtask_0": "Partial result"}
        ... }
        >>> updated_state = corrector.correct(state)
        >>> print(updated_state["final_output"])
    """
    
    def __init__(self, llm):
        """
        Initialize the Corrector agent.
        
        Args:
            llm: Language model instance for correction reasoning.
        """
        self.llm = llm
        
        # Correction prompt with clear instructions
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert correction and refinement agent.

YOUR MISSION:
Take flawed or incomplete output and transform it into high-quality, polished results.

CORRECTION PROCESS:
1. READ the validation feedback carefully
2. IDENTIFY what's missing or wrong
3. FIX each issue systematically
4. ENHANCE overall quality and clarity
5. VERIFY all issues are resolved

QUALITY STANDARDS:
- Complete: No missing information
- Accurate: All facts are correct
- Relevant: Directly addresses the query
- Clear: Well-organized and easy to understand
- Professional: Polished and ready to use

IMPORTANT:
- Don't just patch - improve holistically
- Add missing context and details
- Reorganize for better flow
- Use clear, professional language
- Make it better than just "passing"

Return ONLY the corrected output, nothing else."""),
            ("user", """Validation Feedback:
Issues: {issues}

Original Output:
{output}

Original Query: {query}

Produce a corrected, high-quality version that addresses ALL issues.""")
        ])
    
    def correct(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply corrections to produce final output.
        
        This method:
        1. Checks if corrections are needed
        2. If not, uses execution results as-is
        3. If yes, applies LLM-based corrections
        4. Produces final polished output
        5. Updates state with result
        
        Args:
            state: Current state with corrections_needed and execution_results.
        
        Returns:
            Updated state with final_output populated.
        
        Side Effects:
            - Sets state["final_output"]
            - Adds correction message to state["messages"]
        
        Example:
            >>> state = {
            ...     "corrections_needed": ["Add sources"],
            ...     "execution_results": {"subtask_0": "Analysis done"}
            ... }
            >>> new_state = corrector.correct(state)
            >>> print(new_state["final_output"])
            
        Flow:
            - If no issues: final_output = execution_results
            - If issues: final_output = corrected version
        """
        # Check if corrections are needed
        if not state.get("corrections_needed"):
            # No issues - use execution results directly
            results = state.get("execution_results", {})
            state["final_output"] = "\n\n".join(
                str(v) for v in results.values()
            )
            state["messages"].append({
                "role": "corrector",
                "content": "No corrections needed - output is good!"
            })
            return state
        
        # Corrections needed - apply fixes
        original_query = state["messages"][0]["content"]
        results = state.get("execution_results", {})
        original_output = "\n\n".join(
            str(v) for v in results.values()
        )
        
        # Create correction chain
        chain = self.prompt | self.llm
        
        try:
            # Invoke correction
            corrected = chain.invoke({
                "issues": "\n- ".join(state["corrections_needed"]),
                "output": original_output,
                "query": original_query
            })
            
            # Update state with corrected output
            state["final_output"] = corrected.content
            
            # Add trace message
            state["messages"].append({
                "role": "corrector",
                "content": (
                    f"Applied corrections to address "
                    f"{len(state['corrections_needed'])} issues"
                )
            })
            
        except Exception as e:
            # If correction fails, use original with warning
            state["final_output"] = (
                f"[Correction failed: {str(e)}]\n\n"
                f"Original output:\n{original_output}"
            )
            state["messages"].append({
                "role": "corrector",
                "content": f"Correction failed: {str(e)}"
            })
        
        return state