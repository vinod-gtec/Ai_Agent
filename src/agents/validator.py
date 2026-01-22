from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from src.api.schemas import ValidationResult


class ValidatorAgent:
    """
    Validator Agent - Ensures output quality meets standards.
    
    The Validator is the third agent in the workflow. It examines the
    results from the Executor and checks them against quality criteria.
    
    Key Responsibilities:
        - Verify output completeness
        - Check for accuracy and relevance
        - Identify missing information
        - Assess overall quality
        - Provide specific feedback for corrections
    
    Workflow Position:
        User Query → Planner → Executor → [VALIDATOR] → Corrector (if needed)
    
    Validation Criteria:
        - Completeness: All subtasks produced results
        - Accuracy: Results appear correct and logical
        - Relevance: Output addresses the original query
        - Coherence: Results form a cohesive answer
    
    Attributes:
        llm: Language model for quality assessment.
        parser: Parser for structured validation results.
        prompt: Prompt template for validation.
    
    Example:
        >>> validator = ValidatorAgent(llm)
        >>> state = {"execution_results": {"subtask_0": "Result here"}}
        >>> updated_state = validator.validate(state)
        >>> print(updated_state["validation_status"])  # "passed" or "failed"
    """
    
    def __init__(self, llm):
        """
        Initialize the Validator agent.
        
        Args:
            llm: Language model instance for validation reasoning.
        """
        self.llm = llm
        
        # Parser for structured validation output
        self.parser = PydanticOutputParser(pydantic_object=ValidationResult)
        
        # Validation prompt with clear criteria
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality validation agent. Your job is to ensure outputs meet high standards.

VALIDATION CRITERIA:
1. COMPLETENESS: Are all required elements present?
2. ACCURACY: Does the information appear correct?
3. RELEVANCE: Does it address the original question?
4. COHERENCE: Do all parts fit together logically?
5. CLARITY: Is it well-organized and understandable?

VALIDATION PROCESS:
- Review each piece of output carefully
- Check against all criteria
- Assign confidence score (0.0-1.0)
- List specific issues if found
- Provide actionable suggestions

OUTPUT REQUIREMENTS:
- is_valid: true only if ALL criteria are met
- confidence: how sure you are (0.0-1.0)
- issues: specific problems found (empty if valid)
- suggestions: concrete improvements

{format_instructions}

Be thorough but fair. Minor issues don't mean automatic failure."""),
            ("user", """Original Query: {query}

Execution Results to Validate:
{output}

Perform thorough validation.""")
        ])
    
    def validate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the execution results for quality.
        
        This method:
        1. Combines all execution results
        2. Compares against the original query
        3. Checks all validation criteria
        4. Generates structured feedback
        5. Updates state with validation status
        
        Args:
            state: Current state with execution_results populated.
        
        Returns:
            Updated state with validation_status and corrections_needed.
        
        Side Effects:
            - Sets state["validation_status"] to "passed" or "failed"
            - Populates state["corrections_needed"] with issues
            - Adds validation message to state["messages"]
        
        Example:
            >>> state = {
            ...     "messages": [{"role": "user", "content": "Analyze sales"}],
            ...     "execution_results": {"subtask_0": "Analysis complete"}
            ... }
            >>> new_state = validator.validate(state)
            >>> if new_state["validation_status"] == "failed":
            ...     print(new_state["corrections_needed"])
        
        Note:
            The validator is strict - even minor issues may cause failure.
            This ensures high-quality final output.
        """
        # Get the original query
        original_query = state["messages"][0]["content"]
        
        # Combine all execution results
        results = state.get("execution_results", {})
        combined_output = "\n\n".join(
            f"Subtask {k}: {v}"
            for k, v in results.items()
        )
        
        # If no results, automatic failure
        if not combined_output.strip():
            state["validation_status"] = "failed"
            state["corrections_needed"] = ["No execution results found"]
            state["messages"].append({
                "role": "validator",
                "content": "Validation failed: No results to validate"
            })
            return state
        
        # Create validation chain
        chain = self.prompt | self.llm | self.parser
        
        try:
            # Invoke validation
            validation = chain.invoke({
                "query": original_query,
                "output": combined_output,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Update state based on validation result
            state["validation_status"] = "passed" if validation.is_valid else "failed"
            state["corrections_needed"] = validation.issues
            state["metadata"]["validation_confidence"] = validation.confidence
            
            # Add trace message
            status_emoji = "✓" if validation.is_valid else "✗"
            state["messages"].append({
                "role": "validator",
                "content": (
                    f"Validation {state['validation_status']} {status_emoji} "
                    f"(confidence: {validation.confidence:.2f}). "
                    f"Issues found: {len(validation.issues)}"
                )
            })
            
        except Exception as e:
            # If validation itself fails, mark as failed
            state["validation_status"] = "failed"
            state["corrections_needed"] = [f"Validation error: {str(e)}"]
            state["messages"].append({
                "role": "validator",
                "content": f"Validation failed with error: {str(e)}"
            })
        
        return state
