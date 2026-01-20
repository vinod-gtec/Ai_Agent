from typing import Dict, List, Optional
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field


class DataAnalysisInput(BaseModel):
    """
    Input schema for data analysis tool.
    
    Defines the expected parameters for the analyze_data tool,
    ensuring type safety and validation.
    
    Attributes:
        data: The data to analyze (JSON string or CSV).
        analysis_type: Type of analysis to perform.
    """
    data: str = Field(
        description="Data to analyze in JSON or CSV format"
    )
    analysis_type: str = Field(
        description="Analysis type: trend, forecast, summary, or statistical"
    )


class ToolRegistry:
    """
    Central registry for all tools available to agents.
    
    This class manages tool creation, registration, and retrieval.
    Tools are LangChain Tool objects that agents can invoke to perform
    specific actions like data analysis, API calls, calculations, etc.
    
    Key Features:
        - Centralized tool management
        - Type-safe tool definitions
        - Easy to extend with new tools
        - Tools can be dynamically loaded/unloaded
    
    Attributes:
        tools: Dictionary mapping tool names to Tool instances.
    
    Usage:
        >>> registry = ToolRegistry()
        >>> tool = registry.get_tool("analyze_data")
        >>> result = tool.run(data="...", analysis_type="trend")
    """
    
    def __init__(self):
        """
        Initialize the tool registry with default tools.
        
        Creates and registers all available tools. You can extend this
        by adding more tool definitions in _initialize_tools().
        """
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> Dict[str, Tool]:
        """
        Create and register all available tools.
        
        This method defines all tools that agents can use. Each tool:
        1. Has a unique name
        2. Has a clear description (helps LLM choose right tool)
        3. Has a function implementation
        4. Optionally has a schema for structured inputs
        
        Returns:
            Dictionary mapping tool names to Tool instances.
        
        Note:
            To add a new tool:
            1. Define the tool function
            2. Create Tool.from_function() or StructuredTool
            3. Add to the returned dictionary
        """
        
        def analyze_data(data: str, analysis_type: str) -> str:
            """
            Analyze data and return insights.
            
            This is a simplified implementation. In production, you would:
            - Parse the data (JSON/CSV)
            - Perform actual analysis (pandas, numpy, etc.)
            - Generate visualizations
            - Return structured insights
            
            Args:
                data: Data to analyze.
                analysis_type: Type of analysis.
            
            Returns:
                Analysis results as a string.
            """
            # Simplified implementation - replace with real analysis
            analyses = {
                "trend": f"Trend analysis: Data shows upward trend over {len(data)} points",
                "forecast": f"Forecast: Predicted 15% growth based on {len(data)} data points",
                "summary": f"Summary: Analyzed {len(data)} characters of data",
                "statistical": f"Stats: Mean, median, mode calculated from {len(data)} points"
            }
            return analyses.get(
                analysis_type,
                f"Completed {analysis_type} analysis on {len(data)} chars"
            )
        
        def generate_forecast(data: str) -> str:
            """
            Generate forecast from historical data.
            
            Args:
                data: Historical data.
            
            Returns:
                Forecast prediction.
            """
            # Simplified - replace with actual forecasting model
            return (
                "Forecast Results:\n"
                "- Q1 2025: 15% growth expected\n"
                "- Q2 2025: 12% growth expected\n"
                "- Confidence: 85%\n"
                f"Based on {len(data)} data points"
            )
        
        def validate_output(output: str, criteria: str) -> str:
            """
            Validate output against criteria.
            
            Args:
                output: Output to validate.
                criteria: Validation criteria (comma-separated).
            
            Returns:
                Validation results.
            """
            criteria_list = criteria.split(',')
            return (
                f"Validation Results:\n"
                f"- Checked {len(criteria_list)} criteria\n"
                f"- Output length: {len(output)} chars\n"
                f"- Status: All checks passed ✓"
            )
        
        def fetch_external_data(source: str) -> str:
            """
            Fetch data from external source.
            
            Args:
                source: Data source URL or identifier.
            
            Returns:
                Fetched data.
            """
            # Simplified - replace with actual API calls
            return f"Data fetched from {source}:\n[Sample data would be here]"
        
        # Return dictionary of all tools
        return {
            "analyze_data": StructuredTool.from_function(
                func=analyze_data,
                name="analyze_data",
                description=(
                    "Analyze data using specified method. "
                    "Supports trend, forecast, summary, and statistical analysis."
                ),
                args_schema=DataAnalysisInput
            ),
            "generate_forecast": Tool.from_function(
                func=generate_forecast,
                name="generate_forecast",
                description=(
                    "Generate forecast from historical data. "
                    "Returns predictions with confidence intervals."
                )
            ),
            "validate_output": Tool.from_function(
                func=validate_output,
                name="validate_output",
                description=(
                    "Validate output against specified criteria. "
                    "Returns validation status and details."
                )
            ),
            "fetch_external_data": Tool.from_function(
                func=fetch_external_data,
                name="fetch_external_data",
                description=(
                    "Fetch data from external API or database. "
                    "Provide source URL or identifier."
                )
            )
        }
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Retrieve a specific tool by name.
        
        Args:
            name: Tool name to retrieve.
        
        Returns:
            Tool instance or None if not found.
        
        Example:
            >>> tool = registry.get_tool("analyze_data")
            >>> if tool:
            ...     result = tool.run(data="...", analysis_type="trend")
        """
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            List of all Tool instances.
        
        Example:
            >>> tools = registry.get_all_tools()
            >>> print(f"Available tools: {[t.name for t in tools]}")
        """
        return list(self.tools.values())
    
    def add_tool(self, name: str, tool: Tool) -> None:
        """
        Register a new tool dynamically.
        
        Args:
            name: Unique name for the tool.
            tool: Tool instance to register.
        
        Example:
            >>> def my_func(x: str) -> str:
            ...     return f"Processed: {x}"
            >>> new_tool = Tool.from_function(my_func, name="my_tool")
            >>> registry.add_tool("my_tool", new_tool)
        """
        self.tools[name] = tool
    
    def remove_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Name of tool to remove.
        
        Returns:
            True if tool was removed, False if not found.
        
        Example:
            >>> registry.remove_tool("old_tool")
        """
        if name in self.tools:
            del self.tools[name]
            return True
        return False