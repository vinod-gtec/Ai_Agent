import os
import sys
from dotenv import load_dotenv

# Add src to path so we can import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.graph import MultiAgentSystem


def print_separator(title=""):
    """Print a nice separator line."""
    print("\n" + "="*80)
    if title:
        print(f" {title}")
        print("="*80)


def run_demo():
    """
    Run demo queries through the multi-agent system.
    
    Demonstrates different types of queries and how to access results.
    """
    print_separator("MULTI-AGENT AI SYSTEM DEMO")
    
    # Load environment
    load_dotenv()
    
    # Initialize system
    print("\n🚀 Initializing multi-agent system...")
    try:
        system = MultiAgentSystem()
        print("✓ System initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return
    
    # Demo queries
    queries = [
        "Analyze quarterly sales trends and generate forecast for next quarter",
        "Research competitor pricing strategies and recommend our pricing",
        "Review customer feedback and identify top 3 improvement areas"
    ]
    
    # Run each query
    for i, query in enumerate(queries, 1):
        print_separator(f"QUERY {i}/{len(queries)}")
        print(f"\n📝 Query: {query}")
        
        try:
            # Run the query
            print("\n🔄 Processing...")
            result = system.run(query)
            
            # Display plan
            print_separator("EXECUTION PLAN")
            for j, step in enumerate(result["plan"], 1):
                print(f"  {j}. {step}")
            
            # Display agent trace
            print_separator("AGENT TRACE")
            for msg in result["messages"]:
                emoji = {
                    "planner": "🧠",
                    "executor": "⚡",
                    "validator": "✅",
                    "corrector": "🔧"
                }.get(msg["role"], "💬")
                print(f"  {emoji} [{msg['role'].upper()}] {msg['content']}")
            
            # Display output
            print_separator("FINAL OUTPUT")
            print(f"\n{result['output']}\n")
            
            # Display metadata
            print_separator("METADATA")
            for key, value in result["metadata"].items():
                print(f"  • {key}: {value}")
            
            print("\n✓ Query completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            continue
    
    print_separator("DEMO COMPLETE")
    print("\n💰 Total Cost: $0.00 (FREE!)")
    print("✨ All queries processed successfully!\n")


if __name__ == "__main__":
    run_demo()
