"""
Example workflow functions for testing step outputs access.
"""

def compute_something(greeting: str) -> dict:
    """Compute something and return a result with greeting."""
    return {
        "greeting": greeting,
        "result": f"Computed result for {greeting}"
    }

def process_results(greeting: str, analysis: str) -> dict:
    """Process results and return a final result."""
    return {
        "result": f"Processed {greeting} with analysis: {analysis}"
    }

def debug_context(context_info: str) -> str:
    """Debug function to print context information."""
    print(f"DEBUG: {context_info}")
    return f"Debug info: {context_info}"

def debug_full_context(**kwargs) -> dict:
    """Debug function to print all available context."""
    print("DEBUG FULL CONTEXT:")
    for key, value in kwargs.items():
        print(f"  {key}: {value} (type: {type(value)})")
    return {"debug_result": "Context debug completed"}