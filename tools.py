# tools.py

def get_component_plan(user_request: str) -> list[str]:
    """
    Analyzes a user's request and returns a JSON list of components to deploy.
    Example: "Start the sniffer and rtue" -> ["rtue", "sniffer"]
    """
    # Your existing code to call the Planner LLM goes here...
    print(f"TOOL: Executing get_component_plan for: {user_request}")
    # ...
    return ["rtue", "sniffer"]

def generate_config_json(component_name: str, user_request: str) -> str:
    """
    Generates a full JSON configuration for a single given component.
    """
    # Your existing code to call the Generator LLM goes here...
    print(f"TOOL: Executing generate_config_json for: {component_name}")
    # ...
    # For demonstration, returning a flawed JSON string
    return '{"sample_rate": 2.5e9, "frequency": 9.51e9, "pdcch_coreset_duration": 100}'

def validate_config(component_name: str, config_json: str) -> dict:
    """
    Validates a generated JSON config against a set of rules.
    Returns a dictionary with 'is_valid': True/False and an 'errors' list if applicable.
    """
    # Your existing validator logic goes here...
    print(f"TOOL: Executing validate_config for: {component_name}")
    # ...
    # For demonstration, simulating a validation failure
    errors = [
        "CRITICAL: Invalid sample rate 2500000000.0",
        "CRITICAL: Invalid coreset duration 100"
    ]
    return {"is_valid": False, "errors": errors}

def deploy_to_controller(component_name: str, final_config: str):
    """
    Deploys a final, validated configuration by making an API call.
    Use this tool ONLY when a configuration has been successfully validated.
    """
    print(f"TOOL: DEPLOYING {component_name}!")
    # Your API POST request logic goes here...
    return {"status": "success", "message": f"{component_name} deployed."}

# A dictionary to map tool names to their function and description
AVAILABLE_TOOLS = {
    "get_component_plan": {
        "function": get_component_plan,
        "description": "Analyzes a user's request and returns a JSON list of components to deploy."
    },
    "generate_config_json": {
        "function": generate_config_json,
        "description": "Generates a full JSON configuration for a single given component."
    },
    "validate_config": {
        "function": validate_config,
        "description": "Validates a generated JSON config against a set of rules. Returns validation status and errors."
    },
    "deploy_to_controller": {
        "function": deploy_to_controller,
        "description": "Deploys a final, validated configuration to the system. ONLY use after successful validation."
    }
}
