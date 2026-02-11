# agent_orchestrator.py
import json
from tools import AVAILABLE_TOOLS
# from your_llm_module import call_llm # Assume you have a function to call the LLM

def run_agent(user_request):
    # This is the agent's short-term memory. It grows with each step.
    conversation_history = [
        {"role": "user", "content": f"Goal: {user_request}"}
    ]
    
    # Construct the System Prompt that teaches the LLM how to be an agent
    system_prompt = f"""
    You are an expert 5G systems orchestrator. Your goal is to fulfill the user's request.
    You operate in a loop of Thought -> Action.
    At each step, you must respond with a JSON object containing your 'thought' and the 'action' to take next.
    The action must be a call to one of the available tools.

    Available Tools:
    {json.dumps({name: details['description'] for name, details in AVAILABLE_TOOLS.items()}, indent=2)}

    Your response MUST be in this exact JSON format:
    {{
      "thought": "Your reasoning for the next step.",
      "action": {{
        "tool_name": "name_of_the_tool_to_use",
        "parameters": {{ "arg1": "value1", ... }}
      }}
    }}
    
    If you have fulfilled the user's request, respond with:
    {{
      "thought": "I have completed all necessary steps.",
      "action": {{ "tool_name": "finish", "parameters": {{ "final_message": "Success!" }} }}
    }}
    """
    
    # Add the system prompt to the start of the memory
    conversation_history.insert(0, {"role": "system", "content": system_prompt})

    # The main ReAct loop
    for _ in range(10): # Add a safety break after 10 steps
        print("\n===== AGENT TURN =====\n")
        
        # 1. THINK: Ask the LLM for the next step
        # llm_response_str = call_llm(conversation_history) 
        # For demo, we'll simulate the LLM's response
        llm_response_str = simulate_llm_response(conversation_history)
        
        print(f"LLM Response:\n{llm_response_str}")
        conversation_history.append({"role": "assistant", "content": llm_response_str})
        
        # 2. PARSE ACTION
        try:
            llm_response_json = json.loads(llm_response_str)
            action = llm_response_json.get("action", {})
            tool_name = action.get("tool_name")
            parameters = action.get("parameters", {})
        except json.JSONDecodeError:
            print("Error: LLM did not return valid JSON.")
            conversation_history.append({"role": "system", "content": "Observation: Your response was not valid JSON. Please correct it."})
            continue

        # 3. ACT: Execute the chosen tool
        if tool_name == "finish":
            print(f"\nAGENT FINISHED: {parameters.get('final_message')}")
            break
            
        if tool_name in AVAILABLE_TOOLS:
            tool_function = AVAILABLE_TOOLS[tool_name]["function"]
            try:
                # This is where the magic happens: calling the actual Python function
                observation = tool_function(**parameters)
                observation_str = json.dumps(observation)
            except Exception as e:
                observation_str = f"Error executing tool {tool_name}: {e}"
            
            print(f"Observation from tool '{tool_name}':\n{observation_str}")
            conversation_history.append({"role": "system", "content": f"Observation: {observation_str}"})
        else:
            print(f"Error: LLM chose an unknown tool: {tool_name}")
            conversation_history.append({"role": "system", "content": f"Observation: The tool '{tool_name}' does not exist."})
    else:
        print("AGENT STOPPED: Reached maximum number of steps.")

# You would need a function to simulate the LLM's intelligent choices based on history
def simulate_llm_response(history):
    # This is a dummy function. A real LLM would make these choices.
    last_message = history[-1]['content']
    if "Goal:" in last_message:
        return '{"thought": "First, I need to understand what components the user wants. I will use the get_component_plan tool.", "action": {"tool_name": "get_component_plan", "parameters": {"user_request": "Start the sniffer and rtue"}}}'
    if "rtue" in last_message and "sniffer" in last_message:
        return '{"thought": "The plan is to deploy rtue and sniffer. I will start by generating the config for rtue.", "action": {"tool_name": "generate_config_json", "parameters": {"component_name": "rtue", "user_request": "Start the sniffer and rtue"}}}'
    if "generate_config_json" in history[-2]['content']:
        return '{"thought": "I have generated a config for rtue. Now I must validate it before doing anything else.", "action": {"tool_name": "validate_config", "parameters": {"component_name": "rtue", "config_json": "{\\"sample_rate\\": 2.5e9, \\"frequency\\": 9.51e9, \\"pdcch_coreset_duration\\": 100}"}}}'
    # Add more logic here to simulate the full flow
    return '{"thought": "I am stuck.", "action": {"tool_name": "finish", "parameters": {"final_message": "Error: Unknown state."}}}'


# Start the agent
run_agent("Start the sniffer and rtue")