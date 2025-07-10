# inference.py (Robust Final Version)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import re
import json
import os

# Import your simulation environment to test the output
from simulation_environment import mock_run_simulation_and_get_reward
import simulation_environment

def parse_llm_output(text_output: str) -> dict:
    """A simple function to extract the JSON from the model's raw output."""
    try:
        # Use a robust regex to find a JSON block
        json_match = re.search(r'\{[^{}]*\}', text_output, re.DOTALL)
        if json_match:
            # Basic validation that braces are balanced
            json_str = json_match.group(0)
            if json_str.count("{") == json_str.count("}"):
                return json.loads(json_str)
        print("\n[Warning] No valid JSON object found in the output.")
        return None
    except json.JSONDecodeError:
        print(f"\n[Warning] Could not decode the extracted JSON string.")
        return None

def main(args):
    # --- 1. Sanitize the model path for local loading ---
    # Get the absolute path to resolve any ambiguity.
    model_path = os.path.abspath(args.model_path)
    
    # Check if the directory actually exists before trying to load
    if not os.path.isdir(model_path):
        print(f"FATAL ERROR: The specified model path does not exist or is not a directory.")
        print(f"Checked path: {model_path}")
        return

    print(f"--- Loading fine-tuned model from local path: {model_path} ---")
    
    # Use the same quantization config as in training for consistency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # --- 2. Load the model and tokenizer with local_files_only=True ---
    # This forces the library to treat the path as local and skip Hub validation.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True  # <--- THIS IS THE KEY FIX
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True  # <--- ALSO USE FOR THE TOKENIZER
        )
    except Exception as e:
        print("\n--- FATAL ERROR DURING MODEL LOADING ---")
        print(f"An error occurred: {e}")
        print("Please ensure the model path is correct and contains all necessary files (config.json, model.safetensors, etc.).")
        return

    model.eval() # Set the model to evaluation mode
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # --- 3. Create a prompt ---
    base_prompt_template = (
        "You are a world-class network security expert specializing in radio "
        "frequency analysis. Your task is to generate a precise JSON "
        "configuration for a radio jammer to neutralize a target signal. "
        "The JSON object must contain three keys: 'center_frequency' (float, in Hz), "
        "'bandwidth' (float, in Hz), and 'tx_gain' (float, from 0-90). "
        "Adhere strictly to the JSON format. Do not provide any other text, "
        "explanation, or markdown. Your entire output must be only the JSON object."
        "\n\n### Current Mission\n"
        "Your current mission is to generate a config to jam a target at "
        "{target_freq_ghz:.4f} GHz.\n\n### JSON Output:\n"
    )
    
    prompt = base_prompt_template.format(target_freq_ghz=args.target_freq)
    print(f"\n--- Sending Prompt to Model ---\n{prompt}")

    # --- 4. Generate the configuration ---
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,  # Lower temperature for more confident output
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    generated_part = response_text[len(prompt):]
    
    print("\n--- Model's Raw Output ---")
    print(generated_part)

    # --- 5. Parse, TEST, and display the result ---
    json_config = parse_llm_output(generated_part)

    if json_config:
        print("\n--- Parsed JSON Configuration ---")
        print(json.dumps(json_config, indent=2))

        # Test the configuration and get the reward score
        try:
            # Set the global target frequency for the simulation
            simulation_environment.TARGET_FREQ = args.target_freq * 1e9
            reward_score = mock_run_simulation_and_get_reward(json_config)
            
            print("\n--- Test Result ---")
            print(f"Simulation Reward Score: {reward_score:.4f}")
            
        except Exception as e:
            print(f"\n--- Error during simulation testing ---")
            print(f"Could not calculate reward: {e}")
    else:
        print("\n--- Failed to get a valid configuration from model output ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and test a jammer config using a fine-tuned agent.")
    parser.add_argument(
        "--model_path",
        type=str,
        # This default path assumes you run a final training that saves to this folder.
        # Example: a final_trainer.py or the original llm_rl_agent.py
        default="./final_jammer_agent",
        help="Local path to the directory containing the saved fine-tuned model."
    )
    parser.add_argument(
        "--target_freq",
        type=float,
        default=3.65,
        help="The target frequency to jam, in GHz (e.g., 3.65)."
    )
    main(parser.parse_args())