# inference.py
#
# Assesses the performance of the LLM using recommended hyperparameters
# and can save the generated configuration to a file.
#
# Usage:
#   # Just display the output and score
#   python inference.py --freq 1.85
#
#   # Display, score, AND save the config to a file
#   python inference.py --freq 1.85 --output config_1.85.yaml

import torch
import yaml
import re
import argparse
from transformers import (
    AutoTokenizer, GenerationConfig,
    AutoModelForCausalLM
)
from simulation_env import mock_run_simulation_and_get_reward

def _parse_llm_output(text: str) -> dict:
    """
    Parses the full text from the LLM to find and decode the YAML block.
    """
    try:
        if "### YAML Output:" in text:
            yaml_string = text.split("### YAML Output:")[-1].strip()
            if yaml_string.startswith("```yaml"):
                yaml_string = yaml_string.split("```yaml\n", 1)[-1]
                if "```" in yaml_string:
                     yaml_string = yaml_string.rsplit("```", 1)[0]
            config = yaml.safe_load(yaml_string)
            if isinstance(config, dict):
                return config
    except (yaml.YAMLError, IndexError) as e:
        print(f"[Parser Error] Failed to parse YAML: {e}")
        pass
    return None

def assess_model_performance(model, tokenizer, frequency: float, generation_config: GenerationConfig) -> (dict, float):
    """
    Generates a configuration, parses it, and gets a score from the simulation.
    Returns the generated config dictionary and its score.
    """
    PROMPT_TEMPLATE = """You are a highly skilled RF engineer specializing in electronic countermeasures.
Your mission is to generate a complete YAML configuration file to effectively jam a target frequency.
### Instructions:
1.  Analyze the `High-Level Goal`.
2.  Determine the optimal values for **all** required configuration parameters.
3.  The output MUST be a single, valid YAML block containing all necessary keys.
4.  Use snake_case for all keys (e.g., `center_frequency`).
5.  Use scientific 'e' notation for frequencies and bandwidth.
6.  Do not output anything other than the given set of keys.
### Example:
High-Level Goal: Jam a target at 0.915 GHz
### Example YAML Output:
amplitude: 0.9
amplitude_width: 0.1
center_frequency: 9.15e8
bandwidth: 10e6
initial_phase: 0
sampling_freq: 20e6
num_samples: 20000
output_iq_file: "output.fc32"
output_csv_file: "output.csv"
write_iq: false
write_csv: true
device_args: "type=b200"
tx_gain: 55
---
### Current Task:
High-Level Goal: Jam a target at {freq:.4f} GHz
### YAML Output:
"""
    prompt_text = PROMPT_TEMPLATE.format(freq=frequency)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    print(f"\nGenerating configuration for {frequency} GHz...")
    output_tokens = model.generate(**inputs, generation_config=generation_config)
    full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    parsed_config = _parse_llm_output(full_text)
    
    print("Assessing performance with the simulation environment...")
    score = mock_run_simulation_and_get_reward(parsed_config) if parsed_config else -1.0
        
    return parsed_config, score

def main():
    parser = argparse.ArgumentParser(description="Assess a tuned LLM's performance and save the output.")
    parser.add_argument("--freq", type=float, required=True, help="Target frequency in GHz (e.g., 1.85).")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-Coder-6.7B-Instruct", help="Base model to use.")
    # --- NEW ARGUMENT FOR SAVING THE FILE ---
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the generated YAML config file (e.g., config.yaml).")
    args = parser.parse_args()

    # --- Step 1: Load Model ---
    print("="*20 + " LOADING BASE MODEL (HIGH PRECISION) " + "="*20)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("="*20 + " MODEL LOADED " + "="*20)

    # --- Step 2: Define Optimal Generation Config ---
    recommended_generation_config = GenerationConfig(
        max_new_tokens=250, pad_token_id=tokenizer.eos_token_id, do_sample=True,
        # --- RECOMMENDED HYPERPARAMETERS ---
        temperature=0.7, top_p=0.85, repetition_penalty=1.1
    )
    print("\nUsing Recommended Hyperparameters:")
    print(f"  Temperature: {recommended_generation_config.temperature}, Top_p: {recommended_generation_config.top_p}, Repetition Penalty: {recommended_generation_config.repetition_penalty}")

    # --- Step 3: Generate and Assess ---
    final_config, score = assess_model_performance(
        model=model, tokenizer=tokenizer, frequency=args.freq,
        generation_config=recommended_generation_config
    )

    # --- Step 4: Display and Save Results ---
    print("\n" + "="*20 + " ASSESSMENT COMPLETE " + "="*20)
    print(f"Target Frequency: {args.freq} GHz\n")
    
    print("--- Generated Configuration ---")
    if final_config:
        # Print to console
        print(yaml.dump(final_config, sort_keys=False, default_flow_style=False))

        # --- NEW: Save the configuration to a file if an output path is provided ---
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    yaml.dump(final_config, f, sort_keys=False, default_flow_style=False)
                print(f"\nConfiguration successfully saved to: {args.output}")
            except IOError as e:
                print(f"\n[Error] Could not save configuration to file: {e}")
        # --- END NEW CODE ---

    else:
        print("Failed to generate a valid YAML configuration.")

    print("\n--- Performance Score ---")
    print(f"Score from Simulation: {score:+.4f}")
    print("="*31)


if __name__ == "__main__":
    main()