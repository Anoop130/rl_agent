# inference.py
#
# This script uses the optimal hyperparameters discovered by the auto-tuner
# to generate a YAML configuration for a specific target frequency.
#
# Usage:
# python inference.py --freq 1.85
# python inference.py --freq 2.44 --model "another/model"

import torch
import yaml
import re
import argparse
from transformers import (
    AutoTokenizer, GenerationConfig,
    AutoModelForCausalLM
)

def _parse_llm_output(text: str) -> dict:
    """
    Parses the full text from the LLM to find and decode the YAML block.
    """
    try:
        # The model output contains the prompt, so we split by the final instruction
        if "### YAML Output:" in text:
            yaml_string = text.split("### YAML Output:")[-1].strip()
            
            # Handle the case where the model wraps the output in markdown
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

def generate_config(model, tokenizer, frequency: float, generation_config: GenerationConfig) -> dict:
    """
    Generates and parses the configuration for a single target frequency.
    """
    PROMPT_TEMPLATE = """You are a highly skilled RF engineer specializing in electronic countermeasures.
Your mission is to generate a complete YAML configuration file to effectively jam a target frequency.
### Instructions:
1.  Analyze the `High-Level Goal`.
2.  Determine the optimal values for **all** required configuration parameters.
3.  The output MUST be a single, valid YAML block containing all necessary keys.
4.  Use snake_case for all keys (e.g., `center_frequency`).
5.  Use scientific 'e' notation for frequencies and bandwidth.
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
    # Create the prompt for the specified frequency
    prompt_text = PROMPT_TEMPLATE.format(freq=frequency)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Generate the output
    print(f"\nGenerating configuration for {frequency} GHz...")
    output_tokens = model.generate(**inputs, generation_config=generation_config)
    
    # Decode and parse
    full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    parsed_config = _parse_llm_output(full_text)
    
    return parsed_config

def main():
    parser = argparse.ArgumentParser(description="Generate RF jamming configuration using a tuned LLM.")
    parser.add_argument("--freq", type=float, required=True, help="Target frequency in GHz (e.g., 1.85).")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-Coder-6.7B-Instruct", help="Base model to use.")
    args = parser.parse_args()

    # --- Step 1: Load the pre-trained model and tokenizer ---
    print("="*20 + " LOADING BASE MODEL (HIGH PRECISION) " + "="*20)
    print(f"Using model: {args.model}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16, # Use high-precision bfloat16
        device_map="auto",
        # You can add trust_remote_code=True if the model requires it
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("="*20 + " MODEL LOADED " + "="*20)

    # --- Step 2: Define the optimal generation configuration ---
    # These are the best hyperparameters found by the auto-tuner script.
    optimal_generation_config = GenerationConfig(
        max_new_tokens=250,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        # --- BEST PERFORMING PARAMETERS ---
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.1
    )

    # --- Step 3: Generate and print the configuration ---
    final_config = generate_config(
        model=model,
        tokenizer=tokenizer,
        frequency=args.freq,
        generation_config=optimal_generation_config
    )

    print("\n" + "="*20 + " FINAL CONFIGURATION " + "="*20)
    if final_config:
        # Use yaml.dump for clean, multi-line YAML formatting
        print(yaml.dump(final_config, sort_keys=False))
    else:
        print("Failed to generate a valid configuration. The model may have produced invalid YAML.")
        print("Consider re-running or checking the model's output.")

if __name__ == "__main__":
    main()