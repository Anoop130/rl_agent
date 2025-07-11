# bayesian_tuner_colab.py
#
# This script has been optimized to run on memory-constrained environments
# like Google Colab by implementing 4-bit quantization and reducing batch sizes.

import torch
import yaml
import re
import argparse
from transformers import (
    AutoTokenizer, GenerationConfig,
    AutoModelForCausalLM, BitsAndBytesConfig # Added BitsAndBytesConfig
)
from simulation_env import mock_run_simulation_and_get_reward

# Imports for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# ===================================================================
# SECTION 1: REUSABLE HELPER FUNCTIONS (Unchanged)
# ===================================================================

def _parse_llm_output(text: str) -> dict:
    """Parses the full text from the LLM to find and decode the YAML block."""
    try:
        if "### YAML Output:" in text:
            yaml_string = text.split("### YAML Output:")[-1].strip()
            if yaml_string.startswith("```yaml"):
                yaml_string = yaml_string.split("```yaml\n", 1)[-1]
                if "```" in yaml_string:
                     yaml_string = yaml_string.rsplit("```", 1)[0]
            config = yaml.safe_load(yaml_string)
            if isinstance(config, dict): return config
    except (yaml.YAMLError, IndexError): pass
    return None

def execute_generation_run(hparams: dict, model, tokenizer) -> float:
    """
    Takes a dictionary of hyperparameters, generates text, and returns a
    score from the simulation.
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
    # --- MEMORY SOLUTION: Reduce the number of parallel prompts ---
    test_frequencies = [1.83, 1.85, 1.88, 1.90] # Reduced from 8 to 4
    
    generation_config = GenerationConfig(max_new_tokens=250, pad_token_id=tokenizer.eos_token_id, do_sample=True, **hparams)
    
    prompts = [PROMPT_TEMPLATE.format(freq=f) for f in test_frequencies]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    output_tokens = model.generate(**inputs, generation_config=generation_config)
    full_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    total_score = 0
    for text in full_texts:
        config = _parse_llm_output(text)
        score = mock_run_simulation_and_get_reward(config)
        total_score += score

    avg_score = total_score / len(test_frequencies)
    return avg_score

# ===================================================================
# SECTION 2: BAYESIAN OPTIMIZATION SETUP
# ===================================================================

space  = [
    Real(0.5, 1.0, name='temperature'),
    Real(0.8, 1.0, name='top_p'),
    Real(1.0, 1.2, name='repetition_penalty')
]

model = None
tokenizer = None

@use_named_args(space)
def objective_function(**params):
    """
    Wrapper function for the optimizer. It runs the simulation and returns
    a score to be minimized.
    """
    global model, tokenizer
    print(f"Testing Parameters: temp={params['temperature']:.3f}, top_p={params['top_p']:.3f}, rep_pen={params['repetition_penalty']:.3f}")
    
    score = execute_generation_run(hparams=params, model=model, tokenizer=tokenizer)
    
    # --- MEMORY SOLUTION: Clear CUDA cache after each run ---
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"--> Achieved Score: {score:.4f}")
    return -score

# ===================================================================
# SECTION 3: MAIN CONTROLLER
# ===================================================================

def main():
    global model, tokenizer

    parser = argparse.ArgumentParser(description="Bayesian Optimization for LLM Generation (Colab Optimized).")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-Coder-6.7B-Instruct", help="Base model to use.")
    parser.add_argument("--n_calls", type=int, default=30, help="Number of optimization runs.")
    args = parser.parse_args()

    # --- MEMORY SOLUTION: Load model with 4-bit quantization ---
    print("="*20 + " LOADING BASE MODEL (4-bit Quantized) " + "="*20)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("="*20 + " MODEL LOADED " + "="*20)

    # --- Run the Optimization ---
    print(f"\nRunning Bayesian Optimization for {args.n_calls} iterations...")
    result = gp_minimize(
        func=objective_function,
        dimensions=space,
        n_calls=args.n_calls,
        random_state=42,
        verbose=True
    )

    # --- Final Report ---
    print("\n" + "="*20 + " OPTIMIZATION COMPLETE " + "="*20)
    
    print("\n--- Best Parameters Found ---")
    best_params = {dim.name: val for dim, val in zip(space, result.x)}
    print(f"  - Temperature: {best_params['temperature']:.4f}")
    print(f"  - Top_p: {best_params['top_p']:.4f}")
    print(f"  - Repetition Penalty: {best_params['repetition_penalty']:.4f}")
    
    print(f"\nBest score achieved: {-result.fun:.4f}")

    print("\n" + "="*20 + " Chronological Run History " + "="*20)
    for params, score in zip(result.x_iters, result.func_vals):
        print(f"Score: {-score:+.4f} | Params: temp={params[0]:.3f}, top_p={params[1]:.3f}, rep_pen={params[2]:.3f}")

if __name__ == "__main__":
    main()
