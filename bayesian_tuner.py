# bayesian_tuner.py
#
# Implements a class-based Bayesian optimizer to tune generation
# hyperparameters for a causal language model.

import argparse
import torch
import yaml
from transformers import (
    AutoTokenizer, GenerationConfig,
    AutoModelForCausalLM
)
from simulation_env import mock_run_simulation_and_get_reward

# Imports for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


def _parse_llm_output(text: str) -> dict:
    """
    Parses a YAML configuration block from the LLM's output text.

    Args:
        text: The full string output from the language model.

    Returns:
        A dictionary containing the parsed configuration, or None on failure.
    """
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
    Executes a batched generation run and evaluates the average score.

    This function serves as the expensive black-box objective for optimization.

    Args:
        hparams: A dictionary of generation hyperparameters (e.g., temp, top_p).
        model: The pre-loaded causal language model.
        tokenizer: The pre-loaded tokenizer.

    Returns:
        The average reward score from the simulation environment.
    """
    PROMPT_TEMPLATE = """You are a highly skilled RF engineer specializing in electronic countermeasures.
Your mission is to generate a complete YAML configuration file to effectively jam a target frequency.
### Instructions:
1.  Analyze the `High-Level Goal`.
2.  Determine the optimal values for **all** required configuration parameters.
3.  The output MUST be a single, valid YAML block containing all necessary keys.
4.  Use snake_case for all keys (e.g., `center_frequency`).
5.  Use scientific 'e' notation for frequencies and bandwidth.
### Current Task:
High-Level Goal: Jam a target at {freq:.4f} GHz
### YAML Output:
"""
    test_frequencies = [1.83, 1.842, 1.85, 1.865, 1.88, 1.90]

    generation_config = GenerationConfig(
        max_new_tokens=250,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        **hparams
    )

    prompts = [PROMPT_TEMPLATE.format(freq=f) for f in test_frequencies]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    output_tokens = model.generate(**inputs, generation_config=generation_config)
    full_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    total_score = 0
    for text in full_texts:
        config = _parse_llm_output(text)
        score = mock_run_simulation_and_get_reward(config)
        total_score += score

    return total_score / len(test_frequencies)


class LLMOptimizer:
    """
    Manages the Bayesian Optimization process for LLM hyperparameters.

    This class encapsulates the model, tokenizer, search space, and the
    optimization objective function to provide a clean interface for tuning.
    """
    space = [
        Real(0.5, 1.0, name='temperature'),
        Real(0.8, 1.0, name='top_p'),
        Real(1.0, 1.2, name='repetition_penalty')
    ]

    def __init__(self, model_name: str):
        """
        Initializes the optimizer and loads the specified model and tokenizer.

        Args:
            model_name: The name of the model from the Hugging Face Hub.
        """
        print(f"Loading model: {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model loaded successfully.")

    @use_named_args(space)
    def objective_function(self, **params):
        """
        The objective function to be minimized by scikit-optimize.

        It runs the generation pipeline with a given set of hyperparameters
        and returns the negative of the achieved score.
        """
        score = execute_generation_run(
            hparams=params, model=self.model, tokenizer=self.tokenizer
        )
        param_str = ", ".join([f"{k}={v:.3f}" for k, v in params.items()])
        print(f"Parameters: {param_str} -> Score: {score:.4f}")
        # The optimizer minimizes, so we return the negative of our score.
        return -score

    def run(self, n_calls: int):
        """
        Starts the Bayesian Optimization process.

        Args:
            n_calls: The total number of iterations to run.

        Returns:
            The optimization result object from `skopt.gp_minimize`.
        """
        print(f"\nStarting Bayesian Optimization for {n_calls} iterations...")
        return gp_minimize(
            func=self.objective_function,
            dimensions=self.space,
            n_calls=n_calls,
            random_state=42,
            verbose=False # Custom logging is handled in objective_function
        )

def main():
    """
    Main entry point for the script. Parses arguments, runs the
    optimizer, and reports the results.
    """
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for LLM Generation Hyperparameters."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-Coder-6.7B-Instruct",
        help="Hugging Face model to use for generation."
    )
    parser.add_argument(
        "--n_calls",
        type=int,
        default=25,
        help="Number of optimization iterations."
    )
    args = parser.parse_args()

    optimizer = LLMOptimizer(model_name=args.model)
    result = optimizer.run(n_calls=args.n_calls)

    print("\n" + "="*20 + " OPTIMIZATION COMPLETE " + "="*20)
    
    print("\nBest Parameters Found:")
    best_params = {dim.name: val for dim, val in zip(optimizer.space, result.x)}
    for name, val in best_params.items():
        print(f"  - {name.replace('_', ' ').capitalize()}: {val:.4f}")
    
    print(f"\nBest score achieved: {-result.fun:.4f}")

    print("\n" + "="*20 + " Chronological Run History " + "="*20)
    for i in range(len(result.x_iters)):
        params = {dim.name: val for dim, val in zip(optimizer.space, result.x_iters[i])}
        score = result.func_vals[i]
        param_str = ", ".join([f"{k}={v:.3f}" for k, v in params.items()])
        print(f"  Run {i+1:02d}: Score={-score:+.4f} | Params: {param_str}")


if __name__ == "__main__":
    main()