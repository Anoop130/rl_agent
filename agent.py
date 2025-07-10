# auto_tuner_agent.py (Reverted to the proven parser, keeping other improvements)

import torch
import yaml
import re
import random
import argparse
from tqdm import tqdm
from itertools import product

# Imports for the simplified, no-fine-tuning approach
from transformers import (
    AutoTokenizer, GenerationConfig, BitsAndBytesConfig,
    AutoModelForCausalLM
)
from simulation_env import mock_run_simulation_and_get_reward

# ===================================================================
# SECTION 1: AUTO-TUNER AGENT and HELPER FUNCTION
# ===================================================================
class HyperparameterAgent:
    def __init__(self, action_space: dict):
        self.action_space = action_space
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.epsilon_decay = 0.95
        self.min_epsilon = 0.1

    def get_action(self) -> dict:
        actions = self._get_all_actions()
        if random.random() < self.epsilon:
            print("[Auto-Tuner] ACTION: Exploring with random hyperparameters.")
            return random.choice(actions)
        else:
            print("[Auto-Tuner] ACTION: Exploiting with best-known hyperparameters.")
            if not self.q_table: return random.choice(actions)
            best_action_tuple = max(self.q_table, key=self.q_table.get)
            return dict(best_action_tuple)

    def learn(self, action: dict, reward: float):
        action_tuple = tuple(sorted(action.items()))
        old_q = self.q_table.get(action_tuple, 0.0)
        new_q = old_q + self.learning_rate * (reward - old_q)
        self.q_table[action_tuple] = new_q
        print(f"[Auto-Tuner] LEARNING: Q-value for {dict(action_tuple)} updated to {new_q:.4f}")
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        print(f"[Auto-Tuner] STATUS: New exploration rate (epsilon) is {self.epsilon:.3f}")

    def _get_all_actions(self) -> list[dict]:
        keys, values = self.action_space.keys(), self.action_space.values()
        return [dict(zip(keys, instance)) for instance in product(*values)]

# --- REVERTED: Using the simpler parser that was proven to work ---
def _parse_llm_output(text: str) -> dict:
    """
    Parses the full text from the LLM to find and decode the YAML block.
    This is the simpler version that worked previously.
    """
    try:
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
    except (yaml.YAMLError, IndexError):
        pass
    return None

# ===================================================================
# SECTION 2: THE GENERATION RUNNER
# ===================================================================
def execute_generation_run(hparams: dict, model, tokenizer) -> float:
    print(f"\n--- [Worker] Starting run with Generation HPs: {hparams} ---")
    
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
    total_score = 0
    test_frequencies = [1.83, 1.842, 1.85] 
    
    generation_config = GenerationConfig(
        max_new_tokens=250,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        **hparams
    )

    for freq in test_frequencies:
        prompt_text = PROMPT_TEMPLATE.format(freq=freq)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        output_tokens = model.generate(**inputs, generation_config=generation_config)

        input_token_length = inputs.input_ids.shape[1]
        generated_token_ids = output_tokens[0, input_token_length:]
        generated_text_only = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        print("="*40)
        print(f"DEBUGGING: Generated-only output for frequency {freq} GHz:")
        print(generated_text_only)
        print("="*40)
        
        full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        config = _parse_llm_output(full_text)
        score = mock_run_simulation_and_get_reward(config)
        total_score += score

    avg_score = total_score / len(test_frequencies)
    print(f"--- [Worker] Run Complete. Final average score: {avg_score:.4f} ---")
    return avg_score

# ===================================================================
# SECTION 3: THE MAIN CONTROLLER
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Auto-Tuner for LLM Generation.")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-Coder-6.7B-Instruct", help="Base model to use.")
    parser.add_argument("--loops", type=int, default=50, help="Number of hyperparameter sets to test.")
    args = parser.parse_args()

    # Refined hyperparameter space based on results
    hyperparameter_space = {
        'temperature': [0.6, 0.7, 0.8],
        'top_p': [0.9, 0.95],
        'repetition_penalty': [1.0, 1.1],
    }
    
    auto_tuner = HyperparameterAgent(hyperparameter_space)
    
    run_history = []

    print("="*20 + " LOADING BASE MODEL (ONCE) " + "="*20)
    print(f"Using model: {args.model}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("="*20 + " MODEL LOADED " + "="*20)

    print(f"Running for {args.loops} loops.")

    for i in range(args.loops):
        print(f"\n{'='*15} Auto-Tuner Episode {i+1}/{args.loops} {'='*15}")
        chosen_hps = auto_tuner.get_action()
        reward = execute_generation_run(
            hparams=chosen_hps, 
            model=model,
            tokenizer=tokenizer
        )
        auto_tuner.learn(action=chosen_hps, reward=reward)
        run_history.append({'reward': reward, 'hps': chosen_hps})
    
    print("\n" + "="*20 + " AUTO-TUNING COMPLETE " + "="*20)
    if not auto_tuner.q_table:
        print("No trials were completed.")
        return
        
    sorted_q_table = sorted(auto_tuner.q_table.items(), key=lambda item: item[1], reverse=True)
    print("\nFinal discovered knowledge (Q-Table), from best to worst:")
    for (hps_tuple, score) in sorted_q_table:
        print(f"  Learned Score: {score:.4f} | Hyperparameters: {dict(hps_tuple)}")
    
    best_hps = dict(sorted_q_table[0][0])
    print(f"\nRECOMMENDED GENERATION HYPERPARAMETERS: {best_hps}")

    print("\n" + "="*20 + " HIGH-PERFORMING INDIVIDUAL RUNS (Reward > 0.5) " + "="*20)
    high_performers = [run for run in run_history if run['reward'] > 0.5]
    
    if not high_performers:
        print("No individual runs achieved an average reward > 0.5.")
    else:
        high_performers.sort(key=lambda x: x['reward'], reverse=True)
        for run in high_performers:
             print(f"  Run Reward: {run['reward']:.4f} | Hyperparameters: {run['hps']}")

if __name__ == "__main__":
    main()