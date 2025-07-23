# main.py
# MODIFIED: Extracts components upon success and uses them in the final output.

import torch
from transformers import (
    AutoTokenizer, GenerationConfig,
    AutoModelForCausalLM
)
import yaml
import logging
import time
import os
import sys
from typing import Dict, Optional, Any
import argparse
import pathlib
import json

# This imports your real validator from the adjacent file.
from validator import ResponseValidator

class Config:
    filename: str = ""
    options: Optional[Dict[str, Any]] = None
    log_level: int = logging.DEBUG


def configure() -> None:
    # This function remains  .
    parser = argparse.ArgumentParser(
        description="LLM Worker for RF Configuration Generation")
    parser.add_argument(
        "--config", type=pathlib.Path, required=True,
        help="Path of YAML config for the llm worker")
    parser.add_argument("--log-level",
                        default="DEBUG",
                        help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    args = parser.parse_args()
    Config.log_level = getattr(logging, args.log_level.upper(), logging.DEBUG)
    if not isinstance(Config.log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=Config.log_level,
                        format='%(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    Config.filename = args.config
    with open(str(args.config), 'r') as file:
        Config.options = yaml.safe_load(file)


def generate_response(model, tokenizer, prompt_content: str) -> str:
    # This function remains  .
    messages = [{"role": "user", "content": prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logging.debug("="*20 + " PROMPT BEING SENT TO MODEL " + "="*20)
    logging.debug(formatted_prompt)
    logging.debug("="*20 + " END OF PROMPT " + "="*20)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    output_tokens = model.generate(**inputs, generation_config=generation_config)
    input_length = inputs['input_ids'].shape[1]
    newly_generated_tokens = output_tokens[0, input_length:]
    clean_response = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
    return clean_response

    # TODO: use llm for intent determination



if __name__ == '__main__':
    configure()


    # --- Model and Tokenizer Setup    ---
    model_str = Config.options.get("model")
    if not model_str:
        logging.error("Model not specified in config")
        sys.exit(1)
    logging.info("="*20 + " LOADING BASE MODEL " + "="*20)
    logging.info(f"Using model: {model_str}")
    model = AutoModelForCausalLM.from_pretrained(
        model_str, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    if tokenizer.pad_token is None:
        logging.warning("Tokenizer has no pad_token, setting it to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    logging.info("="*20 + " MODEL LOADED " + "="*20)

    user_request_text, planner_prompt = Config.options.get("user_prompt", ""), Config.options.get("planner_prompt", "")
    logging.info(f"User prompt: {user_request_text}")
    logging.info(f"Planner prompt: {planner_prompt}")
    if not user_request_text:
        logging.error("User prompt not specified in config")
        sys.exit(1)

    if not planner_prompt:
        logging.error("Planner prompt not specified in config")
        sys.exit(1)
    user_plan_raw = generate_response(model, tokenizer, planner_prompt + user_request_text)


    logging.info(f"The output of planner: {user_plan_raw}")
    logging.info("+++++++++++++++++++++++++++++++++++++++")

    validator = ResponseValidator(user_plan_raw, config_type="plan")
    validated_plan = validator.validate()

    if not validated_plan or "steps" not in validated_plan:
        logging.error("Planner output could not be validated as a plan.")

    planner_steps = validated_plan["steps"]

    logging.info(f"Validated planner steps: {planner_steps}")

    for step in planner_steps:
        component = step.get("component")
        instruction = step.get("additional_instruction")
        logging.info(f"Executing step for component: {component} with instruction: {instruction}")  
        if not component or not instruction:
            logging.warning(f"Skipping invalid step: {step}")
            continue
    # =====================================================

    # --- Initial Prompt Construction    ---
    config_type, system_prompt = None, ""
    if "sniffer" in user_request_text.lower():
        config_type, system_prompt = "sniffer", Config.options.get("sniffer_prompt", "")
    elif "jam" in user_request_text.lower():
        config_type, system_prompt = "jammer", Config.options.get("jammer_prompt", "")
    elif "rtue" in user_request_text.lower():
        config_type, system_prompt = "rtue", Config.options.get("rtue_prompt", "")
    else:
        logging.warning("Could not determine config type from user prompt. Using raw prompt.")
        system_prompt, user_request_text = user_request_text, ""
    original_prompt_content = system_prompt + user_request_text

    # --- Initial Generation    ---
    logging.info("="*20 + " EXECUTING PROMPT " + "="*20)
    current_response_text = generate_response(model, tokenizer, original_prompt_content)
    logging.info("="*20 + " MODEL GENERATED OUTPUT " + "="*20)
    logging.info(f"'{current_response_text}'")
    logging.info("="*20 + " END OF MODEL OUTPUT " + "="*20)


    # --- Validation Loop ---
    if config_type in ['sniffer', 'jammer', 'rtue']: # Added rtue here for consistency
        logging.info(f"Config type is '{config_type}'. Starting validation and self-correction loop.")
        
        # Initialize variables to hold the final extracted components
        validated_data = None
        final_config_type = None
        final_config_id = None
        final_config_string = None
        
        max_attempts = 25
        attempt_count = 1

        while attempt_count <= max_attempts:
            logging.info("="*40)
            logging.info(f"VALIDATION ATTEMPT {attempt_count} of {max_attempts}")
            logging.info("="*40)

            validator = ResponseValidator(current_response_text, config_type=config_type)
            validated_data = validator.validate()

            # This is the SUCCESS path
            if validated_data:
                logging.info("Validation successful! Extracting final components.")
                
                # Extract the components into our variables
                final_config_type = validated_data.get('type')
                final_config_id = validated_data.get('id')
                final_config_string = validated_data.get('config')
                
                break # Exit the loop on success


            # --- This is the FAILURE path  ---
            logging.warning("Validation failed. Preparing to self-correct.")
            attempt_count += 1
            
            if attempt_count > max_attempts:
                logging.error("Maximum correction attempts reached.")
                break

            error_details = "\n".join([f"- {e}" for e in validator.get_errors()])
            logging.warning(f"Validation Errors:\n{error_details}")

            correction_prompt_content = (
                f"The previous JSON configuration you provided was invalid for the following reasons:\n"
                f"{error_details}\n\n"
                f"Please regenerate the entire, corrected JSON object based on the original request. "
                f"Remember to adhere to all rules.\n\n"
                f"--- ORIGINAL REQUEST ---\n{original_prompt_content}"
            )
            
            logging.info("Generating corrected response...")
            current_response_text = generate_response(model, tokenizer, correction_prompt_content)
            logging.info("="*20 + f"CORRECTED OUTPUT (ATTEMPT {attempt_count}) " + "="*20)
            logging.info(f"'{current_response_text}'")
            logging.info("="*20 + " END OF CORRECTED OUTPUT " + "="*20)

        # --- Final Outcome for Validated Types ---
        if final_config_string:
            logging.info("="*20 + " FINAL VALIDATED CONFIGURATION " + "="*20)
            logging.info(f"Successfully generated configuration for type: '{final_config_type}' with ID: '{final_config_id}'")
            logging.info("--- CONFIG CONTENT ---")
            logging.info(f"\n{final_config_string}")
            logging.info("----------------------")
            logging.info("Script finished successfully.")
        else:
            logging.error("="*20 + " SCRIPT FAILED " + "="*20)
            logging.error(f"Could not obtain a valid '{config_type}' configuration after all attempts.")
            sys.exit(1)

    else:
        # --- Handling for types that do not have validation    ---
        logging.warning(f"Skipping validation loop: No validation rules defined for config type '{config_type}'.")
        logging.info("="*20 + " FINAL UNVALIDATED OUTPUT " + "="*20)
        logging.info(current_response_text)
        logging.info("Script finished.")