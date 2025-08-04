# main.py (Final, complete version with all functions and looping)

import torch
from transformers import (
    AutoTokenizer, GenerationConfig,
    AutoModelForCausalLM
)
import urllib3
import yaml
import json
import logging
import time
import os
import sys
from typing import List, Dict, Union, Optional, Any
import argparse
import pathlib
import requests

from validator import ResponseValidator

# --- Global variables ---
model = None
tokenizer = None

class Config:
    filename: str = ""
    options: Optional[Dict[str, Any]] = None
    log_level: int = logging.DEBUG
    standalone: bool = False

def verify_env():
    """Verifies production environment (root, CUDA, ENV_VARS)."""
    if os.geteuid() != 0:
        raise RuntimeError("The LLM worker must be run as root.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not passed into container!!!")
    control_ip = os.getenv("CONTROL_IP")
    if not control_ip:
        raise RuntimeError("CONTROL_IP is not set in environment")
    control_token = os.getenv("CONTROL_TOKEN")
    if not control_token:
        raise RuntimeError("CONTROL_TOKEN is not set in environment")
    control_port = os.getenv("CONTROL_PORT")
    if not control_port:
        raise RuntimeError("CONTROL_PORT is not set in environment")
    try:
        control_port = int(control_port)
    except ValueError:
        raise RuntimeError("control port is not an integer")
    return control_ip, control_port, control_token

def configure() -> None:
    """Parses command line arguments and loads the YAML config."""
    parser = argparse.ArgumentParser(
        description="LLM Worker for RF component configuration")
    parser.add_argument(
        "--config", type=pathlib.Path, required=True,
        help="Path of YAML config for the llm worker")
    parser.add_argument("--log-level",
                        default="INFO",
                        help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--standalone",
                        action="store_true",
                        help="Run in standalone mode for local testing, bypassing environment checks and controller calls.")
    args = parser.parse_args()

    Config.standalone = args.standalone
    Config.log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    if not isinstance(Config.log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=Config.log_level,
                        format='%(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    Config.filename = args.config
    with open(str(args.config), 'r') as file:
        Config.options = yaml.safe_load(file)

def list_processes(control_url, auth_header):
    current_endpoint = "/list"
    headers = {"Authorization": auth_header, "Accept": "application/json", "User-Agent": "llm_worker/1.0"}
    try:
        response = requests.get(url=f"{control_url}{current_endpoint}", headers=headers, verify=False)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except requests.exceptions.RequestException as e:
        return False, {"error":str(e)}

def start_process(control_url, auth_header, json_payload):
    current_endpoint = "/start"
    headers = {"Authorization": auth_header, "Accept": "application/json", "User-Agent": "llm_worker/1.0", "Content-Type": "application/json"}
    try:
        response = requests.post(url=f"{control_url}{current_endpoint}", headers=headers, json=json_payload, verify=False)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except requests.exceptions.RequestException as e:
        return False, {"error":str(e)}

def stop_process(control_url, auth_header, process_id):
    current_endpoint = "/stop"
    headers = {"Authorization": auth_header, "Accept": "application/json", "User-Agent": "llm_worker/1.0", "Content-Type": "application/json"}
    json_payload = {"id": process_id}
    try:
        response = requests.post(url=f"{control_url}{current_endpoint}", headers=headers, json=json_payload, verify=False)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except requests.exceptions.RequestException as e:
        return False, {"error":str(e)}

def get_process_logs(control_url, auth_header, json_payload):
    current_endpoint = "/logs"
    headers = {"Authorization": auth_header, "Accept": "application/json", "User-Agent": "llm_worker/1.0", "Content-Type": "application/json"}
    try:
        response = requests.post(url=f"{control_url}{current_endpoint}", headers=headers, json=json_payload, verify=False)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except requests.exceptions.RequestException as e:
        return False, {"error":str(e)}

def generate_response(model, tokenizer, prompt_content: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [{"role": "user", "content": prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    generation_config = GenerationConfig(max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad():
        output_tokens = model.generate(**inputs, generation_config=generation_config)
    input_length = inputs['input_ids'].shape[1]
    newly_generated_tokens = output_tokens[0, input_length:]
    return tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()

def get_intent() -> Optional[List[str]]:
    """Identifies the list of components to be configured using the intent_prompt."""
    user_prompt = Config.options.get("user_prompt", "")
    intent_prompt = Config.options.get("intent_prompt", "")

    if not user_prompt or not intent_prompt:
        logging.error("'user_prompt' or 'intent_prompt' not found in config file.")
        return None

    max_attempts = 5
    attempt_count = 1
    current_prompt_content = intent_prompt.replace("{{ user_prompt }}", user_prompt)

    while attempt_count <= max_attempts:
        logging.info(f"--- Intent extraction attempt {attempt_count}/{max_attempts} ---")
        raw_response = generate_response(model, tokenizer, current_prompt_content)
        logging.info(f"Intent model output:\n---\n{raw_response}\n---")

        validator = ResponseValidator(raw_response, config_type="intent")
        validated_data = validator.validate()

        if validated_data and not validator.get_errors():
            logging.info(f"Intent validation successful. Components: {validated_data}")
            return validated_data

        error_details = "\n".join(validator.get_errors())
        logging.warning(f"Intent validation failed on attempt {attempt_count}. Errors:\n{error_details}")
        attempt_count += 1
        
        if attempt_count > max_attempts: break
        
        correction_prompt = (f"The previous JSON you provided was invalid for the following reasons:\n{error_details}\n\n"
                           f"Please regenerate the entire, corrected JSON object based on the original request.\n"
                           f"--- ORIGINAL REQUEST ---\n{user_prompt}")
        current_prompt_content = correction_prompt

    logging.error("Max attempts reached. Intent extraction failed.")
    return None

def response_validation_loop(current_response_text: str, config_type: str, original_prompt_content: str) -> Optional[Dict]:
    """Validates and self-corrects the generated JSON configuration."""
    if config_type not in ['sniffer', 'jammer', 'rtue']:
        logging.warning(f"Skipping validation loop: No rules for config type '{config_type}'.")
        return None

    logging.info(f"Config type is '{config_type}'. Starting validation and self-correction loop.")
    max_attempts = 25
    attempt_count = 1

    while attempt_count <= max_attempts:
        logging.info("="*40 + f" VALIDATION ATTEMPT {attempt_count} of {max_attempts} " + "="*40)
        validator = ResponseValidator(current_response_text, config_type=config_type)
        validated_data = validator.validate()

        if validated_data and not validator.get_errors():
            logging.info("Validation successful! Extracting final components.")
            return validated_data

        logging.warning("Validation failed. Preparing to self-correct.")
        attempt_count += 1
        if attempt_count > max_attempts:
            logging.error("Maximum correction attempts reached."); break
        error_details = "\n".join([f"- {e}" for e in validator.get_errors()])
        logging.warning(f"Validation Errors:\n{error_details}")

        correction_prompt_content = (
            f"The previous JSON configuration you provided was invalid for the following reasons:\n"
            f"{error_details}\n\n"
            f"Please regenerate the entire, corrected JSON object based on the original request.\n"
            f"--- ORIGINAL REQUEST ---\n{original_prompt_content}"
        )
        logging.info("Generating corrected response...")
        current_response_text = generate_response(model, tokenizer, correction_prompt_content)
        logging.info("="*20 + f" CORRECTED OUTPUT (ATTEMPT {attempt_count}) " + "="*20)
        logging.info(f"'{current_response_text}'")
        logging.info("="*20 + " END OF CORRECTED OUTPUT " + "="*20)
    
    return None

if __name__ == '__main__':
    configure()

    # --- MODEL LOADING (COMMON TO BOTH MODES) ---
    model_str = Config.options.get("model", None)
    if not model_str:
        logging.error("Model not specified in config file")
        sys.exit(1)
    logging.info(f"Loading model: {model_str}...")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    if device_map == "cpu" and not Config.standalone:
        logging.warning("Production mode expects CUDA, but it was not found.")
    elif device_map == "cpu" and Config.standalone:
        logging.info("Running on CPU in standalone mode.")
        
    model = AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.bfloat16, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    logging.info("Model loaded successfully.")

    # --- EXECUTION LOGIC BASED ON MODE ---

    if Config.standalone:
        # ======================================================================
        # --- STANDALONE TEST MODE ---
        # ======================================================================
        logging.info("\n" + "="*20 + " RUNNING IN STANDALONE TEST MODE " + "="*20)
        
        intent_list = get_intent()

        if not intent_list:
            logging.error("Could not determine any valid components from user prompt. Exiting.")
            sys.exit(1)

        logging.info(f"\nIntent processing complete. Will generate configs for: {intent_list}\n")

        for config_type in intent_list:
            logging.info("="*80)
            logging.info(f"PROCESSING COMPONENT: {config_type.upper()}")
            logging.info("="*80)

            component_prompt_key = f"{config_type}"
            system_prompt = Config.options.get(component_prompt_key, "")
            user_prompt = Config.options.get("user_prompt", "")
            
            if not system_prompt:
                logging.error(f"Could not find prompt key '{component_prompt_key}' in the config file. Skipping.")
                continue

            original_prompt_content = system_prompt + user_prompt
            
            logging.info("Generating initial configuration...")
            current_response_text = generate_response(model, tokenizer, original_prompt_content)
            
            logging.info("\n" + "="*20 + f" INITIAL CONFIG FOR {config_type.upper()} " + "="*20)
            logging.info(f"'{current_response_text}'")
            logging.info("="*20 + " END OF INITIAL CONFIG " + "="*20 + "\n")
        
        logging.info("\nStandalone test finished processing all components.")

    else:
        # ======================================================================
        # --- PRODUCTION MODE ---
        # ======================================================================
        logging.info("\n" + "="*20 + " RUNNING IN PRODUCTION MODE " + "="*20)
        control_ip, control_port, control_token = verify_env()
        control_url = f"https://{control_ip}:{control_port}"
        auth_header = f"Bearer {control_token}"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        intent_list = get_intent()
        if not intent_list:
            logging.error("Could not determine any valid components from user prompt. Exiting.")
            sys.exit(1)

        logging.info(f"\nIntent processing complete. Will configure and start: {intent_list}\n")

        for config_type in intent_list:
            logging.info("="*80)
            logging.info(f"PROCESSING COMPONENT: {config_type.upper()}")
            logging.info("="*80)

            component_prompt_key = f"{config_type}_prompt"
            system_prompt = Config.options.get(component_prompt_key, "")
            user_prompt = Config.options.get("user_prompt", "")
            original_prompt_content = system_prompt + user_prompt
            
            logging.info("="*20 + " EXECUTING PROMPT " + "="*20)
            current_response_text = generate_response(model, tokenizer, original_prompt_content)
            logging.info("="*20 + " MODEL GENERATED OUTPUT " + "="*20)
            logging.info(f"'{current_response_text}'")
            logging.info("="*20 + " END OF MODEL OUTPUT " + "="*20)

            validated_data = response_validation_loop(current_response_text, config_type, original_prompt_content)

            if validated_data and validated_data.get('config_str'):
                logging.info("="*20 + " FINAL VALIDATED CONFIGURATION " + "="*20)
                
                controller_retry_max_attempts = 10
                controller_attempt_count = 1
                
                while controller_attempt_count <= controller_retry_max_attempts:
                    final_config_type = validated_data.get('type')
                    final_config_id = validated_data.get('id')
                    final_config_string = validated_data.get('config_str')
                    
                    json_payload = {"id": final_config_id, "type": final_config_type, "config_str": final_config_string}
                    json_payload["rf"] = {"type":"b200","images_dir":"/usr/share/uhd/images"}
                    
                    logging.info(f"Attempting to start process with controller (Attempt {controller_attempt_count}/{controller_retry_max_attempts})...")
                    success, response_data = start_process(control_url, auth_header, json_payload)

                    if success:
                        logging.info(f"Successfully started component '{config_type}'.")
                        logging.info(f"Controller response: {response_data}")
                        break 
                    
                    logging.error(f"Failed to start component '{config_type}' via controller.")
                    controller_error_details = response_data.get("error", "No error details from controller.")
                    logging.error(f"Controller error: {controller_error_details}")
                    
                    controller_attempt_count += 1
                    if controller_attempt_count > controller_retry_max_attempts:
                        logging.critical("Maximum controller retry attempts reached. Aborting this component.")
                        break

                    logging.warning("Attempting to generate a new configuration based on controller feedback.")
                    controller_correction_prompt = (
                        f"The configuration you provided was syntactically valid, but the system controller REJECTED it for the following reason:\n"
                        f"{controller_error_details}\n\n"
                        f"Please analyze this feedback and regenerate the entire, corrected JSON object based on the original request.\n"
                        f"--- ORIGINAL REQUEST ---\n{original_prompt_content}"
                    )
                    
                    current_response_text = generate_response(model, tokenizer, controller_correction_prompt)
                    
                    logging.info("="*20 + " RE-VALIDATING CONTROLLER CORRECTION " + "="*20)
                    validated_data = response_validation_loop(current_response_text, config_type, original_prompt_content)
                    
                    if not validated_data:
                        logging.error("The LLM produced a syntactically invalid configuration while trying to correct a controller error. Aborting this component.")
                        break
            else:
                logging.error(f"Could not obtain a valid and non-empty '{config_type}' configuration after all attempts. Skipping this component.")
                continue

        logging.info("\nProduction run finished processing all components.")