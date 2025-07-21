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
from typing import List, Dict, Union, Optional, Any
import argparse
import pathlib

import requests

from validator import ResponseValidator


class Config:
    filename : str = ""
    
    options : Optional[Dict[str,Any]] = None
    log_level : int = logging.DEBUG

def configure() -> None:
    """
    Reads in CLI arguments
    Parses YAML config
    Configures logging
    """
    parser = argparse.ArgumentParser(
        description="RAN tester UE process controller")
    parser.add_argument(
        "--config", type=pathlib.Path, required=True,
        help="Path of YAML config for the llm worker")
    parser.add_argument("--log-level",
                    default="DEBUG",
                    help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    args = parser.parse_args()
    Config.log_level = getattr(logging, args.log_level.upper(), 1)

    if not isinstance(Config.log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")

    logging.basicConfig(level=Config.log_level,
                    format='%(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    Config.filename = args.config
    with open(str(args.config), 'r') as file:
        Config.options = yaml.safe_load(file)



if __name__ == '__main__':

    
    configure()

    

    model_str = Config.options.get("model", None)
    if not model_str:
        logging.error("Model not specified")
        sys.exit(1)

    logging.info("="*20 + " LOADING BASE MODEL (HIGH PRECISION) " + "="*20)
    logging.info(f"Using model: {model_str}")

    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_str)

    logging.info("="*20 + " MODEL LOADED " + "="*20)

    user_prompt = Config.options.get("user_prompt", None)

    
    if not user_prompt:
        raise RuntimeError(f"user prompt not provided in {Config.filename}")

    generation_config = GenerationConfig(
        max_new_tokens=250,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )


    logging.debug("="*20 + " EXECUTING PROMPT " + "="*20)


    if "jam" in user_prompt.lower():
        jammer_prompt = Config.options.get("jammer_prompt", None)
        prompt = user_prompt + jammer_prompt

    if "sniffer" in user_prompt.lower():
        sniffer_prompt = Config.options.get("sniffer_prompt", None)
        prompt = user_prompt + sniffer_prompt  

    if hasattr(model, 'past_key_values'):
        model.past_key_values = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    output_tokens = model.generate(**inputs, generation_config=generation_config)

    input_length = inputs['input_ids'].shape[1]
    newly_generated_tokens = output_tokens[0, input_length:]
    clean_response = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True)

    # Log the clean response explicitly
    logging.info("="*20 + " MODEL GENERATED OUTPUT " + "="*20)
    logging.info(clean_response)
    logging.info("="*20 + " END OF MODEL OUTPUT " + "="*20)

    # Use the clean response for validation
    validator = ResponseValidator(clean_response)


    while True:
        logging.info("Processing LLM response...")
        logging.info("Validating configuration...")

        validated_data = validator.validate()
        request_json = validated_data

        if validated_data:
            logging.info(f"Validated data: {validated_data}")
        else:
            logging.warning("No valid JSON could be extracted from the response.")


        error_list = validator.get_errors()
        if error_list:
            error_details = "\n".join([f"- {e}" for e in error_list])
            logging.warning(f"Error list:\n{error_details}")
            prompt = f"Previous configuration was invalid: {error_details}. Please provide a corrected RF system configuration for: {jammer_prompt + user_prompt}"

            logging.info("Generating next response...")
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            output_tokens = model.generate(**inputs, generation_config=generation_config)
            response_str = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

            input_length = inputs['input_ids'].shape[1]
            newly_generated_tokens = output_tokens[0, input_length:]
            clean_response = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True)

            #log output
            logging.info("="*20 + " MODEL GENERATED OUTPUT " + "="*20)
            logging.info(clean_response)
            logging.info("="*20 + " END OF MODEL OUTPUT " + "="*20)
            # Use the clean response for validation
            validator = ResponseValidator(clean_response)
                
        
        time.sleep(1)
    