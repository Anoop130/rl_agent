# main.py — Merged: Old stable functions + RAG retrieval flow

import torch
from transformers import (
    AutoTokenizer, GenerationConfig,
    AutoModelForCausalLM, BitsAndBytesConfig
)
import urllib3
import yaml
import json
import logging
import time
import os
import sys
from typing import List, Dict, Optional, Any
import argparse
import pathlib
import requests

# --- RAG imports ---
import chromadb
from chromadb.utils import embedding_functions
# sentence-transformers backend is pulled by chromadb's SentenceTransformerEmbeddingFunction

from validator import ResponseValidator

# --- Globals ---
model = None
tokenizer = None
retriever = None

class Config:
    filename: str = ""
    options: Optional[Dict[str, Any]] = None
    log_level: int = logging.DEBUG
    standalone: bool = False

# ---------------------------
# Old, proven environment & IO
# ---------------------------
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
    parser = argparse.ArgumentParser(description="Unified RAG-Powered System")
    parser.add_argument("--config", type=pathlib.Path, required=True,
                        help="Path of YAML config for the llm worker")
    parser.add_argument("--log-level", default="INFO",
                        help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--standalone", action="store_true",
                        help="Run in standalone mode for local testing, bypassing environment checks and controller calls.")
    args = parser.parse_args()

    Config.standalone = args.standalone
    Config.log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    if not isinstance(Config.log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")

    logging.basicConfig(
        level=Config.log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    Config.filename = str(args.config)
    with open(str(args.config), 'r', encoding="utf-8") as file:
        Config.options = yaml.safe_load(file) or {}

    if not isinstance(Config.options, dict):
        logging.error("Config YAML must deserialize to a dict.")
        sys.exit(1)

def list_processes(control_url, auth_header):
    current_endpoint = "/list"
    headers = {"Authorization": auth_header, "Accept": "application/json", "User-Agent": "llm_worker/1.0"}
    try:
        response = requests.get(url=f"{control_url}{current_endpoint}", headers=headers, verify=False)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def start_process(control_url, auth_header, json_payload):
    current_endpoint = "/start"
    headers = {"Authorization": auth_header, "Accept": "application/json", "User-Agent": "llm_worker/1.0", "Content-Type": "application/json"}
    try:
        response = requests.post(url=f"{control_url}{current_endpoint}", headers=headers, json=json_payload, verify=False)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

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
        return False, {"error": str(e)}

def get_process_logs(control_url, auth_header, json_payload):
    current_endpoint = "/logs"
    headers = {"Authorization": auth_header, "Accept": "application/json", "User-Agent": "llm_worker/1.0", "Content-Type": "application/json"}
    try:
        response = requests.post(url=f"{control_url}{current_endpoint}", headers=headers, json=json_payload, verify=False)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def save_config_to_file(config_str: str, config_type: str, config_id: str, output_dir: str = "/host/configs"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{config_type}_{config_id}.toml"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(config_str)
    logging.info(f"Config saved to: {filepath}")

# ---------------------------
# Old, proven LLM helpers
# ---------------------------
def generate_response(model, tokenizer, prompt_content: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [{"role": "user", "content": prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    generation_config = GenerationConfig(max_new_tokens=2048, do_sample=False, pad_token_id=tokenizer.eos_token_id)
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
    # keep the original {{ user_prompt }} placeholder behavior
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
        if attempt_count > max_attempts:
            break

        correction_prompt = (
            f"The previous JSON you provided was invalid for the following reasons:\n{error_details}\n\n"
            f"Please regenerate the entire, corrected JSON object based on the original request.\n"
            f"--- ORIGINAL REQUEST ---\n{user_prompt}"
        )
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
            logging.error("Maximum correction attempts reached.")
            break

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

# ---------------------------
# RAG Retriever (new)
# ---------------------------
class RAGRetriever:
    def __init__(self, db_dir="vector_db", collection_name="rf_knowledge", model_name="all-MiniLM-L6-v2"):
        logging.info("Initializing RAG Retriever...")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        db_client = chromadb.PersistentClient(path=db_dir)
        try:
            self.collection = db_client.get_collection(name=collection_name, embedding_function=sentence_transformer_ef)
            logging.info("RAG Retriever initialized successfully.")
        except Exception as e:
            logging.error(f"FATAL: Could not initialize ChromaDB collection '{collection_name}'. Error: {e}")
            logging.error("Please ensure you have run the 'build_vector_db.py' script first.")
            sys.exit(1)

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        logging.info(f"--- RAG: Retrieving context for query: '{query}' ---")
        results = self.collection.query(query_texts=[query], n_results=n_results)
        retrieved_docs = results['documents'][0]
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            source = results['metadatas'][0][i].get('source', 'unknown')
            logging.info(f"  [Retrieved Doc {i+1} from '{source}']: {doc[:120].strip().replace(chr(10),' ')}...")
            context_str += f"- From {source}:\n{doc}\n\n"
        return context_str.strip()

# ======================================================================
# MAIN
# ======================================================================
if __name__ == '__main__':
    configure()

    # --- Model Loading ---
    model_str = (Config.options or {}).get("model", None)
    if not model_str:
        logging.error("Model not specified in config file")
        sys.exit(1)

    logging.info(f"Loading LLM: {model_str}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    if device_map == "cpu" and not Config.standalone:
        logging.warning("Production mode expects CUDA, but it was not found.")
    elif device_map == "cpu" and Config.standalone:
        logging.info("Running on CPU in standalone mode.")

    model = AutoModelForCausalLM.from_pretrained(model_str, quantization_config=bnb_config, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    logging.info("LLM loaded successfully.")

    # --- RAG init ---
    retriever = RAGRetriever()

    # --- Intent ---
    intent_list = get_intent()
    if not intent_list:
        logging.error("Could not determine any valid components from user prompt. Exiting.")
        sys.exit(1)
    logging.info(f"\nIntent processing complete. Will generate configs for: {intent_list}\n")

    # --- Execution ---
    if Config.standalone:
        for config_type in intent_list:
            logging.info("="*80)
            logging.info(f"PROCESSING COMPONENT: {config_type.upper()}")
            logging.info("="*80)

            # Pull the prompt block for this component
            system_prompt_full = (Config.options or {}).get(config_type, "")
            user_request = (Config.options or {}).get("user_prompt", "")

            if not system_prompt_full or not user_request:
                logging.error(f"System prompt for '{config_type}' or 'user_prompt' not found in config. Skipping.")
                continue

            # RAG retrieval + augmented prompt
            retrieval_query = f"Rules and engineering constraints for a {config_type} config to fulfill the request: {user_request}"
            retrieved_context = retriever.retrieve_context(retrieval_query)

            # If your prompt blocks contain a "### USER REQUEST:" segment, only use instructions before that.
            system_instructions = system_prompt_full.split("### USER REQUEST:")[0]

            final_prompt_template = """
You are an expert RF systems assistant.
First, review the provided CONTEXT for critical engineering rules.
Then, use that context to follow the INSTRUCTIONS to generate a valid JSON configuration that fulfills the USER REQUEST.

--- CONTEXT (Rules & Formulas) ---
{context}
--- END OF CONTEXT ---

--- INSTRUCTIONS (Schema & Formatting) ---
{system_prompt_instructions}
--- END OF INSTRUCTIONS ---

--- USER REQUEST ---
{user_request}

Provide only the final JSON object.

--- JSON OUTPUT ---
"""
            final_augmented_prompt = final_prompt_template.format(
                context=retrieved_context,
                system_prompt_instructions=system_instructions,
                user_request=user_request
            )

            logging.info("\n--- FINAL AUGMENTED PROMPT FOR LLM ---\n" + final_augmented_prompt + "\n------------------------------------\n")
            logging.info("Generating initial configuration with RAG...")
            initial_response_text = generate_response(model, tokenizer, final_augmented_prompt)

            logging.info("\n" + "="*20 + f" RAG-POWERED INITIAL CONFIG FOR {config_type.upper()} " + "="*20)
            logging.info(f"'{initial_response_text}'")
            logging.info("="*20 + " END OF INITIAL CONFIG " + "="*20 + "\n")

            # Validate / self-correct (old loop)
            validated_data = response_validation_loop(initial_response_text, config_type, user_request)
            if not validated_data:
                logging.error(f"Could not obtain a valid config for '{config_type}' after all attempts. Skipping.")
                continue

            # Pretty print or save
            try:
                pretty_json = json.dumps(validated_data, indent=2)
                logging.info("="*20 + f" STANDALONE MODE: FINAL VALIDATED JSON FOR {config_type.upper()} " + "="*20)
                logging.info(pretty_json)
            except TypeError:
                logging.info(str(validated_data))

            # Optional local save (TOML string expected in 'config_str')
            if validated_data.get("config_str") and validated_data.get("id") and validated_data.get("type"):
                save_config_to_file(validated_data["config_str"], validated_data["type"], validated_data["id"])

        logging.info("\nStandalone test finished processing all components.")

    else:
        # Production: verify env and talk to controller
        logging.info("\n" + "="*20 + " RUNNING IN PRODUCTION MODE " + "="*20)
        control_ip, control_port, control_token = verify_env()
        control_url = f"https://{control_ip}:{control_port}"
        auth_header = f"Bearer {control_token}"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        for config_type in intent_list:
            logging.info("="*80)
            logging.info(f"PROCESSING COMPONENT: {config_type.upper()}")
            logging.info("="*80)

            system_prompt_full = (Config.options or {}).get(config_type, "")
            user_request = (Config.options or {}).get("user_prompt", "")

            if not system_prompt_full or not user_request:
                logging.error(f"System prompt for '{config_type}' or 'user_prompt' not found in config. Skipping.")
                continue

            retrieval_query = f"Rules and engineering constraints for a {config_type} config to fulfill the request: {user_request}"
            retrieved_context = retriever.retrieve_context(retrieval_query)
            system_instructions = system_prompt_full.split("### USER REQUEST:")[0]

            final_prompt_template = """
You are an expert RF systems assistant.
First, review the provided CONTEXT for critical engineering rules.
Then, use that context to follow the INSTRUCTIONS to generate a valid JSON configuration that fulfills the USER REQUEST.

--- CONTEXT (Rules & Formulas) ---
{context}
--- END OF CONTEXT ---

--- INSTRUCTIONS (Schema & Formatting) ---
{system_prompt_instructions}
--- END OF INSTRUCTIONS ---

--- USER REQUEST ---
{user_request}

Provide only the final JSON object.

--- JSON OUTPUT ---
"""
            final_augmented_prompt = final_prompt_template.format(
                context=retrieved_context,
                system_prompt_instructions=system_instructions,
                user_request=user_request
            )

            logging.info("\n--- FINAL AUGMENTED PROMPT FOR LLM ---\n" + final_augmented_prompt + "\n------------------------------------\n")
            logging.info("Generating initial configuration with RAG...")
            initial_response_text = generate_response(model, tokenizer, final_augmented_prompt)

            logging.info("="*20 + " MODEL GENERATED OUTPUT " + "="*20)
            logging.info(f"'{initial_response_text}'")
            logging.info("="*20 + " END OF MODEL OUTPUT " + "="*20)

            validated_data = response_validation_loop(initial_response_text, config_type, user_request)
            if validated_data and validated_data.get('config_str'):
                logging.info("="*20 + " FINAL VALIDATED CONFIGURATION " + "="*20)

                controller_retry_max_attempts = 10
                controller_attempt_count = 1

                while controller_attempt_count <= controller_retry_max_attempts:
                    final_config_type = validated_data.get('type')
                    final_config_id = validated_data.get('id')
                    final_config_string = validated_data.get('config_str')

                    json_payload = {"id": final_config_id, "type": final_config_type, "config_str": final_config_string}
                    json_payload["rf"] = {"type": "b200", "images_dir": "/usr/share/uhd/images"}

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
                        f"--- ORIGINAL REQUEST ---\n{user_request}"
                    )

                    initial_response_text = generate_response(model, tokenizer, controller_correction_prompt)
                    logging.info("="*20 + " RE-VALIDATING CONTROLLER CORRECTION " + "="*20)
                    validated_data = response_validation_loop(initial_response_text, config_type, user_request)

                    if not validated_data:
                        logging.error("The LLM produced a syntactically invalid configuration while trying to correct a controller error. Aborting this component.")
                        break
            else:
                logging.error(f"Could not obtain a valid and non-empty '{config_type}' configuration after all attempts. Skipping this component.")
                continue

        logging.info("\nProduction run finished processing all components.")
