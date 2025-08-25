# main.py (Standalone, Unified RAG-Powered System — no controller)

import os
import sys
import json
import yaml
import torch
import urllib3
import logging
import argparse
import pathlib
from typing import List, Dict, Optional, Any

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# --- RAG Imports ---
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer  # noqa: F401 (kept available if you switch EF)
from reward_api import score_config
from reward_log import RewardLogger


from validator import ResponseValidator

# --- Globals ---
model = None
tokenizer = None
retriever = None


class Config:
    filename: str = ""
    options: Optional[Dict[str, Any]] = None
    log_level: int = logging.DEBUG
    standalone: bool = True  # always standalone in this file


# --- RAG SYSTEM ---
class RAGRetriever:
    def __init__(
        self,
        db_dir: str = "vector_db",
        collection_name: str = "rf_knowledge",
        model_name: str = "all-MiniLM-L6-v2",
        suppress_tf_logs: bool = True,
    ):
        """
        Robust Chroma retriever:
          - tries to get the collection; if missing, creates it
          - safe when DB is empty (returns "")
          - clean logging
        """
        if suppress_tf_logs:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        logging.info("Initializing RAG Retriever...")
        try:
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        except Exception as e:
            logging.warning(f"Embedding init failed ({e}). Falling back to Chroma default embeddings.")
            sentence_transformer_ef = None  # Chroma will use its default function

        try:
            db_client = chromadb.PersistentClient(path=db_dir)
        except Exception as e:
            logging.error(f"FATAL: Could not open ChromaDB at '{db_dir}': {e}")
            sys.exit(1)

        # Get or create collection
        try:
            self.collection = db_client.get_collection(
                name=collection_name,
                embedding_function=sentence_transformer_ef,
            )
            logging.info(f"RAG Retriever: using existing collection '{collection_name}'.")
        except Exception as e_get:
            logging.warning(f"Collection '{collection_name}' not found or failed to open ({e_get}). Creating a new one.")
            try:
                self.collection = db_client.create_collection(
                    name=collection_name,
                    embedding_function=sentence_transformer_ef,
                )
                logging.info(f"RAG Retriever: created empty collection '{collection_name}'.")
            except Exception as e_create:
                logging.error(f"FATAL: Could not create collection '{collection_name}': {e_create}")
                sys.exit(1)

        # Optional: quick count to warn if empty
        try:
            count = self.collection.count()
            if count == 0:
                logging.warning(
                    f"RAG collection '{collection_name}' is empty. Retrieval will return no context until you upsert docs."
                )
            else:
                logging.debug(f"RAG collection '{collection_name}' contains {count} documents.")
        except Exception:
            pass

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """
        Fetch up to n_results relevant chunks. Returns a formatted string.
        Safe on empty DB or no hits.
        """
        if not query or not query.strip():
            logging.debug("RAG: empty query; returning empty context.")
            return ""

        logging.info(
            f"--- RAG: Retrieving context for query: '{query[:200]}{'...' if len(query) > 200 else ''}' ---"
        )
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max(1, n_results),
            )
        except Exception as e:
            logging.warning(f"RAG query failed: {e}. Continuing without context.")
            return ""

        docs = (results or {}).get("documents") or []
        metas = (results or {}).get("metadatas") or []
        if not docs or not docs[0]:
            logging.info("RAG: no matching documents. Proceeding with empty context.")
            return ""

        retrieved_docs = docs[0]
        retrieved_metas = metas[0] if metas else [{}] * len(retrieved_docs)

        lines = []
        for i, doc in enumerate(retrieved_docs):
            meta = retrieved_metas[i] if i < len(retrieved_metas) else {}
            source = (meta or {}).get("source", "unknown")
            preview = (doc or "").replace("\n", " ").strip()
            logging.info(
                f"  [Doc {i+1} from '{source}']: {preview[:120]}{'...' if len(preview) > 120 else ''}"
            )
            lines.append(f"- From {source}:\n{doc}")

        return "\n\n".join(lines).strip()


# --- Core setup ---
def configure() -> None:
    """Parses CLI and loads the YAML config. Forces logging reconfig."""
    parser = argparse.ArgumentParser(
        description="Standalone LLM Worker for RF component configuration (no controller)"
    )
    parser.add_argument("--config", type=pathlib.Path, required=True, help="Path of YAML config for the llm worker")
    parser.add_argument(
        "--log-level", default="INFO", help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="Run in standalone mode for local testing (default in this file)."
    )
    args = parser.parse_args()

    # standalone is always True here
    Config.standalone = True
    Config.log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    if not isinstance(Config.log_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")

    # Force reconfigure logging even if another lib already set handlers
    logging.basicConfig(
        level=Config.log_level,
        format="%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    print(">>> configure(): starting", flush=True)

    Config.filename = str(args.config)
    with open(Config.filename, "r") as file:
        Config.options = yaml.safe_load(file)

    logging.debug(f"Loaded YAML from {Config.filename}")


def generate_response(model, tokenizer, prompt_content: str, max_new_tokens: int = 2048) -> str:
    """
    Simple wrapper for chat-templated generation.
    You can pass a smaller max_new_tokens for quick intent extraction.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [{"role": "user", "content": prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        output_tokens = model.generate(**inputs, generation_config=generation_config)
    input_length = inputs["input_ids"].shape[1]
    newly_generated_tokens = output_tokens[0, input_length:]
    return tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()


# --- Intent & Validation loops (no controller) ---
def get_intent() -> Optional[List[str]]:
    """
    Uses the LLM to extract which components to generate for.
    Expects in YAML:
      user_prompt: "<natural request>"
      intent_prompt: "<prompt template with {user_prompt}>"
    Returns: list like ["sniffer", "rtue"] or None on failure.
    """
    user_prompt: str = (Config.options or {}).get("user_prompt", "") or ""
    intent_prompt: str = (Config.options or {}).get("intent_prompt", "") or ""

    max_attempts = 5
    attempt_count = 1
    current_prompt_content = intent_prompt.replace("{user_prompt}", user_prompt)

    while attempt_count <= max_attempts:
        logging.info(f"[Intent] attempt {attempt_count}/{max_attempts}")
        logging.debug(f"[Intent] prompt >>>\n{current_prompt_content}\n<<<")

        # cap tokens for a fast first output
        raw_response = generate_response(model, tokenizer, current_prompt_content, max_new_tokens=128)
        logging.info(f"[Intent] model output:\n{raw_response}")

        validator = ResponseValidator(raw_response, config_type="intent")
        validated = validator.validate()

        if validated:
            logging.info(f"[Intent] success. components: {validated}")
            return validated  # type: ignore[return-value]

        # on failure, compile concise error feedback and retry
        error_details = "\n".join(validator.get_errors())
        logging.warning(f"[Intent] validation failed.\n{error_details}")
        attempt_count += 1
        if attempt_count > max_attempts:
            break

        current_prompt_content = (
            "The previous intent JSON you provided was invalid for the following reasons:\n"
            f"{error_details}\n\n"
            "Please regenerate the entire, corrected intent JSON array based on the original user request.\n"
            f"--- ORIGINAL USER REQUEST ---\n{user_prompt}"
        )

    logging.error("[Intent] max attempts reached. extraction failed.")
    return None


def response_validation_loop(
    current_response_text: str, config_type: str, original_prompt_content: str
) -> Optional[Dict]:
    if config_type not in ("sniffer", "jammer", "rtue"):
        logging.warning(f"[Validate] unknown config_type '{config_type}'. skipping loop.")
        logging.info("=" * 20 + " FINAL UNVALIDATED OUTPUT " + "=" * 20)
        logging.info(current_response_text)
        return None

    logging.info(f"[Validate] starting loop for '{config_type}'")
    max_attempts = 25
    attempt = 1
    current_text = current_response_text
    prev_json = None  # for R5 minimal-repair

    while attempt <= max_attempts:
        logging.info("=" * 40 + f" VALIDATION ATTEMPT {attempt}/{max_attempts} " + "=" * 40)

        # Reward for this attempt (Phase B; uses prev_json for R5 when available)
        rinfo = score_config(current_text, config_type, prev_json=prev_json, phase="B")
        logging.info(f"[REWARD attempt {attempt}] {rinfo}")
        reward_logger.log(
            phase="B",
            component=config_type,
            prompt=original_prompt_content,
            raw_output=current_text,
            reward=rinfo,
            errors=[],
            parsed=None,
            validated_ok=None
        )

        validator = ResponseValidator(current_text, config_type=config_type)
        validated = validator.validate()

        if validated:
            logging.info("[Validate] success. extracting final components.")
            # Optional final reward pass (Phase C) on the validated string
            final_text_for_scoring = current_text
            rfinal = score_config(final_text_for_scoring, config_type, prev_json=prev_json, phase="C")
            logging.info(f"[REWARD final] {rfinal}")
            reward_logger.log(
                phase="C",
                component=config_type,
                prompt=original_prompt_content,
                raw_output=final_text_for_scoring,
                reward=rfinal,
                errors=validator.get_errors(),
                parsed=validator.parsed_data,
                validated_ok=True
            )
            return validated

        # failed: build targeted correction prompt
        errors = validator.get_errors() or []
        error_blob = "\n".join(f"- {e}" for e in errors)
        logging.warning("[Validate] failed. errors:\n" + error_blob)

        attempt += 1
        if attempt > max_attempts:
            logging.error("[Validate] reached max attempts.")
            break

        # store prev_json for R5 on next attempt
        prev_json = validator.parsed_data if isinstance(validator.parsed_data, dict) else prev_json

        correction_prompt = (
            "The previous JSON configuration you provided was invalid for the following reasons:\n"
            f"{error_blob}\n\n"
            "Fix only the invalid fields. Keep all other keys identical. "
            "Return only a corrected JSON object (no comments).\n"
            f"--- ORIGINAL REQUEST ---\n{original_prompt_content}"
        )
        logging.info("[Validate] generating corrected response...")
        current_text = generate_response(model, tokenizer, correction_prompt)
        logging.info("=" * 20 + f" CORRECTED OUTPUT (ATTEMPT {attempt}) " + "=" * 20)
        logging.info(current_text)
        logging.info("=" * 20 + " END CORRECTED OUTPUT " + "=" * 20)

    return None



# ======================================================================
# --- MAIN EXECUTION BLOCK ---
# ======================================================================
if __name__ == "__main__":
    print(">>> main.py: boot", flush=True)
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    configure()
    reward_logger = RewardLogger("logs/reward_runs.jsonl")


    if Config.options is None:
        logging.error(
            f"Failed to load or parse YAML from '{Config.filename}'. The file might be empty, malformed, or not found at that path."
        )
        sys.exit(1)

    # --- 1) Load LLM ---
    model_str = Config.options.get("model", None)
    if not model_str:
        logging.error("Model not specified in config file")
        sys.exit(1)

    print(">>> loading model...", flush=True)
    logging.info(f"Loading LLM: {model_str}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_str, quantization_config=bnb_config, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    logging.info("LLM loaded successfully.")
    print(">>> model loaded", flush=True)

    # --- 2) Init RAG ---
    print(">>> init RAG...", flush=True)
    retriever = RAGRetriever()
    print(">>> RAG ready", flush=True)

    # --- 3) Intent Extraction ---
    print(">>> intent...", flush=True)
    intent_list = get_intent()
    print(f">>> intent done: {intent_list}", flush=True)

    if not intent_list:
        logging.error("Could not determine any valid components from user prompt. Exiting.")
        sys.exit(1)
    logging.info(f"\nIntent processing complete. Will generate configs for: {intent_list}\n")

    # --- 4) Unified Generation and Validation Loop (Standalone) ---
    for config_type in intent_list:
        logging.info("=" * 80)
        logging.info(f"PROCESSING COMPONENT: {config_type.upper()}")
        logging.info("=" * 80)

        # A) Assemble prompt parts
        system_prompt = Config.options.get(config_type, "")
        user_request = Config.options.get("user_prompt", "")
        if not system_prompt or not user_request:
            logging.error(f"System prompt for '{config_type}' or 'user_prompt' not found in config. Skipping.")
            continue

        retrieval_query = (
            f"Rules and engineering constraints for a {config_type} config to fulfill the request: {user_request}"
        )
        # If you want to bypass RAG briefly to debug, uncomment the next line:
        # retrieved_context = ""
        retrieved_context = retriever.retrieve_context(retrieval_query)

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
""".strip(
            "\n"
        )

        # if system prompt had an embedded "### USER REQUEST", drop it for clean instructions-only block
        system_instructions = system_prompt.split("### USER REQUEST:")[0]
        final_augmented_prompt = final_prompt_template.format(
            context=retrieved_context,
            system_prompt_instructions=system_instructions,
            user_request=user_request,
        )

        logging.info("\n--- FINAL AUGMENTED PROMPT FOR LLM ---\n" + final_augmented_prompt + "\n------------------------------------\n")

        # B) Initial generation
        logging.info("Generating initial configuration with RAG...")
        initial_response_text = generate_response(model, tokenizer, final_augmented_prompt)
        
        # --- Reward: pre-validation (Phase A) ---
        pre_reward = score_config(initial_response_text, config_type, prev_json=None, phase="A")
        logging.info(f"[REWARD pre] {pre_reward}")
        reward_logger.log(
            phase="A",
            component=config_type,
            prompt=final_augmented_prompt,
            raw_output=initial_response_text,
            reward=pre_reward,
            errors=[],
            parsed=None,
            validated_ok=None
        )
        # ---end Reward---

        logging.info("\n" + "=" * 20 + f" RAG-POWERED INITIAL CONFIG FOR {config_type.upper()} " + "=" * 20)
        logging.info(f"'{initial_response_text}'")
        logging.info("=" * 20 + " END OF INITIAL CONFIG " + "=" * 20 + "\n")

        # C) Validation & Self-correction
        validated_data = response_validation_loop(initial_response_text, config_type, user_request)
        if not validated_data:
            logging.error(f"Could not obtain a valid config for '{config_type}' after all attempts. Skipping.")
            continue

        logging.info(f"SUCCESS: Validated '{config_type}' config generated.")

        # D) Standalone final output (pretty print)
        logging.info("=" * 20 + f" STANDALONE MODE: FINAL VALIDATED CONFIG FOR {config_type.upper()} " + "=" * 20)
        try:
            logging.info(json.dumps(validated_data, indent=2))
        except TypeError:
            logging.info(str(validated_data))

    logging.info("\nScript finished processing all components.")
