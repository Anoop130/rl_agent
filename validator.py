# validator.py
# Updated to handle both 'sniffer' and 'jammer' configurations.

import re
import json
import logging

class ResponseValidator:
    # MODIFIED: __init__ now accepts an optional 'config_type' to guide validation.
    def __init__(self, raw_response: str, config_type: str = None):
        self.raw_response = raw_response.strip()
        self.config_type = config_type  # This tells us which rules to apply
        self.errors = []
        self.parsed_data = None
        logging.info(f"Validator initialized for type: '{self.config_type}'")

        # --- Schema Definitions ---
        self.jammer_schema = {
            "id": str, "center_frequency": (float, int), "bandwidth": (float, int),
            "amplitude": (float, int), "amplitude_width": (float, int),
            "initial_phase": (float, int), "sampling_freq": (float, int),
            "num_samples": int, "output_iq_file": str, "output_csv_file": str,
            "write_iq": bool, "write_csv": bool, "tx_gain": (float, int)
        }

        # ADDED: Sniffer schema based on your request
        self.sniffer_schema = {
            "id": str, "file_path": str, "sample_rate": (float, int),
            "frequency": (float, int), "nid_1": int, "ssb_numerology": int,
            "pdcch_coreset_id": int, "pdcch_subcarrier_offset": int,
            "pdcch_num_prbs": int, "pdcch_numerology": int,
            "pdcch_dci_sizes_list": list, "pdcch_scrambling_id_start": int,
            "pdcch_scrambling_id_end": int, "pdcch_rnti_start": int,
            "pdcch_rnti_end": int, "pdcch_interleaving_pattern": str,
            "pdcch_coreset_duration": int, "pdcch_AL_corr_thresholds": list,
            "pdcch_num_candidates_per_AL": list
        }

    def _extract_json(self):
        """Extracts a JSON string from the raw response."""
        match = re.search(r"```(?:json)?\n([\s\S]*?)```", self.raw_response)
        if match:
            return match.group(1).strip()

        start = self.raw_response.find('{')
        end = self.raw_response.rfind('}')
        if start != -1 and end != -1:
            return self.raw_response[start:end+1]

        self.errors.append("No valid JSON structure (e.g., '{...}') found in the response.")
        return None

    def _parse_json(self, json_str: str):
        """Tries to parse the JSON string into a Python dictionary."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.errors.append(f"JSON parsing failed: {str(e)}")
            return None

    # REFACTORED: A generic schema validator
    def _validate_schema(self, data: dict, schema: dict):
        """A generic function to validate data against a given schema for keys and types."""
        for key, expected_type in schema.items():
            if key not in data:
                self.errors.append(f"Missing required key: '{key}'")
                continue # Skip type check if key is missing
            if not isinstance(data[key], expected_type):
                self.errors.append(
                    f"Invalid type for key '{key}': expected {expected_type}, but got {type(data[key])}"
                )

    def _validate_jammer_values(self, data: dict):
        """Specific value-range validation for jammer configs."""
        if not self.errors: # Only check values if the schema is correct
            if data.get("center_frequency", 0) <= 0: self.errors.append("center_frequency must be > 0")
            if data.get("bandwidth", 0) <= 0: self.errors.append("bandwidth must be > 0")
            if not (0 <= data.get("amplitude", -1) <= 1): self.errors.append("amplitude must be between 0 and 1")
            if not (0 <= data.get("amplitude_width", -1) <= 1): self.errors.append("amplitude_width must be between 0 and 1")
            if data.get("sampling_freq", 0) <= 0: self.errors.append("sampling_freq must be > 0")
            if data.get("num_samples", 0) <= 0: self.errors.append("num_samples must be > 0")
            if data.get("tx_gain", -1) < 0: self.errors.append("tx_gain must be >= 0")

    # ADDED: A new function for sniffer-specific value checks
    def _validate_sniffer_values(self, data: dict):
        """Specific value-range validation for sniffer configs."""
        if not self.errors: # Only check values if the schema is correct
            if data.get("sample_rate", 0) <= 0: self.errors.append("sample_rate must be > 0")
            if data.get("frequency", 0) <= 0: self.errors.append("frequency must be > 0")
            if data.get("pdcch_num_prbs", 0) <= 0: self.errors.append("pdcch_num_prbs must be > 0")
            if data.get("pdcch_coreset_duration", 0) <= 0: self.errors.append("pdcch_coreset_duration must be > 0")
            if data.get("pdcch_scrambling_id_end", 0) < data.get("pdcch_scrambling_id_start", 1):
                self.errors.append("pdcch_scrambling_id_end must be >= pdcch_scrambling_id_start")
            if data.get("pdcch_rnti_end", 0) < data.get("pdcch_rnti_start", 1):
                self.errors.append("pdcch_rnti_end must be >= pdcch_rnti_start")

    # MODIFIED: The main 'validate' method now orchestrates the correct checks.
    def validate(self):
        """The main validation orchestrator."""
        json_str = self._extract_json()
        if not json_str:
            return None

        self.parsed_data = self._parse_json(json_str)
        if not self.parsed_data:
            return None

        # Use the config_type to decide which validation to run
        if self.config_type == "jammer":
            self._validate_schema(self.parsed_data, self.jammer_schema)
            self._validate_jammer_values(self.parsed_data)
        elif self.config_type == "sniffer":
            self._validate_schema(self.parsed_data, self.sniffer_schema)
            self._validate_sniffer_values(self.parsed_data)
        else:
            # If no specific type, we can't do detailed validation
            self.errors.append(f"Unknown or unspecified config_type: '{self.config_type}'. Cannot perform detailed validation.")
            
        # If there are any errors after all checks, return None. Otherwise, return the parsed data.
        if self.errors:
            return None
        return self.parsed_data

    def get_errors(self):
        return self.errors