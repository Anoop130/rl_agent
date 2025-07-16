# test_validator.py
import unittest
from validator import ResponseValidator

class TestResponseValidator(unittest.TestCase):
    """Unit tests for the ResponseValidator class."""

    def test_happy_path_jammer_yaml(self):
        """Tests a perfect YAML response for a jammer."""
        llm_response = """
        This is some conversational preamble from the LLM.
        
        type: jammer
        id: jammer_gps_l1_001

        ### YAML Output:
        ```yaml
        center_frequency: 1.57542e9
        bandwidth: 10e6
        tx_gain: 60
        amplitude: 0.9
        amplitude_width: 0.1
        sampling_freq: 20e6
        num_samples: 20000
        ```
        And some text after the block.
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNotNone(result, "Processing should succeed for valid jammer YAML.")
        self.assertEqual(validator.errors, [], "There should be no errors for a valid response.")
        self.assertEqual(result['type'], 'jammer')
        self.assertEqual(result['id'], 'jammer_gps_l1_001')
        self.assertIn('center_frequency: 1575420000.0', result['config_file'])
        self.assertIn('tx_gain: 60', result['config_file'])
        self.assertIn('write_iq: false', result['config_file']) # Default value

    def test_happy_path_sniffer_toml(self):
        """Tests a perfect TOML response for a sniffer."""
        llm_response = """
        type: sniffer
        id: sniffer_5g_main_002

        ### TOML Output:
        ```toml
        [sniffer]
        file_path = "/data/capture.iq"
        sample_rate = 23040000
        frequency = 1842500000
        nid_1 = 1
        ssb_numerology = 0

        [[pdcch]]
        coreset_id = 1
        num_prbs = 50
        dci_sizes_list = [41, 42]
        ```
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNotNone(result, "Processing should succeed for valid sniffer TOML.")
        self.assertEqual(validator.errors, [], "There should be no errors for a valid response.")
        self.assertEqual(result['type'], 'sniffer')
        self.assertEqual(result['id'], 'sniffer_5g_main_002')
        self.assertIn('[sniffer]', result['config_file'])
        self.assertIn('sample_rate = 23040000', result['config_file'])
        self.assertIn('[[pdcch]]', result['config_file'])
        self.assertIn('dci_sizes_list = [41, 42]', result['config_file'])

    def test_numeric_values_as_strings(self):
        """Tests if numeric values provided as strings are correctly converted."""
        llm_response = """
        type: jammer
        id: jammer_string_vals_001
        ### YAML Output:
        center_frequency: "9.15e8"
        bandwidth: "10000000"
        tx_gain: "55"
        amplitude: "0.8"
        amplitude_width: "0.2"
        sampling_freq: "20e6"
        num_samples: "40000"
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNotNone(result)
        self.assertEqual(validator.errors, [])
        # Check that the compiled config string has numbers, not quoted strings
        self.assertIn('center_frequency: 915000000.0', result['config_file'])
        self.assertIn('tx_gain: 55', result['config_file'])
        self.assertIn('num_samples: 40000', result['config_file'])

    def test_malformed_yaml_parsing_error(self):
        """Tests response with improperly indented YAML."""
        llm_response = """
        type: jammer
        id: jammer_fail_001
        ### YAML Output:
        center_frequency: 1.5e9
          bandwidth: 10e6 # Bad indent
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()
        
        self.assertIsNone(result)
        self.assertIn("YAML parsing failed", validator.errors[0])

    def test_malformed_toml_parsing_error(self):
        """Tests response with invalid TOML syntax."""
        llm_response = """
        type: sniffer
        id: sniffer_fail_001
        ### TOML Output:
        [sniffer]
        file_path = /data/capture.iq # Missing quotes
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNone(result)
        self.assertIn("TOML parsing failed", validator.errors[0])

    def test_jammer_validation_missing_required_key(self):
        """Tests jammer config validation with a missing required key."""
        llm_response = """
        type: jammer
        id: jammer_incomplete_001
        ### YAML Output:
        bandwidth: 10e6
        tx_gain: 60
        amplitude: 0.9
        amplitude_width: 0.1
        sampling_freq: 20e6
        num_samples: 20000
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNone(result)
        self.assertIn('Missing required parameter: center_frequency', validator.errors)

    def test_jammer_validation_value_out_of_range(self):
        """Tests jammer config with a value outside its allowed range."""
        llm_response = """
        type: jammer
        id: jammer_invalid_val_001
        ### YAML Output:
        center_frequency: 1.5e9
        bandwidth: 10e6
        tx_gain: 60
        amplitude: -0.5 # Invalid: must be between 0 and 1
        amplitude_width: 0.1
        sampling_freq: 20e6
        num_samples: 20000
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNone(result)
        self.assertIn('Amplitude must be between 0 and 1', validator.errors)
        
    def test_sniffer_validation_missing_sniffer_section(self):
        """Tests sniffer config validation when [sniffer] section is missing."""
        llm_response = """
        type: sniffer
        id: sniffer_incomplete_001
        ### TOML Output:
        [[pdcch]]
        coreset_id = 1
        num_prbs = 50
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNone(result)
        self.assertIn('Missing [sniffer] section', validator.errors)

    def test_sniffer_validation_empty_pdcch_list(self):
        """Tests sniffer config validation with an empty pdcch list."""
        llm_response = """
        type: sniffer
        id: sniffer_empty_pdcch_001
        ### TOML Output:
        [sniffer]
        file_path = "/data/capture.iq"
        sample_rate = 23040000
        frequency = 1842500000
        nid_1 = 1
        ssb_numerology = 0
        pdcch = [] # Explicitly empty list
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNone(result)
        self.assertIn('PDCCH config must be a non-empty list', validator.errors)

    def test_extraction_failure_missing_id(self):
        """Tests a valid config block but with a missing 'id' in the preamble."""
        llm_response = """
        type: jammer
        # The id line is missing
        ### YAML Output:
        center_frequency: 1.5e9
        bandwidth: 10e6
        tx_gain: 60
        amplitude: 0.9
        amplitude_width: 0.1
        sampling_freq: 20e6
        num_samples: 20000
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNone(result)
        self.assertIn("Could not extract 'id' from response", validator.errors)
        
    def test_case_insensitive_type_id_extraction(self):
        """Tests if type and id extraction is case-insensitive and handles spacing."""
        llm_response = """
        TypE:   jammer
        ID= jammer_case_001
        
        ### YAML Output:
        center_frequency: 1.5e9
        bandwidth: 10e6
        tx_gain: 60
        amplitude: 0.9
        amplitude_width: 0.1
        sampling_freq: 20e6
        num_samples: 20000
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'jammer')
        self.assertEqual(result['id'], 'jammer_case_001')

    def test_endpoint_determination_failure(self):
        """Tests a config with ambiguous keys that don't match jammer or sniffer."""
        llm_response = """
        type: unknown_device
        id: unknown_001
        ### YAML Output:
        foo: "bar"
        baz: 123
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()

        self.assertIsNone(result)
        self.assertIn("Could not determine endpoint type from configuration", validator.errors)

    def test_conf_format_is_parsed_but_fails_validation(self):
        """Tests if a CONF/INI file is parsed, but fails endpoint determination as expected."""
        llm_response = """
        type: jammer
        id: jammer_conf_001
        ### CONF Output:
        ```ini
        [jammer_settings]
        center_frequency = 1.5e9
        bandwidth = 10e6
        tx_gain = 60
        amplitude = 0.9
        amplitude_width = 0.1
        sampling_freq = 20e6
        num_samples = 20000
        ```
        """
        validator = ResponseValidator(llm_response)
        # Test parsing alone
        config = validator.extract_config()
        self.assertIsNotNone(config)
        self.assertIn('jammer_settings', config)
        self.assertEqual(config['jammer_settings']['tx_gain'], '60')

        # Test the full pipeline
        validator.errors = [] # Reset errors
        result = validator.process_response()
        self.assertIsNone(result, "Pipeline should fail because endpoint cannot be determined from CONF structure.")
        self.assertIn("Could not determine endpoint type from configuration", validator.errors)
        
    def test_no_config_block_found(self):
        """Tests a response that has no '### ... Output:' header."""
        llm_response = """
        I'm sorry, I cannot generate that configuration.
        type: none
        id: none_000
        """
        validator = ResponseValidator(llm_response)
        result = validator.process_response()
        self.assertIsNone(result)
        self.assertIn("No valid configuration found", validator.errors)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
