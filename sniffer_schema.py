from jsonformer import Jsonformer

# Flattened sniffer schema with two PDCCH entries
sniffer_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "file_path": {"type": "string", "default": "/sniffer/captures/sniffer_01.fc32"},
        "sample_rate": {"type": "number", "default": 20e6},
        "frequency": {"type": "number"},
        "nid_1": {"type": "number"},
        "ssb_numerology": {"type": "number"},

        # Flattened PDCCH entry 0
        "pdcch_0_coreset_id": {"type": "number"},
        "pdcch_0_subcarrier_offset": {"type": "number"},
        "pdcch_0_num_prbs": {"type": "number"},
        "pdcch_0_numerology": {"type": "number"},
        "pdcch_0_dci_sizes_list": {
            "type": "array", "items": {"type": "number"}
        },
        "pdcch_0_scrambling_id_start": {"type": "number"},
        "pdcch_0_scrambling_id_end": {"type": "number"},
        "pdcch_0_rnti_start": {"type": "number", "default": 65500},
        "pdcch_0_rnti_end": {"type": "number", "default": 65510},
        "pdcch_0_interleaving_pattern": {"type": "string", "default": "non-interleaved"},
        "pdcch_0_coreset_duration": {"type": "number"},
        "pdcch_0_AL_corr_thresholds": {
            "type": "array", "items": {"type": "number"}
        },
        "pdcch_0_num_candidates_per_AL": {
            "type": "array", "items": {"type": "number"}
        },

        # Flattened PDCCH entry 1
        "pdcch_1_coreset_id": {"type": "number"},
        "pdcch_1_subcarrier_offset": {"type": "number"},
        "pdcch_1_num_prbs": {"type": "number"},
        "pdcch_1_numerology": {"type": "number"},
        "pdcch_1_dci_sizes_list": {
            "type": "array", "items": {"type": "number"}
        },
        "pdcch_1_scrambling_id_start": {"type": "number"},
        "pdcch_1_scrambling_id_end": {"type": "number"},
        "pdcch_1_rnti_start": {"type": "number", "default": 65500},
        "pdcch_1_rnti_end": {"type": "number", "default": 65510},
        "pdcch_1_interleaving_pattern": {"type": "string", "default": "non-interleaved"},
        "pdcch_1_coreset_duration": {"type": "number"},
        "pdcch_1_AL_corr_thresholds": {
            "type": "array", "items": {"type": "number"}
        },
        "pdcch_1_num_candidates_per_AL": {
            "type": "array", "items": {"type": "number"}
        }
    },
    "required": [
        "id", "file_path", "sample_rate", "frequency", "nid_1", "ssb_numerology",
        "pdcch_0_coreset_id", "pdcch_0_subcarrier_offset", "pdcch_0_num_prbs", "pdcch_0_numerology",
        "pdcch_0_dci_sizes_list", "pdcch_0_scrambling_id_start", "pdcch_0_scrambling_id_end",
        "pdcch_0_coreset_duration", "pdcch_0_AL_corr_thresholds", "pdcch_0_num_candidates_per_AL",

        "pdcch_1_coreset_id", "pdcch_1_subcarrier_offset", "pdcch_1_num_prbs", "pdcch_1_numerology",
        "pdcch_1_dci_sizes_list", "pdcch_1_scrambling_id_start", "pdcch_1_scrambling_id_end",
        "pdcch_1_coreset_duration", "pdcch_1_AL_corr_thresholds", "pdcch_1_num_candidates_per_AL"
    ]
}
