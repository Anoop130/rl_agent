# reward_api.py
from __future__ import annotations
import math
from typing import Optional, Dict, Any, List
from validator import ResponseValidator

# Canonical values and soft ranges used for scoring
CANON = {
    "sniffer": {"sample_rates": [15.36e6, 23.04e6, 30.72e6, 61.44e6]},
    "rtue": {"rf_srate": [15.36e6, 23.04e6, 30.72e6, 61.44e6]},
}

# Soft range envelopes for fields. These inform R2, while some domain rules
# are enforced as hard negatives below.
RANGES = {
    "sniffer": {
        # Broad envelope. A hard negative below enforces NR band membership.
        "sample_rate": (1e6, 61.44e6),
        "frequency": (410e6, 52.6e9),
        "pdcch_num_prbs": (1, 275),
        "ssb_numerology": (0, 4),
    },
    "jammer": {
        "bandwidth": (1e3, 200e6),
        "amplitude": (0.0, 1.0),
        "tx_gain": (0.0, 90.0),
        "sampling_freq": (1e6, 61.44e6),
        "center_frequency": (410e6, 52.6e9),
    },
    "rtue": {
        "rf_srate": (7.68e6, 61.44e6),
        "rf_tx_gain": (0, 90),
        "rf_rx_gain": (0, 90),
        "rat_nr_nof_prb": (1, 275),
        "rat_nr_max_nof_prb": (1, 275),
        "rat_nr_nof_carriers": (0, 4),
        "rat_eutra_nof_carriers": (0, 4),
    },
}

def _soft_range_score(x: float, lo: float, hi: float, tol_frac: float = 0.1) -> float:
    if x is None:
        return 0.0
    if lo <= x <= hi:
        return 1.0
    span = max((hi - lo) * tol_frac, 1e-12)
    dist = abs(x - (lo if x < lo else hi))
    return max(0.0, 1.0 - dist / span)

def _closest(a: float, anchors: List[float], tol_frac: float = 0.01) -> float:
    if a is None or not anchors:
        return 0.0
    best = min(abs(a - t) for t in anchors)
    tol = max(abs(a) * tol_frac, 1e-9)
    if best == 0:
        return 1.0
    if best <= tol:
        return 0.5
    return 0.0

def _is_nr_band_freq_ok(f_hz: float) -> bool:
    """True if frequency is inside NR FR1 or FR2."""
    in_fr1 = 410e6 <= f_hz <= 7125e6
    in_fr2 = 24.25e9 <= f_hz <= 52.6e9
    return in_fr1 or in_fr2

def score_config(
    raw_text: str,
    component: str,
    prev_json: Optional[Dict[str, Any]] = None,
    phase: str = "A",
) -> Dict[str, Any]:
    """
    Computes reward signals R0 through R6 and a total in [0, 1].
    Uses the existing ResponseValidator to parse and type check the JSON.
    Does not mutate inputs.
    """
    v = ResponseValidator(raw_text, config_type=component)
    _ = v.validate()  # populates v.parsed_data and v.errors
    parsed = v.parsed_data

    # R0: JSON presence and basic structure
    R0 = 1.0 if parsed else 0.0
    if R0 == 0.0:
        return {"R0": 0.0, "total": 0.0, "hard_negative": True, "reasons": ["json.parse.fail"]}

    # R1: schema conformance and type sanity
    schema_errors = [
        e
        for e in v.get_errors()
        if ("Missing required key" in e) or ("Invalid type" in e) or ("Unknown key" in e)
    ]
    if component == "sniffer":
        denom = len(v.sniffer_schema)
    elif component == "jammer":
        denom = len(v.jammer_schema)
    elif component == "rtue":
        denom = len(v.rtue_required_keys)
    else:
        denom = 1
    R1 = max(0.0, 1.0 - (len(schema_errors) / max(denom, 1)))

    # R2: soft domain scoring with additional hard negative guards
    hard_neg = False
    reasons: List[str] = []

    # Component specific hard negatives before averaging R2
    if component == "sniffer":
        f = parsed.get("frequency") if isinstance(parsed, dict) else None
        if isinstance(f, (int, float)) and not _is_nr_band_freq_ok(f):
            return {
                "R0": R0,
                "R1": R1,
                "R2": 0.0,
                "R3": 0.0,
                "R4": 0.0,
                "R5": 0.0,
                "R6": 0.0,
                "hard_negative": True,
                "total": 0.0,
                "reasons": ["sniffer.frequency.out_of_NR_bands"],
            }

    r2_list: List[float] = []
    for k, (lo, hi) in RANGES.get(component, {}).items():
        val = parsed.get(k) if isinstance(parsed, dict) else None
        r2_list.append(_soft_range_score(val, lo, hi))

        # Generic safety checks
        if k == "amplitude" and val is not None and not (lo <= val <= hi):
            hard_neg = True
            reasons.append("safety.amplitude")
        if k in ("frequency", "center_frequency") and val is not None and val <= 0:
            hard_neg = True
            reasons.append("safety.nonpositive_freq")

    R2 = sum(r2_list) / len(r2_list) if r2_list else 0.0
    if hard_neg:
        return {
            "R0": R0,
            "R1": R1,
            "R2": R2,
            "R3": 0.0,
            "R4": 0.0,
            "R5": 0.0,
            "R6": 0.0,
            "hard_negative": True,
            "total": 0.0,
            "reasons": reasons,
        }

    # R3: cross field consistency checks
    R3_checks: List[float] = []
    if component == "jammer":
        sf = parsed.get("sampling_freq")
        bw = parsed.get("bandwidth")
        if isinstance(sf, (int, float)) and isinstance(bw, (int, float)):
            R3_checks.append(1.0 if sf >= 2.0 * bw else 0.0)

    elif component == "rtue":
        srate = parsed.get("rf_srate")
        prb = parsed.get("rat_nr_nof_prb")
        prb_max = parsed.get("rat_nr_max_nof_prb")
        if isinstance(srate, (int, float)) and isinstance(prb, (int, float)):
            if abs(srate - 23.04e6) < 1e5 or abs(srate - 30.72e6) < 1e5:
                R3_checks.append(1.0 if prb == 106 else 0.0)
            else:
                R3_checks.append(0.5)
        if isinstance(prb_max, (int, float)) and isinstance(prb, (int, float)):
            R3_checks.append(1.0 if prb_max >= prb else 0.0)

    R3 = sum(R3_checks) / len(R3_checks) if R3_checks else 0.0

    # R4: canonical snaps to preferred anchor values
    R4 = 0.0
    if component == "sniffer":
        R4 = _closest(parsed.get("sample_rate"), CANON["sniffer"]["sample_rates"])
    elif component == "rtue":
        R4 = _closest(parsed.get("rf_srate"), CANON["rtue"]["rf_srate"])

    # R5: minimal repair distance relative to previous JSON
    R5 = 0.0
    if prev_json and isinstance(parsed, dict):
        touched = 0
        changed = 0
        for k, v_prev in prev_json.items():
            if k in parsed:
                touched += 1
                if parsed[k] != v_prev:
                    changed += 1
        if touched:
            R5 = 1.0 - (changed / touched)

    # R6: diversity placeholder
    R6 = 0.0

    # Phase weights
    weights_by_phase = {
        "A": {"R0": 0.40, "R1": 0.35, "R2": 0.25},
        "B": {"R0": 0.30, "R1": 0.25, "R2": 0.25, "R3": 0.20},
        "C": {"R0": 0.25, "R1": 0.20, "R2": 0.20, "R3": 0.20, "R4": 0.10, "R5": 0.05},
    }
    W = weights_by_phase.get(phase, weights_by_phase["A"])

    # Total score
    total = 0.0
    locals_map = {"R0": R0, "R1": R1, "R2": R2, "R3": R3, "R4": R4, "R5": R5, "R6": R6}
    for k, w in W.items():
        total += w * locals_map.get(k, 0.0)
    total = max(0.0, min(1.0, total))

    return {
        "R0": R0,
        "R1": R1,
        "R2": R2,
        "R3": R3,
        "R4": R4,
        "R5": R5,
        "R6": R6,
        "hard_negative": False,
        "total": total,
        "reasons": [],
    }
