# reward_api.py
from __future__ import annotations
import math
from typing import Optional, Dict, Any, List
from validator import ResponseValidator

# === Light domain tables (adjust to your hardware/rules) ===
CANON = {
    "sniffer": {"sample_rates": [15.36e6, 30.72e6, 61.44e6]},
}
RANGES = {
    "sniffer": {
        "sample_rate": (1e6, 61.44e6),
        "frequency":   (6e8, 6e9),
        "pdcch_num_prbs": (1, 275),
        "ssb_numerology": (0, 4),
    },
    "jammer": {
        "bandwidth": (1e3, 200e6),
        "amplitude": (0.0, 1.0),
        "tx_gain":   (0.0, 90.0),
        "sampling_freq": (1e6, 61.44e6),
        "center_frequency": (6e8, 6e9),
    },
    "rtue": {
        # Fill if you want soft range scoring on rf_* fields
        # "rf_srate": (1e6, 61.44e6),
        # "rf_tx_gain": (0, 90),
        # "rf_rx_gain": (0, 90),
    }
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

def score_config(
    raw_text: str,
    component: str,
    prev_json: Optional[Dict[str, Any]] = None,
    phase: str = "A",
) -> Dict[str, Any]:
    """
    Returns component scores (R0..R6) and 'total' in [0,1].
    Uses your existing ResponseValidator; does NOT mutate inputs.
    """
    v = ResponseValidator(raw_text, config_type=component)
    _ = v.validate()  # populates v.parsed_data and v.errors
    parsed = v.parsed_data

    # --- R0: JSON/structure gate ---
    R0 = 1.0 if parsed else 0.0
    if R0 == 0.0:
        return {"R0": 0.0, "total": 0.0, "hard_negative": True, "reasons": ["json.parse.fail"]}

    # --- R1: schema/type/unknown ---
    schema_errors = [e for e in v.get_errors() if ("Missing required key" in e) or ("Invalid type" in e) or ("Unknown key" in e)]
    if component == "sniffer":
        denom = len(v.sniffer_schema)
    elif component == "jammer":
        denom = len(v.jammer_schema)
    elif component == "rtue":
        denom = len(v.rtue_required_keys)
    else:
        denom = 1
    R1 = max(0.0, 1.0 - (len(schema_errors) / max(denom, 1)))

    # --- R2: domain ranges (soft) + basic hard-negatives ---
    hard_neg = False
    reasons: List[str] = []
    r2_list = []
    for k, (lo, hi) in RANGES.get(component, {}).items():
        val = parsed.get(k) if isinstance(parsed, dict) else None
        r2_list.append(_soft_range_score(val, lo, hi))
        if k == "amplitude" and val is not None and not (lo <= val <= hi):
            hard_neg = True; reasons.append("safety.amplitude")
        if k in ("frequency", "center_frequency") and val is not None and val <= 0:
            hard_neg = True; reasons.append("safety.nonpositive_freq")
    R2 = sum(r2_list) / len(r2_list) if r2_list else 0.0
    if hard_neg:
        return {
            "R0": R0, "R1": R1, "R2": R2, "R3": 0.0, "R4": 0.0, "R5": 0.0, "R6": 0.0,
            "hard_negative": True, "total": 0.0, "reasons": reasons
        }

    # --- R3: cross-field consistency (examples you have data for) ---
    R3_checks: List[float] = []
    if component == "jammer":
        sf = parsed.get("sampling_freq"); bw = parsed.get("bandwidth")
        if sf is not None and bw is not None:
            R3_checks.append(1.0 if sf >= 2.0 * bw else 0.0)  # Nyquist
    # add more per-component checks as you expose fields
    R3 = sum(R3_checks) / len(R3_checks) if R3_checks else 0.0

    # --- R4: canonical snaps ---
    R4 = 0.0
    if component == "sniffer":
        R4 = _closest(parsed.get("sample_rate"), CANON["sniffer"]["sample_rates"])

    # --- R5: minimal-repair distance (needs prev_json) ---
    R5 = 0.0
    if prev_json and isinstance(parsed, dict):
        touched = 0; changed = 0
        for k, v_prev in prev_json.items():
            if k in parsed:
                touched += 1
                if parsed[k] != v_prev:
                    changed += 1
        if touched:
            R5 = 1.0 - (changed / touched)

    # --- R6: diversity placeholder (0 for now) ---
    R6 = 0.0

    weights_by_phase = {
        "A": {"R0": 0.40, "R1": 0.35, "R2": 0.25},
        "B": {"R0": 0.30, "R1": 0.25, "R2": 0.25, "R3": 0.20},
        "C": {"R0": 0.25, "R1": 0.20, "R2": 0.20, "R3": 0.20, "R4": 0.10, "R5": 0.05},
    }
    W = weights_by_phase.get(phase, weights_by_phase["A"])

    total = 0.0
    locals_map = {"R0": R0, "R1": R1, "R2": R2, "R3": R3, "R4": R4, "R5": R5, "R6": R6}
    for k, w in W.items():
        total += w * locals_map.get(k, 0.0)
    total = max(0.0, min(1.0, total))

    return {
        "R0": R0, "R1": R1, "R2": R2, "R3": R3, "R4": R4, "R5": R5, "R6": R6,
        "hard_negative": False, "total": total, "reasons": []
    }
