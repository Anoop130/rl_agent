# dpo_miner.py
import json
import sys
from typing import Dict, Any, List
from collections import defaultdict


def load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _normalize_prompt(p: str) -> str:
    """
    Produces a normalized prompt key so that records from different phases
    can be grouped together. If the augmented prompt contains a USER REQUEST
    section, only that section is used. Otherwise the full prompt is used.
    """
    if not isinstance(p, str):
        return ""
    tag = "--- USER REQUEST ---"
    if tag in p:
        return p.split(tag, 1)[1].strip()
    return p.strip()


def mine_pairs(records: List[Dict[str, Any]], min_gap: float = 0.2) -> List[Dict[str, Any]]:
    """
    Builds preference pairs where the chosen item has a higher total reward
    than the rejected item by at least min_gap. Groups across phases using
    a normalized prompt key and component so that A, B and C attempts are comparable.
    """
    # Group by logical task identity: component and normalized prompt
    groups = defaultdict(list)
    for r in records:
        if not r or "reward" not in r or r["reward"] is None:
            continue
        comp = r.get("component", "")
        norm = _normalize_prompt(r.get("prompt", ""))
        groups[(comp, norm)].append(r)

    pairs: List[Dict[str, Any]] = []
    for _, recs in groups.items():
        candidates = [r for r in recs if r.get("reward") and isinstance(r["reward"].get("total"), (int, float))]
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda r: r["reward"]["total"], reverse=True)
        best = candidates[0]
        for bad in candidates[1:]:
            gap = best["reward"]["total"] - bad["reward"]["total"]
            if gap >= min_gap:
                pairs.append(
                    {
                        "prompt": _normalize_prompt(best["prompt"]),
                        "component": best.get("component"),
                        "chosen": best.get("raw"),
                        "rejected": bad.get("raw"),
                        "chosen_reward": best.get("reward"),
                        "rejected_reward": bad.get("reward"),
                    }
                )
    return pairs


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/reward_runs.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "logs/dpo_pairs.jsonl"
    recs = load(path)
    pairs = mine_pairs(recs, min_gap=0.2)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {out}")
