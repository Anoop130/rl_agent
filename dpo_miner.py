# dpo_miner.py
import json, sys
from typing import Dict, Any, List

def load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def mine_pairs(records: List[Dict[str, Any]], min_gap: float = 0.2):
    # group by (phase, component, prompt)
    key = lambda r: (r["phase"], r["component"], r["prompt"])
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        if "reward" in r and r["reward"] is not None:
            groups[key(r)].append(r)

    pairs = []
    for _, recs in groups.items():
        recs = [r for r in recs if r["reward"] and "total" in r["reward"]]
        recs.sort(key=lambda r: r["reward"]["total"], reverse=True)
        if len(recs) < 2:
            continue
        best = recs[0]
        for bad in recs[1:]:
            gap = best["reward"]["total"] - bad["reward"]["total"]
            if gap >= min_gap:
                pairs.append({
                    "prompt": best["prompt"],
                    "chosen": best["raw"],
                    "rejected": bad["raw"],
                    "component": best["component"],
                    "chosen_reward": best["reward"],
                    "rejected_reward": bad["reward"],
                })
    return pairs

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/reward_runs.jsonl"
    out  = sys.argv[2] if len(sys.argv) > 2 else "logs/dpo_pairs.jsonl"
    recs = load(path)
    pairs = mine_pairs(recs, min_gap=0.2)
    with open(out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {out}")
