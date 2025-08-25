# reward_log.py
import os, json
from typing import Dict, Any, Optional, List

class RewardLogger:
    def __init__(self, path: str = "logs/reward_runs.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self,
            phase: str,
            component: str,
            prompt: str,
            raw_output: str,
            reward: Dict[str, Any],
            errors: Optional[List[str]] = None,
            parsed: Optional[Dict[str, Any]] = None,
            validated_ok: Optional[bool] = None):
        rec = {
            "phase": phase,
            "component": component,
            "prompt": prompt,
            "raw": raw_output,
            "reward": reward,
            "errors": errors or [],
            "parsed": parsed,
            "validated_ok": bool(validated_ok),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
