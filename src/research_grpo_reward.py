"""Versioned research trajectory rewards. Never inspect final answers or test data."""
from collections import Counter

from .research_checker import call_signature, protocol_plan
from .research_environment import ResearchEnvironment
from .reward_v2 import _same_value, _well_formed
from .rewards import compute_reward

VERSIONS = {"reward-v1-legacy", "reward-v2-research-v1"}


def score(record: dict, trace: list[dict], version: str) -> dict:
    if record["split"] != "train" or version not in VERSIONS:
        raise ValueError("train-only, explicit research reward version required")
    plan, _ = protocol_plan(record)
    expected = [{"name": s["name"], "arguments": dict(s["arguments"])} for s in plan]
    dynamic = {(e["target"], e["target_argument"]): e["source"] for e in record["dependency_graph"]}
    # Preserve legacy V1's dynamic placeholder deficiency: never give G1 gold values.
    legacy = [{"name": s["name"], "arguments": {k: "__dynamic__" if (i, k) in dynamic else v
               for k, v in s["arguments"].items()}} for i, s in enumerate(expected)]
    calls = [{"name": s["name"], "arguments": s["arguments"]} for s in trace]
    if version == "reward-v1-legacy":
        return {"version": version, "total": compute_reward(calls, legacy)}
    zero = dict.fromkeys(("format", "tool", "sequence", "keys", "static", "dynamic", "completion", "cap"), 0.0)
    if not calls or not all(_well_formed(c) for c in calls):
        return {"version": version, "total": -3.0, "baseline": -3.0, **zero}
    env = ResearchEnvironment(record["environment_snapshot"])
    truthful = [env.execute(s["name"], s["arguments"]) == s["observation"]
                and not s["observation"].startswith("Error:") for s in trace]
    def matching(i):
        try:
            return call_signature(calls[i]["name"], calls[i]["arguments"]) == call_signature(plan[i]["name"], plan[i]["arguments"])
        except (ValueError, TypeError, SyntaxError, ArithmeticError):
            return False
    names, gold = [c["name"] for c in calls], [c["name"] for c in expected]
    n = max(len(calls), len(expected))
    prefix = 0
    for a, b in zip(names, gold, strict=False):
        if a != b:
            break
        prefix += 1
    keys = static = dependent = 0.0
    denom = sum(len(s["arguments"]) for s in expected) or 1
    matched_all = len(calls) == len(expected) and all(truthful)
    for i, (actual, reference) in enumerate(zip(calls, expected, strict=False)):
        if actual["name"] != reference["name"]:
            matched_all = False
            continue
        a, b = actual["arguments"], reference["arguments"]
        union = set(a) | set(b)
        keys += len(set(a) & set(b)) / len(union) if union else 1.0
        matched_all = matched_all and matching(i)
        for key, value in b.items():
            if (i, key) in dynamic:
                source = dynamic[i, key]
                valid = (key in a and matching(i) and source < i and source < len(trace)
                         and all(matching(j) and truthful[j] for j in range(source + 1))
                         and trace[source]["turn"] < trace[i]["turn"])
                dependent += float(valid)
                matched_all = matched_all and valid
            elif key in a:
                static += float(_same_value(key, value, a[key]))
    parts = {"baseline": -3.0, "format": 1.0,
             "tool": sum((Counter(names) & Counter(gold)).values()) / n,
             "sequence": 1.5 * (0.5 * prefix / n + 0.25 * float(names == gold) + 0.25 * min(len(calls), len(expected)) / n),
             "keys": keys / n, "static": 1.5 * static / denom,
             "dynamic": 1.5 * dependent / denom, "completion": float(matched_all)}
    raw = sum(parts.values())
    parts["cap"] = min(0.0, 2.0 - raw) if not matched_all else 0.0
    return {"version": version, "total": raw + parts["cap"], **parts}
