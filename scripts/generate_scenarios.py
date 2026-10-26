"""Generate training and validation scenario JSONL files."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scenarios import generate_dataset


# 中文注释：确保目标目录存在，并把场景逐条写成 JSONL。
def write_jsonl(path: str, data: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


# 中文注释：脚本模式下生成固定规模的训练集和验证集。
if __name__ == "__main__":
    train = generate_dataset(count=200, seed=0)
    val = generate_dataset(count=50, seed=10000)

    write_jsonl("data/scenarios_train.jsonl", train)
    write_jsonl("data/scenarios_val.jsonl", val)

    print(f"Generated {len(train)} training + {len(val)} validation scenarios")
