#!/usr/bin/env python3
"""CPU/tokenizer-only R3.1–R3.3 build. Never load weights or read test records."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research_environment import TOOL_SCHEMAS, canonical_json
from src.research_eval_protocol import load_development_records
from src.research_model_identity import file_hash
from src.research_schema import SYSTEM_PROMPT, digest
from src.research_sft import VERSION, build_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/research_v1_frozen"))
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.output.exists():
        raise ValueError("output exists; refusing overwrite")
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        raise ValueError("output must not be inside source dataset")
    transformers = importlib.import_module("transformers")

    records, provenance = load_development_records(args.dataset, split="train")
    tokenizer_dir = args.tokenizer_dir.resolve(strict=True)
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    before = {name: file_hash(tokenizer_dir / name) for name in names}
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False, use_fast=True)
    experts, tokens, summary = build_corpus(records, tokenizer, max_length=args.max_length)
    if not experts:
        raise ValueError("no accepted examples")
    # Confirm input/tokenizer bytes did not change during build.
    _, after = load_development_records(args.dataset, split="train")
    if after != provenance or before != {name: file_hash(tokenizer_dir / name) for name in names}:
        raise ValueError("source changed while building")
    artifacts = {
        "experts.jsonl": "".join(canonical_json(e) + "\n" for e in experts),
        "tokens.jsonl": "".join(canonical_json(t) + "\n" for t in tokens),
    }
    manifest = {"version": VERSION, "source": provenance, "summary": summary,
                "tokenizer_files_sha256": before, "chat_template_sha256": digest(tokenizer.chat_template),
                "transformers_version": transformers.__version__, "tokenizer_class": type(tokenizer).__name__,
                "system_prompt_sha256": digest(SYSTEM_PROMPT), "tool_schema_sha256": digest(TOOL_SCHEMAS),
                "max_length": args.max_length, "truncation": "reject-whole-example-no-crop",
                "mask": "assistant-payload-and-im_end-only; headers/context/padding=-100; unshifted",
                "training_performed": False, "packing": False,
                "source_hashes": {p: file_hash(Path(p)) for p in (
                    "src/research_sft.py", "scripts/build_research_sft.py", "src/research_generator.py",
                    "src/research_checker.py", "src/research_answers.py", "src/research_environment.py")},
                "files": {name: {"sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode())}
                          for name, text in artifacts.items()}}
    artifacts["manifest.json"] = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.check:
        if {p.name for p in args.output.iterdir()} != set(artifacts):
            raise ValueError("artifact inventory mismatch")
        for name, text in artifacts.items():
            if (args.output / name).read_bytes() != text.encode():
                raise ValueError("artifact bytes mismatch: " + name)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, text in artifacts.items():
            (args.output / name).write_text(text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
