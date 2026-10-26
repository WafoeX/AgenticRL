#!/usr/bin/env python3
"""Explicit create or read-only check of the operational holdout; no model/test inference."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research_registration import (  # noqa: E402
    create_registration,
    verify_registration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true", help="one-time byte-preserving seal; refuses existing release/lock")
    mode.add_argument("--check", action="store_true", help="read-only hashes/counts/source verification; no label parsing")
    args = parser.parse_args()
    try:
        receipt = create_registration(ROOT) if args.create else verify_registration(ROOT)
        print("PASS: " + json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        print("NOTE: budget/readiness annex and separate authorization required; no training/GPU/model inference")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
