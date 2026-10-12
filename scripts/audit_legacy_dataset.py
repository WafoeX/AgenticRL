"""Audit old data without training, overwriting datasets, or importing ART.

Usage: python scripts/audit_legacy_dataset.py
       python scripts/audit_legacy_dataset.py --output results/research/legacy_audit/report.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_audit import audit_repository  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON under results/research/legacy_audit/")
    parser.add_argument("--overwrite", action="store_true", help="Replace only an existing audit output")
    args = parser.parse_args()
    output = None
    if args.output:
        output = args.output.resolve()
        allowed = (ROOT / "results/research/legacy_audit").resolve()
        if not output.is_relative_to(allowed) or output == allowed:
            parser.error("output must be a file under results/research/legacy_audit/")
        if output.exists() and not args.overwrite:
            parser.error("audit output exists; use --overwrite to replace this report")
    try:
        report = audit_repository(ROOT)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")
    for name, summary in report["datasets"].items():
        print(f"{name}: {summary['rows']} rows, {summary['unique_tasks']} unique task texts")
    overlap = report["overlap"]["train_to_validation"]
    print(f"Validation rows overlapping train: {overlap['right_rows_with_exact_text_in_left']}")
    print(f"Template capacities: {report['raw_task_capacities']}")
    print(f"Comparison winner slots: {report['semantic_probes']['comparison_winner_slots']}")
    if output:
        print(f"Saved: {output.relative_to(ROOT)}")
    else:
        print("Read-only summary; no report file written.")


if __name__ == "__main__":
    main()
