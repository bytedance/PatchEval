#!/usr/bin/env python3
"""Convert patch_agent_runner.py outputs to PatchEval evaluation JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runner-output", type=Path, required=True, help="Run directory produced by patch_agent_runner.py")
    ap.add_argument("--process-data-path", type=Path, required=True)
    ap.add_argument("--test-data-path", type=Path, required=True, help="patcheval_verified.json")
    args = ap.parse_args()

    dataset = read_json(args.test_data_path)
    cve2lang = {item["cve_id"]: item["programing_language"] for item in dataset}
    cve2image = {item["cve_id"]: item.get("image_url") for item in dataset if item.get("image_url")}
    patch_dir = args.runner_output / "patches"
    if not patch_dir.is_dir():
        raise SystemExit(f"missing patches dir: {patch_dir}")
    rows = []
    for patch_path in sorted(patch_dir.glob("CVE-*.patch")):
        cve = patch_path.stem
        row = {
            "cve": cve,
            "language": cve2lang[cve],
            "fix_patch": patch_path.read_text(encoding="utf-8", errors="replace"),
        }
        if cve in cve2image:
            row["image_url"] = cve2image[cve]
        rows.append(row)
    args.process_data_path.parent.mkdir(parents=True, exist_ok=True)
    with args.process_data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} patches to {args.process_data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
