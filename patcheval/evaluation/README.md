# Patch Evaluation

This directory contains the PatchEval PoC evaluator. It evaluates an existing patch JSON/JSONL file by mounting each patch into the corresponding CVE Docker image and running the case validation script.

## Input format

`run_evaluation.py` expects a JSON or JSONL patch file. Each item should contain:

```json
{
  "cve": "CVE-2021-23376",
  "fix_patch": "diff --git ..."
}
```

A `language` field may be supplied. If absent, language is read from:

```text
../datasets/patcheval_230.json
```

## Run manually

```bash
cd patcheval/evaluation
python run_evaluation.py \
  --output example \
  --patch_file ./example_patch.json \
  --input_file ../datasets/patcheval_230.json \
  --images_file ../../scripts/images.txt \
  --max_workers 4 \
  --log_level INFO
```

## Output

Results are written under the current working directory:

```text
evaluation_output/<output>/
├── run_evaluation.log
├── summary_report.txt
├── summary.json
└── logs/<CVE>/
    ├── fix.patch
    └── success_output.log or erro_output.log
```

When invoked from `patcheval/exp_agent/run_eval.sh`, the output name is `results/<prefix>`, so results are written to:

```text
patcheval/evaluation/evaluation_output/results/<prefix>/
```

## Notes

- The evaluator uses `scripts/images.txt` to map CVE IDs to Docker images.
- `patcheval_230.json` provides CVE metadata such as `programing_language`.
- The current verified workflow evaluates patches with the case PoC validation command.
