# PatchEval CLI Agent Patch Generation

This directory contains a lightweight patch-generation workflow for CLI coding agents.
It is designed to plug into the PatchEval-Verified evaluation flow:

```text
run_infer.sh + patch_agent_runner.py
  -> generate patches
  -> process_data.py
  -> ../evaluation/run_evaluation.py
  -> evaluate generated patches
```

The supported CLI agents share the same runner and differ only in their command and mounts. The shared runner handles Docker setup, prompt generation, patch collection, and output layout.

## Quickstart

Run one smoke case with Codex, then evaluate the generated patch:

```bash
conda activate patcheval
cd patcheval/exp_agent

export CODEX_BIN=/path/to/bin/codex
export CODEX_CONFIG=/path/to/codex-home/my-profile.config.toml

LIMIT=1 CONCURRENCY=1 bash run_infer.sh codex codex_smoke
bash run_eval.sh codex_smoke
```

For OpenCode or TraeCLI, set the corresponding runtime variables and replace
the agent name:

```bash
# OpenCode
conda activate patcheval
cd patcheval/exp_agent

export OPENCODE_BIN=/path/to/bin/opencode
export OPENCODE_CONFIG=/path/to/opencode-home/config/opencode/opencode.json
LIMIT=1 CONCURRENCY=1 bash run_infer.sh opencode opencode_smoke
bash run_eval.sh opencode_smoke

# TraeCLI / TraeX
conda activate patcheval
cd patcheval/exp_agent

export TRAE_BIN=/path/to/bin/traex
export TRAE_CONFIG=/path/to/trae-home/my-profile.traecli.toml
LIMIT=1 CONCURRENCY=1 bash run_infer.sh traecli trae_smoke
bash run_eval.sh trae_smoke
```

Generated patches are written to `agent_runs/<timestamp>-<prefix>/patches/`.
Evaluation results are written to `../evaluation/evaluation_output/results/<prefix>/`.

## Layout

```text
exp_agent/
├── README.md
├── agents/
│   ├── codex.sh       # Codex CLI adapter
│   ├── opencode.sh    # OpenCode adapter
│   └── traecli.sh     # TraeCLI / TraeX adapter
├── patch_agent_runner.py
├── process_data.py
├── run_infer.sh
└── run_eval.sh
```

## Data and images

By default, scripts use:

```text
../datasets/patcheval_verified.json
```

`patcheval_verified.json` provides CVE metadata, prompt content, and the `image_url` used for agent execution and evaluation. `../../scripts/images.txt` contains the same 230 image references for bulk download.

## Step 1: Generate patches

Use the unified inference entrypoint:

```bash
bash run_infer.sh <agent> <prefix>
```

Supported agents:

```text
codex
opencode
traecli
```

The generated run is written under:

```text
agent_runs/<timestamp>-<prefix>/
```

with patches in:

```text
agent_runs/<timestamp>-<prefix>/patches/CVE-....patch
```

Common controls:

```bash
export CONCURRENCY=5             # parallel containers / agents
export LIMIT=-1                  # -1 means all selected cases
```

### Codex

Required environment variables:

```bash
export CODEX_BIN=/path/to/bin/codex
export CODEX_CONFIG=/path/to/codex-home/my-profile.config.toml
```

Run:

```bash
CONCURRENCY=5 bash run_infer.sh codex codex_my_profile
```

The Codex adapter mounts the Codex executable and config home into each case container. It derives the profile name from `CODEX_CONFIG` (`<profile>.config.toml`) and runs:

```text
codex exec --profile <profile> ... -C {workdir} < {prompt_file}
```

### OpenCode

Required environment variables:

```bash
export OPENCODE_BIN=/path/to/bin/opencode
export OPENCODE_CONFIG=/path/to/opencode-home/config/opencode/opencode.json
```

Run:

```bash
CONCURRENCY=5 bash run_infer.sh opencode opencode_default
```

The OpenCode adapter mounts the OpenCode executable, mounts the config home
read-only, and copies the data home into `/tmp/opencode-data` inside each case
container before running:

```text
opencode run --format json --auto < {prompt_file}
```

Example:

```bash
export OPENCODE_BIN=/path/to/opencode-runtime/bin/opencode
export OPENCODE_CONFIG=/path/to/opencode-home/config/opencode/opencode.json
```

### TraeCLI / TraeX

Required environment variables:

```bash
export TRAE_BIN=/path/to/bin/traex
export TRAE_CONFIG=/path/to/trae-home/my-profile.traecli.toml
```

Run:

```bash
CONCURRENCY=5 bash run_infer.sh traecli trae_my_profile
```

The TraeCLI adapter mounts the `traex` executable and the directory containing
`TRAE_CONFIG`. It derives the profile name from `TRAE_CONFIG`
(`<profile>.traecli.toml`) and copies the Trae home into `/tmp` inside each case
container before running:

```text
traex exec --profile <profile> ... < {prompt_file}
```

For example:

```bash
export TRAE_BIN=/path/to/traecli-runtime/bin/traex
export TRAE_CONFIG=/path/to/trae-home/gpt54-gggso.traecli.toml
```

## Step 2: Evaluate patches

After patch generation, run:

```bash
bash run_eval.sh <prefix>
```

Example:

```bash
bash run_eval.sh codex_my_profile
```

`run_eval.sh` will:

1. find the latest `agent_runs/*-<prefix>` directory;
2. convert `patches/*.patch` to PatchEval evaluation JSONL with `process_data.py`;
3. call `../evaluation/run_evaluation.py`.

The converted patch file is written to:

```text
eval_inputs/<prefix>.jsonl
```

Evaluation results are written by `run_evaluation.py` under:

```text
../evaluation/evaluation_output/results/<prefix>/
```

You can also pass the run directory explicitly:

```bash
bash run_eval.sh <prefix> /path/to/agent_runs/<timestamp>-<prefix>
```

## Runner behavior

`patch_agent_runner.py` is patch-generation-only. It does **not** run evaluation.

For each sample it:

1. reads the CVE image from the sample's `image_url` field;
2. starts a Docker container;
3. selects the repository workdir from `/workspace/<repo basename>`, then `/workspace/<repo basename lower-case>`, then `/workspace`;
4. hides non-repository files under `/workspace` when a repository subdirectory is selected;
5. writes a prompt to `/results/prompt.txt`;
6. runs the selected agent command;
7. collects `/workspace/fix.patch` or `git diff HEAD -U3`;
8. writes patches, logs, and `results.jsonl`.

## Output files

A generation run contains:

```text
agent_runs/<timestamp>-<prefix>/
├── patches/           # CVE-keyed patches for process_data.py
├── .work/             # prompt and agent stdout/stderr
├── results.jsonl
├── run_metadata.json
└── summary.json
```

## Notes

- `patch_url` metadata should not be used by agents as a repair source.
- During patch generation, the CLI prints case-level progress when a case
  finishes. Detailed agent logs are written under `.work/<case>/agent_stdout.txt`
  and `.work/<case>/agent_stderr.txt`.
- `run_infer.sh` exits with a non-zero status if any selected case fails, while
  preserving all generated patches and logs. Failed cases are represented by
  empty patch files and are counted as failed repairs during evaluation.
- Keep credentials and runtime homes outside version control.
