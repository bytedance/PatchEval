# PatchEval (Inspect AI)

PatchEval provides a Dockerized vulnerability-repair benchmark for evaluating whether agents can generate source-code patches for real vulnerabilities. Each sample packages a vulnerable repository, a vulnerability description, and an in-image runtime validator so that generated patches can be tested in a reproducible container environment.

## Overview

PatchEval (Inspect AI) is an Inspect AI evaluation package for the PatchEval verified Dockerized runtime-validation subset. The evaluation asks an agent to repair a vulnerable repository inside a per-CVE Docker image, submit a patch, and validate that patch in a clean evaluator container.

## Task

Task name:

```text
patcheval_verified
```

Run from a source checkout:

```bash
inspect eval src/patcheval_inspect/task.py@patcheval_verified \
  --model <provider/model> \
  --limit 1
```

Import from Python:

```python
from patcheval_inspect import patcheval_verified
```

## Dataset

The packaged dataset is:

```text
src/patcheval_inspect/data/patcheval_verified.json
```

It contains 230 verified PatchEval samples. Each record includes:

- `cve_id`: CVE identifier
- `description`: vulnerability description shown to the agent
- `workdir`: target repository path inside the Docker image
- `image_url`: per-sample Docker image reference

PatchEval images are hosted on GHCR, for example:

```text
ghcr.io/patcheval-cve/patcheval-cve:cve-2021-23376
```

## Evaluation flow

Each sample starts two Docker sandboxes from the same image:

- `default`: agent workspace for patch generation
- `evaluator`: clean workspace for patch validation

Before the agent runs, evaluator-only files outside the target repository are hidden from the `default` workspace. The agent should modify the target repository and write its final patch to:

```text
/workspace/fix.patch
```

The scorer collects the patch from `/workspace/fix.patch`. If that file is missing or empty, it falls back to:

```bash
cd <workdir> && git diff HEAD -U3
```

The collected patch is copied into the clean `evaluator` sandbox and validated with:

```bash
cd /workspace && bash fix-run.sh
```

A sample scores `C` if validation succeeds and `I` otherwise. Inspect accuracy reports the fraction of samples scored `C`.

Scoring is binary. If `fix-run.sh` exits with status code `0`, the sample receives `C`. Any non-zero exit code, timeout, missing patch, or sandbox error receives `I`.

## Install

```bash
python -m pip install -e .
```

From the repository root, run a smoke evaluation with:

```bash
inspect eval src/patcheval_inspect/task.py@patcheval_verified \
  --model <provider/model> \
  --limit 1
```
