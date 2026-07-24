<div align="center">
  <img src="docs/figs/banner1.jpg" alt="Logo" width="400">
  <h3 align="center">A New Benchmark for Evaluating LLMs / Agents on Patching Real-World Vulnerabilities</h3>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2511.11019">
    <img src="https://img.shields.io/badge/Tech Report-arXiv-green">
  </a>
  <a href="https://huggingface.co/datasets/ByteDance/PatchEval">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange">
  </a>
  <!-- <a href="https://patcheval.github.io/">
    <img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-PatchEval-blue">
  </a> -->
  <a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-1f425f.svg?color=purple">
  </a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache 2.0-yellow">
  </a>
</p>

---

## 📢 News

* **[2026/07/24]** We release **PatchEval-Verified**, with updated Docker evaluation environments for 230 CVEs. In the original PatchEval, some PoC tests were adapted from project regression tests and were tied too closely to particular patch implementations. We revised these tests to evaluate whether a vulnerability is fixed, rather than requiring a specific fix, improving robustness and reducing false negatives.
* **[2025/11/18]** PatchEval is released as a benchmark for evaluating Large Language Models and agents on real-world vulnerability repair.

## 👋 Overview

**PatchEval-Verified** evaluates LLMs and coding agents on automated repair of real-world vulnerabilities. This repository provides **230 CVE cases with Docker-based evaluation environments**, covering vulnerabilities reported between 2015 and 2025 across Go, JavaScript, and Python. Every image contains the vulnerable repository and a validation entrypoint.

The main improvement over the original PatchEval release is the upgraded evaluation harness:

* **Less implementation coupling:** a valid repair should not need to reproduce the official patch or one specific implementation strategy.
* **Stronger vulnerability validation:** the validation tests are designed to check whether the vulnerability remains exploitable after applying the generated patch.
* **More robust scoring:** strengthened test logic reduces accidental passes and rejection of semantically correct alternative fixes.

The repository also provides a streamlined two-stage workflow:

```text
1. Generate patches with a CLI coding agent
   patcheval/exp_agent/run_infer.sh

2. Evaluate generated patches in the verified Docker environments
   patcheval/exp_agent/run_eval.sh
   -> patcheval/evaluation/run_evaluation.py
```

The included agent adapters are:

```text
Codex CLI
OpenCode
TraeCLI
```

The figure below illustrates the overall design of the original PatchEval benchmark. This repository focuses on the 230 Docker-verified cases and the dynamic evaluation workflow.

<p align="center">
  <img src="docs/figs/overview.png" style="max-width: 60%; height: auto;"/>
</p>

## 💻 Getting Started

### Requirements

* **Operating System**: Linux.
* **Python**: Python 3.10+; Python 3.12 is recommended.
* **Docker**: Docker must be installed and available to the current user.
* **CPU**: 16 or more cores are recommended for parallel experiments.
* **Disk Storage**: Reserve at least 500 GB if pulling all 230 images locally.

### Setup

```bash
git clone https://github.com/bytedance/PatchEval.git
cd PatchEval

conda create -n patcheval python=3.12
conda activate patcheval
pip install -r requirements.txt
```

Verify both Docker and the Python Docker SDK:

```bash
docker version
python -c "import docker; print(docker.__version__)"
```

## 📜 Repository Structure

```text
./
├── docs/
├── patcheval/
│   ├── datasets/
│   │   └── patcheval_verified.json   # metadata and image URL for 230 verified cases
│   ├── evaluation/
│   │   ├── README.md
│   │   ├── run_evaluation.py         # Docker-based patch evaluator
│   │   ├── utils.py
│   │   └── example_patch.json
│   └── exp_agent/
│       ├── agents/                   # Codex, OpenCode, and TraeCLI adapters
│       ├── patch_agent_runner.py     # patch-generation-only runner
│       ├── process_data.py           # converts generated patches to evaluation JSONL
│       ├── run_infer.sh              # patch generation entrypoint
│       ├── run_eval.sh               # evaluation entrypoint
│       └── README.md
├── scripts/
│   ├── download_images.py
│   └── images.txt                    # list of the 230 Docker images
├── README.md
└── requirements.txt
```

## 📊 Verified Dataset and Docker Images

### Dataset

The metadata used by both patch generation and evaluation is:

```text
patcheval/datasets/patcheval_verified.json
```

It contains 230 entries with fields including:

```text
cve_id
cve_description
cwe_info
repo
patch_url
programing_language
vul_func
fix_func
image_url
```

> [!NOTE]
> `programing_language` is the field name used by the released dataset and is intentionally preserved for compatibility.

The agent runner uses `cve_description`, `repo`, and `image_url` to construct the task and runtime environment. Reference information such as `patch_url` and `fix_func` is not included in the agent prompt and should not be used as a repair source.

### Docker Images

`scripts/images.txt` contains the same 230 image references recorded by `image_url` in the verified dataset. Image names follow this form:

```text
ghcr.io/patcheval-cve/patcheval-cve:cve-<year>-<id>
```

Pull all images with:

```bash
cd scripts
python download_images.py
```

The script pulls images concurrently. Make sure the images required by your selected cases are available locally before evaluation.

## 🚀 CLI Agent Patch Generation and Evaluation

The verified workflow has been smoke-tested with the included `codex`, `opencode`, and `traecli` adapters. All adapters use the same two-step interface:

```bash
conda activate patcheval
cd patcheval/exp_agent
bash run_infer.sh <agent> <prefix>   # generate patches
bash run_eval.sh <prefix>            # evaluate generated patches
```

Start with one case before scaling to the complete dataset.

### Codex smoke test

```bash
conda activate patcheval
cd patcheval/exp_agent

export CODEX_BIN=/path/to/codex
export CODEX_CONFIG=/path/to/codex-home/<profile>.config.toml

LIMIT=1 CONCURRENCY=1 bash run_infer.sh codex codex_smoke
MAX_WORKERS=1 bash run_eval.sh codex_smoke
```

### Full 230-case run

```bash
conda activate patcheval
cd patcheval/exp_agent

CONCURRENCY=10 bash run_infer.sh codex codex_full
MAX_WORKERS=4 bash run_eval.sh codex_full
```

`CONCURRENCY` controls parallel patch-generation containers, while `MAX_WORKERS` controls parallel evaluation containers. Replace `codex` with `opencode` or `traecli` to use another adapter. See [patcheval/exp_agent/README.md](patcheval/exp_agent/README.md) for the required agent-specific environment variables.

## 🧪 Standalone Patch Evaluation

To evaluate patches generated by another system, prepare a JSON or JSONL file whose entries contain:

```json
{
  "cve": "CVE-2021-23376",
  "fix_patch": "diff --git ..."
}
```

Then run:

```bash
cd patcheval/evaluation
python run_evaluation.py \
  --output my_run \
  --patch_file /path/to/patches.jsonl \
  --input_file ../datasets/patcheval_verified.json \
  --max_workers 4 \
  --log_level INFO
```

For every patch, the evaluator:

1. starts the Docker image specified by the case's `image_url`;
2. mounts the submitted patch at `/workspace/fix.patch`;
3. runs `cd /workspace && bash fix-run.sh` with a 600-second command timeout;
4. classifies failures heuristically as patch-application, compilation, validation, or execution errors;
5. removes the temporary container.

A repair passes only when `fix-run.sh` exits successfully. See [patcheval/evaluation/README.md](patcheval/evaluation/README.md) for details.

`run_infer.sh` returns a non-zero exit code if any selected case fails, while preserving all generated patches and logs. Failed generation cases are represented by empty patch files and remain in the evaluation input, where they are counted as failed repairs.

## 📁 Outputs

A patch-generation run creates:

```text
patcheval/exp_agent/agent_runs/<timestamp>-<prefix>/
├── patches/           # one patch file per CVE
├── .work/             # prompts and agent stdout/stderr
├── results.jsonl      # per-case generation status
├── run_metadata.json
└── summary.json       # generated/failed counts
```

`run_eval.sh` converts these patches to:

```text
patcheval/exp_agent/eval_inputs/<prefix>.jsonl
```

Evaluation results are written to:

```text
patcheval/evaluation/evaluation_output/results/<prefix>/
├── run_evaluation.log
├── summary_report.txt
├── summary.json
└── logs/<CVE>/
    ├── fix.patch
    └── success_output.log or error_output.log
```

## 📈 Results on PatchEval-Verified

Coming soon...

## 🚀 Contributions

We welcome issues, bug reports, improvements to the verified validation environments, agent adapters, and reproducibility scripts.

## 📖 Citation

If you find PatchEval useful for your research and applications, please cite:

```bibtex
@misc{wei2025patcheval,
      title={PATCHEVAL: A New Benchmark for Evaluating LLMs on Patching Real-World Vulnerabilities},
      author={Zichao Wei and Jun Zeng and Ming Wen and Zeliang Yu and Kai Cheng and Yiding Zhu and Jingyi Guo and Shiqi Zhou and Le Yin and Xiaodong Su and Zhechao Ma},
      year={2025},
      eprint={2511.11019},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2511.11019},
}
```

## ✍️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
