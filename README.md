<div align="center">
  <img src="docs/figs/banner1.jpg" alt="Logo" width="400">
  <h3 align="center">A Benchmark and Workflow for Evaluating Coding Agents on Real-World Vulnerability Repair</h3>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2511.11019">
    <img src="https://img.shields.io/badge/Tech Report-arXiv-green">
  </a>
  <a href="https://huggingface.co/datasets/ByteDance/PatchEval">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange">
  </a>
  <a href="https://patcheval.github.io/">
    <img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-PatchEval-blue">
  </a>
  <a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-1f425f.svg?color=purple">
  </a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache 2.0-yellow">
  </a>
</p>

---

## 📢 News

* **[2025/11/18]** PatchEval is released as a benchmark for evaluating Large Language Models and agents on real-world vulnerability repair.

## 👋 Overview

PatchEval is a benchmark for evaluating automated vulnerability repair. It contains CVE-derived tasks across Go, JavaScript, and Python, and a 230-case subset with Dockerized sandbox environments for runtime validation.

This `SecPatch-verified` copy focuses on a streamlined two-stage workflow:

```text
1. Patch generation with a CLI coding agent
   patcheval/exp_agent/run_infer.sh

2. Patch evaluation
   patcheval/exp_agent/run_eval.sh
   -> patcheval/evaluation/run_evaluation.py
```

Currently supported CLI agents:

```text
Codex CLI
OpenCode
TraeCLI / TraeX
```

The generated patches are evaluated with the case validation script.

## 💻 Getting Started

### Requirements

* **Operating System**: Linux.
* **Python**: Python 3.10+; Python 3.12 is recommended.
* **Docker**: Docker must be installed and available to the current user.
* **Disk Storage**: The full 230-image PatchEval set is large. Reserve hundreds of GB if pulling all images locally.

### Setup

```bash
git clone <your-repo-url>
cd SecPatch-verified

conda create -n patcheval python=3.12
conda activate patcheval
pip install -r requirements.txt
```

Verify the Python Docker SDK is installed:

```bash
python -c "import docker; print(docker.__version__)"
```

## 📜 Repo Structure

```text
./
├── docs/
├── patcheval/
│   ├── datasets/
│   │   └── patcheval_230.json        # 230-case metadata used by generation/evaluation
│   ├── evaluation/
│   │   ├── run_evaluation.py         # patch evaluator
│   │   └── example_patch.json
│   └── exp_agent/
│       ├── agents/                   # agent adapters: codex/opencode/traecli
│       ├── patch_agent_runner.py     # patch-generation-only runner
│       ├── process_data.py           # converts generated patches to eval JSONL
│       ├── run_infer.sh              # unified patch generation entrypoint
│       ├── run_eval.sh               # unified evaluation entrypoint
│       └── README.md
├── scripts/
│   ├── download_images.py
│   └── images.txt                    # CVE -> Docker image list
├── README.md
└── requirements.txt
```

## 📊 Dataset and Docker Images

### Dataset

The verified 230-case metadata file is:

```text
patcheval/datasets/patcheval_230.json
```

Each entry keeps benchmark metadata such as:

```text
cve_id
cve_description
cwe_info
repo
patch_url
programing_language
vul_func
fix_func
```

The agent runner uses `cve_description` as the vulnerability description and derives runtime details from the Docker image and repository metadata.

### Docker Images

The image list is:

```text
scripts/images.txt
```

Each line is a Docker image reference for a PatchEval CVE case. Use a CVE-style image basename, for example:

```text
<registry>/<namespace>/cve-<year>-<id>:<tag>
```

To pull all images:

```bash
cd scripts
python download_images.py
```

## 🚀 CLI Agent Patch Generation and Evaluation

This verified workflow has been smoke-tested with all three supported CLI agents:

```text
codex
opencode
traecli
```

Each agent follows the same two-step interface:

```bash
conda activate patcheval
cd patcheval/exp_agent
bash run_infer.sh <agent> <prefix>   # generate patches
bash run_eval.sh <prefix>            # evaluate generated patches
```

Start with a one-case smoke test, then scale to all 230 cases.

### Codex smoke test

```bash
conda activate patcheval
cd patcheval/exp_agent

export CODEX_BIN=/path/to/codex
export CODEX_CONFIG=/path/to/codex-home/gpt54-gggso.config.toml

LIMIT=1 CONCURRENCY=1 bash run_infer.sh codex codex_smoke
bash run_eval.sh codex_smoke
```

### Full 230-case run

```bash
conda activate patcheval
cd patcheval/exp_agent

CONCURRENCY=10 bash run_infer.sh codex codex_full
bash run_eval.sh codex_full
```

Replace `codex` with `opencode` or `traecli` to run other tested adapters. See [patcheval/exp_agent/README.md](patcheval/exp_agent/README.md) for the required environment variables and detailed agent-specific examples.

## 🧪 Patch Evaluation

For normal usage, evaluate generated patches through the agent workflow:

```bash
conda activate patcheval
cd patcheval/exp_agent
bash run_eval.sh <prefix>
```

For standalone patch-file evaluation, see [patcheval/evaluation/README.md](patcheval/evaluation/README.md).

## 📁 Patch Generation Outputs

A generation run creates:

```text
patcheval/exp_agent/agent_runs/<timestamp>-<prefix>/
├── patches/           # CVE-keyed patches for evaluation
├── .work/             # prompt and agent stdout/stderr
├── results.jsonl
├── run_metadata.json
└── summary.json
```

During patch generation, the CLI prints case-level progress when each case finishes. Detailed logs are available in:

```text
.work/<case>/agent_stdout.txt
.work/<case>/agent_stderr.txt
```

## 🚀 Contributions

We welcome issues, bug reports, improvements to agent adapters, and additional reproducibility scripts.

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
