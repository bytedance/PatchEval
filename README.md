<div align="center">
  <img src="docs/figs/banner1.jpg" alt="Logo" width="400">
  <h3 align="center">A New Benchmark for Evaluating LLMs on Patching Real-World Vulnerabilities</h3>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2511.11019">
    <img src="https://img.shields.io/badge/Tech Report-arXiv-green">
  </a>
  <a href="https://huggingface.co/datasets/ByteDance/PatchEval">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-orange">
  </a>
  <a href="https://patcheval.github.io/">
    <img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-PatchEval-bule">
  </a>
  <a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-1f425f.svg?color=purple">
  </a>
  <a href="/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache 2.0-yellow">
  </a>
</p>

---

## 📢 News
* **[2025/11/18]** 🎉 We are excited to release PatchEval, a benchmark for evaluating Large Language Models (LLMs) on real-world vulnerability repair.

## 👋 Overview
PatchEval is a benchmark designed to systematically evaluate LLMs and Agents in the task of automated vulnerability repair.
It includes 1,000 vulnerabilities sourced from CVEs reported between 2015 and 2025, covering 65 CWE categories across Go, JavaScript, and Python.
A subset of 230 CVEs is paired with Dockerized sandbox environments that enable runtime patch validation through Proof-of-Concept (PoC) and unit testing.

<p align="center">
  <img src="docs/figs/overview.png" style="max-width: 60%; height: auto;"/>
</p>

## 💻 Getting Started
### Requirements
* **Operating System**: Linux (Tested on Ubuntu 20.04 and 18.04)
* **CPU**: ≥ 16 cores 
* **Disk Storage**: ≥ 500 GB of free storage.
> [!NOTE]
> PatchEval uses Docker as sandbox environments for patch evaluations and agent interactions. We recommend allocating at least 500 GB of disk space, as PatchEval includes 230 Docker containers, each consuming approximately 2 GB of storage.

### Setup
To get started, first install Docker by following the official [Docker Setup Guide](https://docs.docker.com/engine/install/).
Once Docker is ready, you can build PatchEval from source using the steps below:
```bash
git clone https://github.com/XXX
# # We recommend using Miniconda to manage Python environments, but you may use a native Python installation if preferred.
conda create -n patcheval python==3.12
conda activate patcheval
pip install -r requirements.txt
```

## 📜 Repo Structure

```
./
├── docs
├── patcheval
│   ├── evaluation    # Scripts for evaluation
│   ├── datasets      # Experimental datasets
│   ├── exp_agent     # Scripts for agent-based experiments
│   ├── exp_llm       # Scripts for LLM-based experiments
│   └── log           # Experimental Logs
├── scripts           # Scripts for running experiments and validating results
├── README.md
└── requirements.txt
```


## 📊 Evaluation
### Docker Images
PatchEval includes 230 Docker images, where each contains five key files in `/workspace` for patch validation:
```
llm.patch: The ground-truth (official) patch.
fix-run.sh: Run fix-run.sh to execute the PoC and verify that the patched vulnerability is no longer exploitable.
vul-run.sh: Run vul-run.sh to execute the PoC and verify that the original (unpatched) vulnerability is exploitable.
unit_test.sh: (if present) Run unit_test.sh to execute unit tests and validate functional correctness.
prepare.sh: Resets all changes in the repository. Run this script before each evaluation.
```

### Dataset
The vulnerability dataset is located in `patcheval/datasets`.
The file `input.json` contains CVE metadata required to run vulnerability repair experiments, such as `cve_id`, `cve_description`, `programming_language` and `vul_func`.

To download Docker images for patch validation, follow these steps:
```bash
cd scripts
python download_images.py
```

### Usage
- **Patch Generation**:
You can either follow the steps in [Reproducibility](#💫-reproducibility) or use your own approach to generate vulnerability patches.
After generating patches, convert them into the JSONL format as follows (e.g., `patcheval/evaluation/example_patch.json`) for validation:
  ```json
  [
    {
      "cve": "CVE_ID",
      "fix_patch": "YOUR_PATCH"
    }
  ]
  ```

* **Patch Validation**:
  ```bash
  python patcheval/evaluation/run_evaluation.py -h
  # Usage:
  # `--output`: Path to store experimental results.
  # `--patch_file`: Path to the patch file.
  # `--max_workers`: Maximum number of parallel workers for evaluation. Default is `4`.
  # `--log_level`: Logging level. Default is `INFO`.
  # `--artifact_eval`: Use this mode only when evaluating results in the patcheval/log/llm directory. Default is `False`.
  # Example:
  # cd patcheval/evaluation/
  # python run_evaluation.py \
  #   --output example \
  #   --patch_file ./example_patch.json \
  #   --log_level DEBUG
  ```


## 💫 Reproducibility
To ensure full reproducibility of the results reported in the paper and facilitate future research, we provide [evaluation logs](patcheval/log) of all experiments (including ablation studies)

* **Agentless-based Vulnerability Repair**
1.   Download CVE repositories
  ```bash
  cd patcheval/exp_llm/projects
  python clone.py
  ```

2. Config LLM API
  ```bash
  # patcheval/exp_llm/API-ENV.json
  {
    "model_name": {
      "api_key": "",         // your api key
      "api_url": "",         // base url
      "model": ""            // endpoint
    },
  }
```

3. Run experiments: You can start evaluation using the default configuration below.
For ablation studies, you can switch between different prompt templates under `patcheval/exp_llm/prompt_templates`.
  ```bash
  cd patcheval
  # running logs are saved in `patcheval/exp_llm/output/logs`
python -m exp_llm.main \
--epochs 1 \
--model your_model_name \
--template ./exp_llm/prompt_templates/Default.txt \
--input ./datasets/input.json \
--local_repo_path ./exp_llm/projects \
--max_workers 5
  ```

4. (Optional) Artifact evaluation: We provide pre-computed evaluation logs of all experiments in `patcheval/log/llm`. If you want to evaluate the artifact, you only need to run the following command:
  ```bash
  cd patcheval/evaluation
  python run_evaluation.py \
    --output artifact_eval_gemini2_5 \
    --patch_file ../log/llm/fixed_gemini2_5_Default.json \
    --artifact_eval
  ```

* **Agent-based Vulnerability Repair (SWEAgent with doubao-1.6 as an example)**
1. Install SWE-Agent
  ```bash
  # Setup Guide: https://swe-agent.com/latest/installation/source/
  git clone https://github.com/SWE-agent/SWE-agent.git
  cd SWE-agent/
  git checkout 8089c8baa55be1b12a61767e9b8e52bb63443b40 && patch -p1 -f < ../sweagent_diff.patch
  python -m pip install --upgrade pip && pip install --editable .
  ```
2. Configure your LLM using the template file `configs/template_without_feedback.yaml`
  ```bash
  cd ../
  # Use your own model name to create an evaluation template
  cp configs/template_without_feedback.yaml configs/template_doubao_without_feedback.yaml
  ```
  Edit the file `configs/template_doubao_without_feedback.yaml` to add your LLM endpoint, API key and base url
  ```yaml
  name: openai/endpoint (e.g., openai/ep-20251031xxxx)
  api_base: your_api_base_url
  api_key: your_api_key
  api_version: only used in AzureOpenAI e.g., `2024-03-01-preview`
  ```
  Register your model in `exp_agent/sweagent/SWE-agent/sweagent/run/run.py`
  ```python
  # Python
  litellm.register_model({
    # If you don’t know the exact model name (e.g., doubao-seed-1-6-251015), you can simply use a placeholder like doubao and run the patch generation script below — it will automatically display the correct model name in the error messages.
    "openai/doubao-seed-1-6-251015": {
      "max_tokens": 8192,
      "max_input_tokens": 128000,
      "max_output_tokens": 8192,
      "input_cost_per_token": 0,
      "output_cost_per_token": 0,
      "litellm_provider": "openai",
      "mode": "chat",
      "supports_function_calling": True,
      "supports_tool_choice": True
    }
  })
  ```
3. Run patch generation and validation
  ```bash
  # patch generation
  bash shells/run_exp1.sh doubao
  # patch evaluation
  bash shells/run_eval.sh doubao_exp1
  # results are saved in patcheval/exp_agent/sweagent/evaluation_output/results/doubao_exp1/summary.json
  ```

4. (Optional) Artifact evaluation: We provide pre-computed evaluation logs of all experiments in `patcheval/log/agent`. If you want to evaluate the artifact, you only need to run the following command:
  ```bash
  cd patcheval/evaluation
  python run_evaluation.py \
    --output artifact_eval_swe_agent_gemini \
    --patch_file ../log/agent/sweagent/gemini_exp1.jsonl \
  ```

For other agents, you can refer to the corresponding folder([ClaudeCode](patcheval/exp_agent/claudecode/README.md), [OpenHands](patcheval/exp_agent/openhands/README.md)) for more details.

## 🚀 Contributions
We would love to hear from the broader Security, Machine Learning, and Software Engineering communities!
Whether you report a bug, suggest an idea, or submit a pull request, just open an issue or PR — we’ll get back to you soon!

Contact us: [Jun ZENG](https://jun-zeng.github.io/), [Ming WEN](https://mingwen-cs.github.io/)

## 📖 Citation
If you find PatchEval useful for your research and applications, feel free to give us a star ⭐ or cite us using:
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
This project is licensed under the Apache License 2.0. 
See the [LICENSE](/LICENSE) file for more details.

## 📜 Acknowledgment
We would like to thank Caiyong Lin, Guangyu Zhou, Sen Cheng, Xufeng Zhou, Ke Sun, Jinhuang Liang, Zhongfu Su, Pengfei Sun, Zequn Fang, and Yongheng Yang at ByteDance for their dedicated efforts in reviewing the quality of the dataset. We thank Zhengqin Luo, Zhi Liu, Zach Zhang, and Yuan Zhang for their valuable feedback and advice. We also thank Shengqiang Li for helping artifact evaluation.