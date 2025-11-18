# PatchEval: A New Benchmark for Evaluating LLMs on Patching Real-World Vulnerabilitie

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


## Submit to PatchEval Leaderboard

If you would like to submit your model results to [PatchEval](https://patcheval.github.io/) leaderboard, please follow the steps below.

1. Setup: Fork and clone the [PatchEval repository](https://github.com/PatchEval/experiments). Set up the evaluation environment as described in the repository’s [README.md](https://github.com/bytedance/PatchEval/blob/main/README.md).

2. Run evaluation: You can evaluate your own results using the provided script (e.g., `patcheval/evaluation/run_evaluation.py`).

    a. Prepare your patch file:

    ```
    [
        {
            "cve": "CVE-ID",
            "fix_patch": "YOUR_PATCH (diff)"
        },
    ]
    ```

    b. Run the evaluation command:

    ```
    cd patcheval/evaluation
    python run_evaluation.py \
        --output example \
        --patch_file YOUR_PATCH_PATH \
    ```

3. Submission: Your submission directory should contain the following files and structure:

    - `summary.json`: Evaluation results
    - `run_evaluation.log`: Evaluation logs
    - `your_patch.json`: Must include 'cve' and 'fix_patch' keys
    - `logs/`: Evaluation artifacts for each CVE. The folder structure should be as follows:
        ```
        logs
        ├── CVE-ID
        │   ├── fix.patch
        │   └── success_output.log
        └── CVE-ID
            ├── error_output.log
            └── fix.patch
        ```
    - `metadata.yaml`: Metadata of your submission, including how it will be displayed on the website:
        - name: The name of your leaderboard entry
        - site: URL/link to more information about your system
        - verified: false (See below for results verification)
    - `trajs/`: (Optional) Traces that show how your agent repair each CVE

4. Create a pull request to the [repository](https://github.com/PatchEval/experiments) with a new folder.

## Verify Your Results
A Verified ✅ badge on the leaderboard indicates that the PatchEval team has independently reproduced your patch generation results.
To request verification for your submission, please follow these steps:

1. Open an Issue: Create a new issue in [GitHub repository](https://github.com/PatchEval/experiments) to request verification.
2. Provide Instructions: In the issue, include detailed, step-by-step instructions for running your model on the PatchEval benchmark.
3. Await Confirmation: Once the team successfully reproduces your results, your submission will receive the “Verified” checkmark on the leaderboard.