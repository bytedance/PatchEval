## 👋 Overview
PatchEval is a benchmark designed to systematically evaluate LLMs and Agents in the task of automated vulnerability repair.
It includes 1,000 vulnerabilities sourced from CVEs reported between 2015 and 2025, covering 65 CWE categories across Go, JavaScript, and Python.
A subset of 230 CVEs is paired with Dockerized sandbox environments that enable runtime patch validation through Proof-of-Concept (PoC) and unit testing.

## 📜 Data Instances Structure
Each vulnerability in the PatchEval dataset is a JSON object with the following structure:
```
cve_id: (str) - The unique CVE identifier from NVD (e.g., CVE-2024-42005).
cve_description: (str) - The official description of the CVE from NVD.
cwe_info: (dict) - A dictionary containing details about the associated Common Weakness Enumeration (CWE).
repo: (str) - The URL of the GitHub repository.
patch_url: (list[str]) - A list of URLs on GitHub.
programming_language: (str) - The primary programming language of the vulnerable code.
vul_func: (list[dict]) - A list of vulnerable code snippet.
fix_func: (list[dict]) - A list of fixed code snippet.
vul_patch: (str) - The patch diff of the CVE.
poc_test_cmd: (str) - The command to execute the Proof-of-Concept (PoC) test within the provided Docker environment. A null value indicates that no PoC environment is available.
unit_test_cmd: (str) - The command to execute the unit test within the provided Docker environment. A null value indicates that no unit test is available.
```

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