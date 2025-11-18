

# Running SWEagent on PatchEval

1. Install SWE-Agent
  ```bash
  # Setup Guide: https://swe-agent.com/latest/installation/source/
  git clone https://github.com/SWE-agent/SWE-agent.git
  cd SWE-agent/
  git checkout 8089c8baa55be1b12a61767e9b8e52bb63443b40 && patch -p1 -f < ../sweagent_diff.patch
  python -m pip install --upgrade pip && pip install --editable .
  ```
  
  (option) Replace the `_get_swerex_start_cmd` function in `deployment/docker.py` within the local `swerex` package with:
  ```python
    def _get_swerex_start_cmd(self, token: str) -> list[str]:
        rex_args = f"--auth-token {token}"
        pipx_install = "python3 -m pip install pipx && python3 -m pipx ensurepath"
        if self._config.python_standalone_dir:
            cmd = f"{self._config.python_standalone_dir}/python3.11/bin/{REMOTE_EXECUTABLE_NAME} {rex_args}"
        else:
            cmd = f"/root/.local/bin/{REMOTE_EXECUTABLE_NAME} {rex_args} || ({pipx_install} && pipx run {PACKAGE_NAME} {rex_args})"
        # cmd = f"/root/.local/bin/{REMOTE_EXECUTABLE_NAME} {rex_args}"
        # Need to wrap with /bin/sh -c to avoid having '&&' interpreted by the parent shell
        return [
            "/bin/bash",
            # "-l",
            "-c",
            cmd,
        ]
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
  # AzureOpenAI
  litellm.register_model({
    "azure/yourmodelname": {
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
  # results are saved in patcheval/exp_agent/sweagent/evaluation_output/results/doubao_exp1/summury.txt
  ```