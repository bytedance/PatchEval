# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import sys
import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List
from datetime import datetime

from exp_llm.helper.vul_fixer import VulFixer
from exp_llm.helper.logger import setup_logger
from exp_llm.helper.analysis_results import process_results_and_save, load_template, merge_results, read_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AVR Generate")

    # Input arguments (Required)
    parser.add_argument("-i", "--input", required=True, help="Input JSON file path")
    parser.add_argument("--local_repo_path", required=True, help="Folder of project repositories")

    # Output arguments
    parser.add_argument("-o", "--output", required=False, default=None, help="Output JSON file path")
    parser.add_argument("--log_file", required=False, default=None, help="Log file path")

    # LLM API-related parameters
    parser.add_argument("--model_name", help="Model name")

    # Run mode
    parser.add_argument(
        "--debug", type=lambda x: str(x).lower() == "true", default=False,
        help="Debug mode. Use --debug True to enable (default: False)"
    )

    # LLM Generation parameters
    parser.add_argument("--temperature", type=float, default=0, help="Temperature (0-1)")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens")
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count")
    parser.add_argument("--pass_k", type=int, default=1, help="Pass k times")
    parser.add_argument("--max_workers", type=int, default=4, help="Max worker threads")
    parser.add_argument("--timeout", type=int, default=600, help="API timeout seconds")

    # Prompt
    parser.add_argument("--template", help="Prompt template file path (default: ./exp_llm/prompt_templates/Default.txt)", default="./exp_llm/prompt_templates/Default.txt")
    return parser.parse_args()

def init():
    OUTPUT_RESULT_DIR = "./exp_llm/output/results/"
    OUTPUT_LOG_DIR = "./exp_llm/output/logs/"
    API_ENV_FILE = "exp_llm/API-ENV.json"

    args = parse_args()

    API_CONFIGS = read_json(API_ENV_FILE)
    if args.model_name not in API_CONFIGS:
        print(f"Unsupported model name: {args.model_name}")
        sys.exit(1)
    api_config = API_CONFIGS[args.model_name]
    args.api_key = api_config["api_key"]
    args.api_url = api_config["api_url"]
    args.model = api_config["model"]

    if not (0 <= args.temperature <= 1):
        print(f"Temperature must be between 0 and 1, got {args.temperature}")
        sys.exit(1)

    if not os.path.exists(args.template):
        print(f"Template file not found: {args.template}")
        sys.exit(1)
    args.prompt_template = load_template(args.template)

    os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)

    date_str = datetime.now().strftime('%Y_%m%d_%H%M')

    if not args.output:
        args.output = f"{OUTPUT_RESULT_DIR}/fix_{args.model}_{os.path.basename(args.template).split('.')[0]}_epoch_{args.epochs}_{date_str}.json"
    
    if not args.log_file:
        args.log_file = f"{OUTPUT_LOG_DIR}/fix_{args.model}_{os.path.basename(args.template).split('.')[0]}_epoch_{args.epochs}_{date_str}.log"
    return args

def main() -> None:
    args = init()
    logger = setup_logger(args.log_file)
    logger.info(f"args: {args}", extra={"cve": "GLOBAL"})
    
    fixer = VulFixer(args)

    cve_results_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    cve_logs_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    input_data=read_json(args.input)

    for result in fixer.process_vulnerability(input_data):
        match result["type"]:
            case "task_result":
                task_data = result["data"]
                task_log = task_data["log"]
                cve = task_log["cve"]
                cve_logs_cache[cve].append(task_log)

                task_result = task_data.get("result")
                if task_result:
                    patch_id = task_result["id"]
                    epoch = task_result["epoch"]
                    cwe_id = task_result["cwe_id"]
                    language = task_result["language"]
                    patch_url = task_result["patch_url"]
                    groundtruth = task_result["groundtruth"]
                    new_patch_entry = {
                        "id": patch_id,
                        "cve": cve,
                        "cwe": cwe_id,
                        "patch_url": patch_url,
                        "language": language,
                        "groundtruth": groundtruth,
                        "patch": task_result["patch"],
                        "import": task_result.get("import", []),
                        "epoch": epoch,
                        "poc_status": None,
                        "unittest_status": None,
                        "status": None,
                        "fix_code": None,
                        "test_msg": None,
                        "unittest_msg": None,
                        "error_type": None,
                        "diff_content": None,
                        "token_stat": task_result.get("token_stat", {}),
                    }
                    cve_results_cache[cve][patch_id].append(new_patch_entry)
                logger.info(
                    f"Vulnerability ID: {task_log['vul_id']} Patch generation status: {task_log['status']}"
                )

            case "cve_test_result":
                test_data = result["data"]
                cve = test_data["cve"]
                epoch = test_data["epoch"]
                cve_patches = cve_results_cache.get(cve, {})

                updated = False
                for patch_id, patch_entries in cve_patches.items():
                    for patch_entry in patch_entries:
                        if patch_entry.get("epoch") == epoch:
                            patch_entry.update(
                                {
                                    "poc_status": test_data["poc_status"],
                                    "unittest_status": test_data["unittest_status"],
                                    "status": test_data["status"],
                                    "fix_code": test_data["fix_code"],
                                    "test_msg": test_data["test_msg"],
                                    "unittest_msg": test_data["unittest_msg"],
                                    "error_type": test_data["error_type"],
                                    "diff_content": test_data["diff_content"],
                                    "token_stat": test_data["token_stat"],
                                }
                            )
                            updated = True
                            break
                    if updated:
                        break
                if not updated:
                    logger.warning(f"CVE {cve} (epoch: {epoch}) - Corresponding patch entry not found, skipping update.")

                all_patch_entries = [
                    entry for patch_list in cve_patches.values() for entry in patch_list
                ]
                current_cve_results = [{cve: all_patch_entries}]

                if os.path.exists(args.output):
                    existing_results = read_json(args.output)
                else:
                    existing_results = []
                merged_results = merge_results(existing_results, current_cve_results, flag="result")

                with open(args.output, "w") as f:
                    json.dump(merged_results, f, indent=2, ensure_ascii=False)

                logger.info(
                    f"✅ CVE {cve} (epoch: {epoch}) evaluation result: {test_data['status']} (saved)")

            case "summary":
                summary = result["data"]
                logger.info(f"⏱️ Total duration: {summary['total_duration']}s")
                process_results_and_save(args.output, args.input)


if __name__ == "__main__":
    main()
