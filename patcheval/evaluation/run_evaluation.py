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

import secrets
import logging
import argparse
import os
import re
from collections import defaultdict
import json
import tempfile
from typing import Optional, Tuple
import docker
import tqdm
from logging.handlers import MemoryHandler
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class DockerManager:
    """A small facade around Docker Python SDK."""

    def __init__(self, logger: logging.Logger, cve: str):
        self.logger = logger
        self.cve = cve
        self.client = docker.from_env()

    def start_container(self, image_name: str, container_name: str, llm_patch: str) -> str | None:
        """
        Start a Docker container for the given image tag.
        If llm_patch is provided, create a temp file and mount it to /workspace/fix.patch in the container.
        """
        def _create_patch_file(llm_patch):
            fd, tmp_file_path = tempfile.mkstemp(suffix='.patch')
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
                tmp_file.write(llm_patch)
                if not llm_patch.endswith('\n'):
                    tmp_file.write('\n')
            return tmp_file_path
        
        tmp_file_path = _create_patch_file(llm_patch)
        volumes = {tmp_file_path: {'bind': '/workspace/fix.patch', 'mode': 'rw'}}
        try:
            self.client.containers.run(
                image_name,
                "/bin/bash",
                name=container_name,
                detach=True,
                tty=True,
                stdin_open=True,
                volumes=volumes
            )
            return container_name
        except Exception as e:
            self.logger.debug(f"Failed to start container: {e}", extra={"cve": self.cve})
            return None
        finally:
            # clean up temp file
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                tmp_file_path = None

    def rm_container(self, container_name: str) -> None:
        """Stop and remove the container if it exists. Also remove temp patch file if exists."""
        try:
            container = self.client.containers.get(container_name)
            container.stop()
            container.remove(force=True)
        except Exception as e:
            self.logger.debug(f"Failed to remove container: {e}", extra={"cve": self.cve})

    def exec_container(self, container_name: str, cmd: str, timeout: int = 600) -> Tuple[Optional[str], Optional[str]]:
        """Execute a command inside the container using bash -c."""
        try:
            shell_cmd = cmd
            if timeout and timeout > 0:
                shell_cmd = f"timeout {timeout} {cmd}"
            
            container = self.client.containers.get(container_name)
            exec_result = container.exec_run(f"/bin/bash -c \"{shell_cmd}\"", demux=True)
            stdout, stderr = exec_result.output if exec_result.output else (b'', b'')
            stdout = stdout.decode('utf-8', errors='replace') if stdout else ''
            stderr = stderr.decode('utf-8', errors='replace') if stderr else ''
            print_res = (
                f"-"*30 + " Standard Output " + "-"*30 + "\n"
                + f"{stdout if stdout else '<empty>'}\n"
            )
            if stderr:
                print_res += (
                    f"-"*30 + " Standard Error " + "-"*30 + "\n"
                    + f"{stderr if stderr else '<empty>'}\n"
                )
            if exec_result.exit_code == 124:
                timeout_msg = f"\n\n[TIMEOUT] Command execution exceeded the {timeout} second limit and was terminated."
                stderr += timeout_msg
                
            print_res += (
                f"-"*30 + " Finish Evaluation " + "-"*30 + "\n"
            )
            return exec_result.exit_code, print_res
        except Exception as e:
            error_msg = (
                f"\n⚠️ Command Execution Failed ⚠️\n"
                f"Container: {container_name}\n"
                f"Command: {cmd}\n"
                f"Error: {str(e)}\n"
                "----------------------"
            )
            return None, error_msg

class Evaluation:
    """Evaluation orchestrator using DockerManager."""

    def __init__(self, logger: logging.Logger, cve: str=""):
        self.logger = logger
        self.cve = cve
        self.docker_manager = DockerManager(self.logger, self.cve)

    def _run_script(self, container_name: str, script_name: str) -> Tuple[Optional[bool], Optional[str]]:
        """Run a script inside the container and return result."""
        exit_code, msg = self.docker_manager.exec_container(container_name, f"bash {script_name}")
        self.logger.debug(f"Run {script_name}: {msg}")
        if exit_code is None:
            return None, msg  
        return exit_code == 0, msg

    def _run_sh_cmd(self, container_name: str) -> Tuple[Optional[bool], Optional[str]]:
        """Run the PatchEval validation script from /workspace.

        PatchEval images are pre-normalized to the prepared baseline. Running
        prepare.sh again during evaluation can mutate lockfiles/module metadata
        and make the collected patch no longer apply to the evaluated tree.  The
        per-case validation entrypoint is always /workspace/fix-run.sh, so run it
        from /workspace regardless of the image's Docker WORKDIR.
        """
        exit_code, msg = self.docker_manager.exec_container(
            container_name,
            "bash -lc 'cd /workspace && bash fix-run.sh'",
        )
        self.logger.debug(f"Run fix-run.sh from /workspace: {msg}")
        if exit_code is None:
            return None, msg
        return exit_code == 0, msg

    def run_evaluation(
        self, cve: str, llm_patch: str, language: str, test_name: str, image_name: str
    ) -> Tuple[Optional[bool], Optional[str], Optional[str]]:
        """Execute evaluation: start container, prepare files, run PoC, clean up."""
        
        container_name = f"{test_name}_{cve.lower()}_tmp_{secrets.token_hex(4)}"

        if not self.docker_manager.start_container(image_name, container_name, llm_patch):
            fail_msg = f"Failed to start container {container_name}"
            self.logger.error(fail_msg)
            return None, fail_msg, None

        run_poc_result, run_poc_msg = self._run_sh_cmd(container_name)
        run_poc_msg = "="*30 + " Run PoC " + "="*30 + "\n" + run_poc_msg
        self.logger.debug(f"Successfully Evaluate")

        self.docker_manager.rm_container(container_name)
        self.logger.info(f"Finish eval and remove container {container_name}")

        if run_poc_result:
            errpr_type="Repair Success"
        elif run_poc_result==False and run_poc_msg is not None:
            errpr_type=self._error_type(run_poc_msg, language)
        else:
            errpr_type=None
            
        return run_poc_result, run_poc_msg, errpr_type


    def _error_type(self, error_log, language):
        if "patch does not apply" in error_log:
            return "apply_fail"

        if language == "Python":
            if "SyntaxError" in error_log or "IndentationError" in error_log:
                return "compilation_fail"
            else:
                return "validation_fail"

        if language == "JavaScript":
            if "SyntaxError" in error_log or "TypeError" in error_log:
                return "compilation_fail"
            else:
                return "validation_fail"

        if language == "Go":
            if re.search(r'^.*\.go:\d+:\d+: ', error_log, re.MULTILINE) and \
            not re.search(r'panic:', error_log):
                return 'compilation_fail'
            else:
                return "validation_fail"


def main():
    def _init():
        all_info = utils.read_json(args.input_file)
        cve2lang = {item["cve_id"]: item["programing_language"] for item in all_info}
        cve2image = {item["cve_id"]: item.get("image_url") for item in all_info if item.get("image_url")}
        missing = [item["cve_id"] for item in all_info if not item.get("image_url")]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
            raise ValueError(f"dataset samples missing image_url: {preview}{suffix}")
        return cve2lang, cve2image

    import utils
    if args.patch_file.endswith(".json"):
        patchs = utils.read_json(args.patch_file)
    else:
        patchs = utils.read_jsonl(args.patch_file)

    cve2lang, cve2image = _init()
    if args.log_level.upper() == "DEBUG":
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    main_logger = utils.get_logger(f"./evaluation_output/{args.output}/run_evaluation.log", log_level)
    main_logger.info(args, extra={'cve': 'SETUP'})
    def process_patch(patch):
        cve, fix_patch = patch['cve'], patch['fix_patch']
        
        task_logger_name = f"task-{cve}-{threading.get_ident()}"
        task_logger = logging.getLogger(task_logger_name)
        task_logger.setLevel(log_level)
        task_logger.propagate = False
        
        cve_filter = utils.CveContextFilter(cve_id=cve)
        task_logger.addFilter(cve_filter)
        
        main_file_handler = main_logger.handlers[0] 
        buffer_handler = MemoryHandler(capacity=1024, target=main_file_handler)
        
        task_logger.addHandler(buffer_handler)
        evaluation = Evaluation(logger=task_logger)
        
        language = patch.get("language") or cve2lang[cve]
        image_name = patch.get("image_url") or cve2image.get(cve)
        if not image_name:
            raise ValueError(f"missing image_url for {cve}")
        _, log_dir = utils.creat_patch_file(f"./evaluation_output/{args.output}/logs/{cve}", fix_patch)
        validation_type = None
        try:
            run_poc_result, run_poc_msg, validation_type = evaluation.run_evaluation(
                cve=cve, llm_patch=fix_patch, language=language, test_name="run_evaluation", image_name=image_name
            )
            output = (f"[PoC RESULT]: {run_poc_result}\n" + f"[PoC MSG]:\n{run_poc_msg}\n\n" +
                      f"[Validation TYPE]: {validation_type}")
            
            is_success = (validation_type == "Repair Success")

            if is_success:
                with open(f"{log_dir}/success_output.log", 'w') as f: f.write(output)
            else:
                with open(f"{log_dir}/error_output.log", 'w') as f: f.write(output)
            
            return (cve, language, validation_type, image_name, is_success, False)
        except Exception as e:
            task_logger.error(f"{image_name} RUN ERROR")
            task_logger.error(e)
            with open(f"{log_dir}/error_output.log", 'w') as f:
                f.write(str(e))
            return (cve, language, validation_type, image_name, False, True)  
        finally:
            buffer_handler.flush()
            buffer_handler.close()
            task_logger.removeHandler(buffer_handler)
            
    # Multi-threaded execution
    all_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_patch = {executor.submit(process_patch, patch): patch for patch in patchs}
        for future in tqdm.tqdm(as_completed(future_to_patch), total=len(patchs)):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                patch = future_to_patch[future]
                main_logger.error(f"Task for CVE {patch['cve']} failed unexpectedly: {e}", exc_info=True, extra={'cve': 'EXECUTOR'})

    success_summary = defaultdict(list)
    fail_summary = defaultdict(list)
    error_images = []

    for res in all_results:
        cve, language, validation_type, image_name, is_success, is_error = res
        
        if is_success:
            success_summary[language].append(cve)

        if not is_success:
            fail_summary[f"{language}_{validation_type}"].append(cve)

        if is_error:
            error_images.append(image_name)

    def generate_summary_report(title, success_summary, total_cases):
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"{title:^60}")
        report_lines.append("=" * 60)

        total_success = 0
        lang_breakdown = {}
        for lang, cves in success_summary.items():
            count = len(cves)
            total_success += count
            lang_breakdown[lang] = count
        
        pass_rate = (total_success / total_cases) * 100 if total_cases > 0 else 0
        
        report_lines.append(f"Total Cases Evaluated: {total_cases}")
        report_lines.append(f"Total Successful Repairs: {total_success}")
        report_lines.append(f"Overall Pass Rate: {pass_rate:.2f}%")
        report_lines.append("-" * 60)
        report_lines.append("Success Breakdown by Language:")
        if not lang_breakdown:
            report_lines.append("  None")
        else:
            for lang, count in sorted(lang_breakdown.items()):
                report_lines.append(f"  - {lang}: {count}")
        report_lines.append("=" * 60)
        
        json_data = {
            "title": title,
            "total_cases": total_cases,
            "total_success": total_success,
            "pass_rate": f"{pass_rate:.2f}%",
            "success_breakdown": lang_breakdown,
            "successful_cves": {lang: cves for lang, cves in success_summary.items()}
        }
        
        return "\n".join(report_lines), json_data

    total_patches = len(patchs)
    
    eval_report_str, eval_report_json = generate_summary_report(
        "PoC Evaluation Summary", success_summary, total_patches
    )

    final_report_str = eval_report_str
    
    fail_analysis = {key: len(cves) for key, cves in fail_summary.items()}
    fail_report_str = "\n" + "="*60 + f"\n{'Failure Analysis':^60}\n" + "="*60 + "\n"
    if not fail_analysis:
        fail_report_str += "No failures recorded.\n"
    else:
        for reason, count in sorted(fail_analysis.items()):
            fail_report_str += f"- {reason}: {count}\n"
    if error_images:
        fail_report_str += f"\nImages with execution errors: {len(error_images)}\n"
        for img in error_images:
            fail_report_str += f"  - {img}\n"
    
    final_report_str += fail_report_str

    main_logger.info("--- FINAL EVALUATION REPORT ---", extra={'cve': 'SUMMARY'})
    for line in final_report_str.split('\n'):
        main_logger.info(line, extra={'cve': 'SUMMARY'})
    print(final_report_str)

    with open(f"./evaluation_output/{args.output}/summary_report.txt", 'w') as f:
        f.write(final_report_str)

    full_json_summary = {
        "poc_evaluation": eval_report_json,
        "failure_analysis": {
            "breakdown": fail_analysis,
            "failed_cves": {key: cves for key, cves in fail_summary.items()}
        },
        "execution_errors": error_images
    }
    with open(f"./evaluation_output/{args.output}/summary.json", 'w') as f:
        json.dump(full_json_summary, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--patch_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=False, default="../datasets/patcheval_verified.json")
    parser.add_argument("--log_level", type=str, required=False, default="INFO")
    parser.add_argument("--max_workers", type=int, default=4, required=False, help="max workers")
    args = parser.parse_args()
    main()
