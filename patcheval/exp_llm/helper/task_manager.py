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

import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple


@dataclass
class TaskSpec:
    """Specification for a unit of work to be executed by TaskManager.

    Attributes:
        key: Stable identifier for the task (e.g., CVE id). Used by the caller to correlate results.
        func: Callable to execute.
        args: Positional arguments for the callable.
        kwargs: Keyword arguments for the callable.
    """

    key: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Optional[dict] = None


@dataclass
class TaskOutcome:
    """Outcome for a task executed by TaskManager.

    Attributes:
        key: The same key provided in TaskSpec.
        result: The callable return value if succeeded; otherwise None.
        error: The exception raised during execution if failed; otherwise None.
    """

    key: str
    result: Any
    error: Optional[BaseException]


class TaskManager:
    """Manage task execution with optional multithreading.

    Default behavior preserves the current implementation by enabling a thread pool
    when `max_workers` > 1. If `max_workers` is 1 or None, tasks run sequentially.

    Multithreading is opt-in via constructor injection to avoid user-visible changes
    unless configured by the caller.
    """

    def __init__(self, local_repo_path, max_workers: Optional[int] = None, log_manager=None):
        self.local_repo_path = local_repo_path
        self.max_workers: int = int(max_workers or 1)
        self._lock = threading.Lock()
        self.log_manager = log_manager

    def run(self, specs: Iterable[TaskSpec]) -> Generator[TaskOutcome, None, None]:
        """Execute tasks and yield TaskOutcome as tasks complete.

        If `max_workers` > 1, uses ThreadPoolExecutor and yields outcomes in completion order
        (matching previous behavior based on `as_completed`). Otherwise runs sequentially.
        """
        specs_list: List[TaskSpec] = list(specs)
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(spec.func, *spec.args, **(spec.kwargs or {})): spec.key
                    for spec in specs_list
                }
                for future in as_completed(future_map):
                    key = future_map[future]
                    try:
                        result = future.result()
                        yield TaskOutcome(key=key, result=result, error=None)
                    except BaseException as exc:
                        yield TaskOutcome(key=key, result=None, error=exc)
        else:
            # Sequential execution preserves deterministic order
            for spec in specs_list:
                try:
                    result = spec.func(*spec.args, **(spec.kwargs or {}))
                    yield TaskOutcome(key=spec.key, result=result, error=None)
                except BaseException as exc:
                    yield TaskOutcome(key=spec.key, result=None, error=exc)

    # ===== Migrated task lifecycle helpers (behavior-preserving) =====
    def finalize_task_context(self, task_ctx: dict) -> None:
        """Finalize the task context by recording end timestamps and duration.

        Args:
            task_ctx: The task context dict containing a nested "log" dict with
                a "start_ts" key.
        """
        end_ts = time.time()
        task_ctx.setdefault("log", {}).update({
            "end_ts": end_ts,
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts)),
            "total_duration": round(end_ts - task_ctx.get("start_ts", end_ts), 2),
        })
        # Unbind current task logger context for this thread; mapping cleanup happens at CVE end
        try:
            if self.log_manager is not None:
                self.log_manager.unbind_current_task()
        except Exception:
            pass

    def init_repo_context(
        self,
        vul_entries: List[dict],
        cve: str,
        project: str,
        template_path: str,
        model_name: str,
        replacer_factory: Callable[[Any], Any],
    ) -> dict:
        """Initialize repository context for a CVE based on the first vulnerability entry.
        """
        first_entry = vul_entries[0]
        template_name = template_path.split("/")[-1].split(".txt")[0]
        test_name = f"{model_name.replace('/', '')}_{template_name}"
        first_entry["repo_path"] = os.path.join(self.local_repo_path, project)
        return {
            "replacer": replacer_factory(self.log_manager),
            "commit": first_entry["commit"],
            "repo": first_entry["repo_path"],
            "test_repo": first_entry["repo_path"].replace(
                "projects", f"projects_test/{test_name}/{cve}"
            ),
            "template_name": template_name,
            "test_name": test_name,
        }

    def init_task_context(
        self,
        cve: str,
        cve_knowledge: dict,
        vul_entry: dict,
    ) -> dict:
        """Initialize the task context payload identical to the previous helper.

        Args:
            cve: CVE identifier.
            cve_knowledge: CVE knowledge slice for the current CVE.
            vul_entry: Vulnerability entry dict.

        Returns:
            A dict representing task context with nested log fields.
        """
        start_ts = time.time()
        # Bind current task id to thread-local for contextual logging; mapping is created at CVE start
        try:
            if self.log_manager is not None:
                self.log_manager.bind_current_task(cve)
        except Exception:
            pass
        return {
            "cve": cve,
            "vul_id": vul_entry["id"],
            "cwe_id": cve_knowledge["cwe_id"],
            "patch_url": cve_knowledge["patch_url"],
            "language": cve_knowledge["programming_language"],
            "groundtruth": cve_knowledge["fix_func"],
            "start_ts": start_ts,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts)),
            "log": {
                "cve": cve,
                "vul_id": vul_entry["id"],
                "poc_status": "pending",
                "unittest_status": "pending",
                "status": "pending",
                "start_ts": start_ts,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts)),
                "vul_info": vul_entry.get("vul_info", [])[0] if vul_entry.get("vul_info") else {},
                "original_code": "",
                "fixed_code": "",
                "api_duration": 0,
                "total_duration": 0,
                "error": "",
            },
        }

    def create_success_result(
        self,
        task_ctx: dict,
        processed_code: str,
        api_time: float,
        epoch: int,
        cve_token_stats: dict,
    ) -> dict:
        """Create a success task_result identical to previous behavior."""
        task_ctx["log"].update({
            "status": "success",
            "fixed_code": processed_code,
            "api_duration": api_time,
        })
        return {
            "type": "task_result",
            "data": {
                "log": task_ctx["log"],
                "result": {
                    "id": task_ctx["vul_id"],
                    "cwe_id": task_ctx["cwe_id"],
                    "patch_url": task_ctx["patch_url"],
                    "language": task_ctx["language"],
                    "groundtruth": task_ctx["groundtruth"],
                    "cve": task_ctx["cve"],
                    "patch": [processed_code],
                    "epoch": epoch + 1,
                    "token_stat": cve_token_stats,
                },
            },
        }

    def create_skipped_result(self, task_ctx: dict, reason: str) -> dict:
        """Create a skipped task_result and finalize context."""
        task_ctx["log"].update({"status": "skipped", "error": reason})
        self.finalize_task_context(task_ctx)
        return {"type": "task_result", "data": {"log": task_ctx["log"], "result": None}}

    def create_api_fail_result(self, task_ctx: dict, api_time: float) -> dict:
        """Create an API failure task_result and finalize context."""
        task_ctx["log"].update({
            "status": "api_failure",
            "error": "LLM returned empty response",
            "api_duration": api_time,
        })
        self.finalize_task_context(task_ctx)
        return {"type": "task_result", "data": {"log": task_ctx["log"], "result": None}}

    def create_error_result(self, task_ctx: dict, error_msg: str) -> dict:
        """Create an error task_result and finalize context."""
        task_ctx["log"].update({"status": "error", "error": error_msg})
        self.finalize_task_context(task_ctx)
        return {"type": "task_result", "data": {"log": task_ctx["log"], "result": None}}

    def create_cve_test_result(
        self,
        cve: str,
        function_results: dict,
        epoch: int,
        test_result: Any,
        test_msg: str,
        unittest_res: Any,
        unittest_msg: str,
        error_type: str,
        diff: str,
        token: dict,
        fix_successful: bool,
    ) -> dict:
        """Create a cve_test_result payload.
        The caller is responsible for computing the fix_successful flag to avoid embedding logic within TaskManager.
        """
        return {
            "type": "cve_test_result",
            "data": {
                "cve": cve,
                "fix_code": function_results,
                "epoch": epoch + 1,
                "poc_status": test_result,
                "unittest_status": unittest_res,
                "status": "success" if fix_successful else "fail",
                "test_msg": test_msg,
                "unittest_msg": unittest_msg,
                "error_type": error_type,
                "diff_content": diff,
                "token_stat": token,
            },
        }

    def create_summary(self, stats: dict, total_tasks: int, global_start: float) -> dict:
        """Create a final summary identical to the original helper."""
        total_duration = round(time.time() - global_start, 2)
        success_rate = (stats["success"] / total_tasks * 100) if total_tasks else 0.0
        return {
            "type": "summary",
            "data": {
                "total_duration": total_duration,
                "total_tasks": total_tasks,
                "success_count": stats["success"],
                "success_rate": f"{success_rate:.1f}%",
                "api_failures": stats["api_failures"],
                "errors": stats["errors"],
                "total_cves": stats["total_cves"],
            },
        }
