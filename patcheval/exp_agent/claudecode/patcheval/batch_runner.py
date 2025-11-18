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
import logging
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Optional, Set

from .dataset import load_dataset
from .single_runner import run_single_cve


def _auto_generate_readable_log(outputs_root: Path, problem_id: str):
    try:
        from .log_parser import StreamJsonLogParser
        
        log_file_path = outputs_root / "agent_logs" / f"{problem_id}.log"
        if not log_file_path.exists():
            return
        
        readable_logs_dir = outputs_root / "readable_logs"
        readable_logs_dir.mkdir(exist_ok=True)
        
        parser = StreamJsonLogParser()
        readable_log_path = readable_logs_dir / f"{problem_id}_readable.md"
        
        parser.generate_human_readable_log(str(log_file_path), str(readable_log_path))
        
    except ImportError:
        pass
    except Exception:
        pass


def _load_completed_ids(run_index_path: Path) -> Set[str]:
    completed_ids = set()
    
    if run_index_path.exists():
        try:
            with open(run_index_path, 'r', encoding='utf-8') as f:
                text = f.read()
                for line in text.replace("\\n", "\n").split("\n"):
                    if line.strip():
                        record = json.loads(line.strip())
                        if record.get("is_success", False):
                            completed_ids.add(record.get("problem_id", ""))
        except Exception as e:
            logging.warning(f"{e}")
    return completed_ids


def _update_run_index(run_index_path: Path, record, result: Dict[str, Any]):
    try:
        index_entry = {
            "problem_id": record.problem_id,
            "cve_id": record.cve_id,
            "is_success": result["is_success"],
            "agent_duration": result.get("agent_duration", 0),
            "total_duration": result.get("total_duration", 0),
            "stage": result.get("stage", "unknown"),
            "error_message": result.get("error_message", ""),
            "patch_stats": result.get("patch_stats", {}),
            "timestamp": time.time()
        }
        
        with open(run_index_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\\n")
            
    except Exception as e:
        logging.error(f"{e}")


def run_batch_cves(dataset_path: Path,
                  outputs_root: Path,
                  max_workers: int = 1,
                  timeout_seconds: int = 2700,
                  claude_timeout_seconds: int = 1800,
                  strategy: str = "iterative",
                  api_provider: str = "anthropic",
                  resume: bool = False,
                  limit: Optional[int] = None,
                  include_ids: Optional[Set[str]] = None,
                  exclude_ids: Optional[Set[str]] = None,
                  keep_containers: bool = False,
                  tool_limits: Optional[Dict[str, int]] = None,
                  max_total_tool_calls: Optional[int] = None,
                  max_cost_usd: float = 10.0,
                  enable_detailed_logging: bool = True,
                  save_process_logs: bool = False,
                  allow_git_diff_fallback: bool = False,
                  settings_file: Optional[str] = None,
                  port: str="8082") -> Dict[str, Any]:
    
    start_time = time.time()
    logger = logging.getLogger(__name__)
    
    outputs_root.mkdir(parents=True, exist_ok=True)
    (outputs_root / "patches").mkdir(exist_ok=True)
    (outputs_root / "agent_logs").mkdir(exist_ok=True)
    
    records = load_dataset(dataset_path, include_ids, exclude_ids, limit)
    if not records:
        return {"total": 0, "successful": 0, "failed": 0}
    
    
    run_index_path = outputs_root / "run_index.jsonl"
    if resume:
        completed_ids = _load_completed_ids(run_index_path)
        original_count = len(records)
        records = [r for r in records if r.problem_id not in completed_ids]
    
    if not records:
        return {"total": 0, "successful": 0, "failed": 0}
    
    results = []
    success_count = 0
    failed_count = 0
    
    semaphore = threading.Semaphore(max_workers)
    
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(
                run_single_cve,
                record=record,
                outputs_root=outputs_root,
                semaphore=semaphore,
                timeout_seconds=timeout_seconds,
                claude_timeout_seconds=claude_timeout_seconds,
                strategy=strategy,
                api_provider=api_provider,
                keep_container=keep_containers,
                tool_limits=tool_limits,
                max_total_tool_calls=max_total_tool_calls,
                max_cost_usd=max_cost_usd,
                enable_detailed_logging=enable_detailed_logging,
                save_process_logs=save_process_logs,
                allow_git_diff_fallback=allow_git_diff_fallback,
                settings_file=settings_file,
                port=port
            ): record for record in records
        }
        
        for future in as_completed(future_to_record):
            record = future_to_record[future]
            try:
                result = future.result()
                results.append(result)
                
                if result["is_success"]:
                    success_count += 1
                    status = "✅ success"
                else:
                    failed_count += 1
                    status = f"❌ fail ({result.get('stage', 'unknown')})"
                
                completed = len(results)
                total = len(records)
                duration = result.get("agent_duration", 0)
                total_duration = result.get("total_duration", 0)
                
                logger.info(f"{status} [{completed}/{total}] {record.problem_id} ({record.cve_id}) "
                           f"Agent: {duration:.1f}s, Total: {total_duration:.1f}s")
                
                if result["is_success"] and result.get("patch_stats"):
                    stats = result["patch_stats"]
                    logger.info(f"patch summury +{stats.get('additions', 0)}/-{stats.get('deletions', 0)} "
                               f"file: {stats.get('files_changed', 0)}")
                elif not result["is_success"]:
                    error = result.get("error_message", "Unknown error")[:100]
                    logger.warning(f"    fail: {error}")
                
                _update_run_index(run_index_path, record, result)
                
                if result["is_success"]:
                    try:
                        _auto_generate_readable_log(outputs_root, record.problem_id)
                    except Exception as log_error:
                        logger.warning(f"error: {record.problem_id}: {log_error}")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"error: {record.problem_id}: {e}")
                
                error_result = {
                    "problem_id": record.problem_id,
                    "cve_id": record.cve_id,
                    "is_success": False,
                    "error_message": f"error: {str(e)}",
                    "stage": "executor_exception",
                    "agent_duration": 0,
                    "total_duration": 0
                }
                _update_run_index(run_index_path, record, error_result)
    
    total_duration = time.time() - start_time
    success_rate = success_count / max(len(records), 1)
    
    summary = {
        "total_processed": len(records),
        "successful": success_count,
        "failed": failed_count,
        "success_rate": success_rate,
        "total_duration": total_duration,
        "average_duration": total_duration / max(len(records), 1),
        "strategy": strategy,
        "api_provider": api_provider,
        "max_workers": max_workers,
        "timeout_seconds": timeout_seconds,
        "claude_timeout_seconds": claude_timeout_seconds,
        "timestamp": time.time()
    }
    
    if results:
        agent_durations = [r.get("agent_duration", 0) for r in results if r.get("agent_duration")]
        if agent_durations:
            summary["agent_stats"] = {
                "avg_duration": sum(agent_durations) / len(agent_durations),
                "min_duration": min(agent_durations),
                "max_duration": max(agent_durations),
                "total_agent_time": sum(agent_durations)
            }
        
        successful_results = [r for r in results if r.get("is_success")]
        if successful_results:
            patch_stats = [r.get("patch_stats", {}) for r in successful_results]
            total_additions = sum(p.get("additions", 0) for p in patch_stats)
            total_deletions = sum(p.get("deletions", 0) for p in patch_stats)
            total_files = sum(p.get("files_changed", 0) for p in patch_stats)
            
            summary["patch_stats"] = {
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "total_files_changed": total_files,
                "avg_additions_per_cve": total_additions / len(successful_results),
                "avg_deletions_per_cve": total_deletions / len(successful_results),
                "avg_files_per_cve": total_files / len(successful_results)
            }
    
    summary_path = outputs_root / "summary.json"
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"saved: {summary_path}")
    except Exception as e:
        logger.error(f"error: {e}")
    
    _generate_detailed_report(outputs_root, results, summary)
    
    if summary.get("agent_stats"):
        agent_stats = summary["agent_stats"]
    
    return summary


def _generate_detailed_report(outputs_root: Path, results: list, summary: Dict[str, Any]):
    try:
        report_path = outputs_root / "detailed_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Claude CVE\\n\\n")
            
            f.write(f"- **strategy**: {summary['strategy']}\\n")
            f.write(f"- **API**: {summary['api_provider']}\\n")
            f.write(f"- **total_processed**: {summary['total_processed']}\\n")
            f.write(f"- **successful**: {summary['successful']}\\n")
            f.write(f"- **failed**: {summary['failed']}\\n")
            f.write(f"- **success_rate**: {summary['success_rate']:.1%}\\n")
            f.write(f"- **total_duration**: {summary['total_duration']:.1f}s\\n\\n")
            
            if summary.get("agent_stats"):
                stats = summary["agent_stats"]
                f.write("## Agent\\n\\n")
                f.write(f"- **avg_duration**: {stats['avg_duration']:.1f}s\\n")
                f.write(f"- *min_duration**: {stats['min_duration']:.1f}s\\n")
                f.write(f"- **max_duration**: {stats['max_duration']:.1f}s\\n")
                f.write(f"- **total_agent_time**: {stats['total_agent_time']:.1f}s\\n\\n")
            
            if summary.get("patch_stats"):
                stats = summary["patch_stats"]
                f.write(f"- **total_additions**: {stats['total_additions']}\\n")
            
            for result in results:
                status = "✅" if result.get("is_success") else "❌"
                cve_id = result.get("cve_id", "N/A")
                problem_id = result.get("problem_id", "N/A")
                agent_duration = result.get("agent_duration", 0)
                total_duration = result.get("total_duration", 0)
                error = result.get("error_message", "").replace("|", "\\|")[:50]
                
                f.write(f"| {cve_id} | {problem_id} | {status} | {agent_duration:.1f}s | {total_duration:.1f}s | {error} |\\n")
        
    except Exception as e:
        logging.error(f"error: {e}")