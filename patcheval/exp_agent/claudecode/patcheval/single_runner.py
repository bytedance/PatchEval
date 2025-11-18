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
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .dataset import CVERecord
from .docker_utils import (
    pull_image_with_retry, 
    stop_container, run_work_container_no_mount
)
from .claude_runner_enhanced import ClaudeRunnerEnhanced
from .patch import write_patch_file, get_patch_stats, validate_patch


def run_single_cve(record: CVERecord,
                  outputs_root: Path,
                  semaphore: Optional[threading.Semaphore] = None,
                  timeout_seconds: int = 2700,
                  claude_timeout_seconds: int = 1800,
                  strategy: str = "iterative",
                  api_provider: str = "anthropic",
                  keep_container: bool = False,
                  tool_limits: Optional[Dict[str, int]] = None,
                  max_total_tool_calls: Optional[int] = None,
                  max_cost_usd: float = 10.0,
                  enable_detailed_logging: bool = True,
                  save_process_logs: bool = False,
                  allow_git_diff_fallback: bool = False,
                  settings_file: Optional[str] = None,
                  port: str="8082") -> Dict[str, Any]:
    
    if semaphore is None:
        semaphore = threading.Semaphore(1)
    
    problem_id = record.problem_id
    start_time = time.time()
    logger = logging.getLogger(__name__)
    
    result = {
        "problem_id": problem_id,
        "cve_id": record.cve_id,
        "is_success": False,
        "agent_duration": 0.0,
        "total_duration": 0.0,
        "container_id": "",
        "patch_stats": {},
        "error_message": "",
        "stage": "initialization",
        "strategy": strategy,
        "api_provider": api_provider
    }
    
    container_id = ""
    
    try:

        
        result["stage"] = "api_check"
        if api_provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable")
        elif api_provider == "bedrock":
            aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            bedrock_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            
            if not aws_region:
                raise RuntimeError("Missing AWS_REGION environment variable")
            if not (aws_access_key and aws_secret_key) and not bedrock_token:
                raise RuntimeError("Missing AWS credentials or AWS_BEARER_TOKEN_BEDROCK")
            api_key = bedrock_token or f"{aws_access_key}:{aws_secret_key}:{aws_region}"
        elif api_provider == "vertex":
            vertex_token = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("VERTEX_AUTH_TOKEN")
            vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT_ID")
            vertex_region = os.getenv("CLOUD_ML_REGION") or "us-central1"
            
            if not vertex_token:
                raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS or VERTEX_AUTH_TOKEN")
            if not vertex_project:
                raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT")
            api_key = f"{vertex_token}:{vertex_project}:{vertex_region}"
        else:
            raise RuntimeError(f"Unsupported API provider: {api_provider}")
        
        result["stage"] = "docker_setup"
        pull_image_with_retry(record.image_name, semaphore)
        
        modelname=os.getenv("MY_MODEL")
        result["stage"] = "work_container"
        container_id = run_work_container_no_mount(
            record.image_name, problem_id, semaphore, modelname)
        result["container_id"] = container_id
        
        
        claude = ClaudeRunnerEnhanced(
            container_id, 
            record.work_dir,
            tool_limits=tool_limits,
            max_total_tool_calls=max_total_tool_calls,
            max_cost_usd=max_cost_usd,
            enable_detailed_logging=enable_detailed_logging,
            allow_git_diff_fallback=allow_git_diff_fallback,
            settings_file=settings_file
        )
        
        if not claude.setup_environment(record, strategy, api_key, api_provider, port):
            pass
        
        result["stage"] = "claude_execution"
        
        claude_start = time.time()
        success, output_log, patch_content = claude.execute_cve_repair(
            strategy, claude_timeout_seconds)
        claude_duration = time.time() - claude_start
        
        result["agent_duration"] = claude_duration
        
        if not success:
            if not patch_content:
                patch_content = claude._extract_patch()
        
        result["stage"] = "patch_processing"
        
        if not patch_content or not patch_content.strip():
            if allow_git_diff_fallback:
            
                try:
                    import subprocess
                    git_diff = subprocess.run(
                        f"docker exec {container_id} bash -c 'cd {record.work_dir} && git diff'",
                        shell=True, capture_output=True, text=True
                    ).stdout
                    if git_diff.strip():
                        patch_content = git_diff
                       
                        result["patch_source"] = "git_diff_fallback"  
                    else:
                        pass
                except Exception as e:
                    pass
            else:
                pass   
        if not validate_patch(patch_content, relaxed=True):
            pass
        
        patch_stats = get_patch_stats(patch_content)
        result["patch_stats"] = patch_stats
        
        logger.info(f" {patch_stats}")
        
        result["stage"] = "output_writing"
        
        outputs_root.mkdir(parents=True, exist_ok=True)
        (outputs_root / "patches").mkdir(exist_ok=True)
        (outputs_root / "agent_logs").mkdir(exist_ok=True)
        
        patch_file_path = outputs_root / "patches" / f"{problem_id}.patch"
        write_patch_file(patch_content, patch_file_path)
        
        log_file_path = outputs_root / "agent_logs" / f"{problem_id}.log"
        
        container_logs = claude.get_container_logs()
        claude.set_success_and_finalize_log(True, patch_content, container_logs)
        
        full_log = {
            "problem_id": problem_id,
            "cve_id": record.cve_id,
            "strategy": strategy,
            "api_provider": api_provider,
            "duration": claude_duration,
            "patch_stats": patch_stats,
            "claude_output": output_log,
            "container_logs": container_logs
        }
        
        if enable_detailed_logging:
            full_log["detailed_process"] = claude.get_detailed_process_log()
        
        import json
        log_file_path.write_text(json.dumps(full_log, indent=2, ensure_ascii=False))
        
        try:
            from .log_parser import StreamJsonLogParser
            
            readable_logs_dir = outputs_root / "readable_logs"
            readable_logs_dir.mkdir(exist_ok=True)
            
            parser = StreamJsonLogParser()
            readable_log_path = readable_logs_dir / f"{problem_id}_readable.md"
            
            readable_content = parser.generate_human_readable_log(str(log_file_path), str(readable_log_path))
            
           
            
        except Exception as log_parse_error:
            pass
        
        if save_process_logs:
            process_log_path = outputs_root / "process_logs" / f"{problem_id}_process.json"
            process_log_path.parent.mkdir(exist_ok=True)
            claude.save_process_log(str(process_log_path))
        
        result["is_success"] = True
        
        if result.get("patch_source") == "git_diff_fallback":
            result["is_success"] = False  
            result["is_partial_success"] = True  
            result["warning"] = ""
           
        
        claude.cleanup()
        
        result["stage"] = "completed"
        result["total_duration"] = time.time() - start_time
        

    except Exception as e:
        result["error_message"] = str(e)
        result["is_success"] = False
        result["total_duration"] = time.time() - start_time
        logger.error(f"{result['stage']}: {e}")
        
        try:
            if container_id and "claude" in locals():
                container_logs = claude.get_container_logs() if 'claude' in locals() else ""
                claude.set_success_and_finalize_log(False, "", container_logs)
                
                log_file_path = outputs_root / "agent_logs" / f"{problem_id}_failed.log"
                outputs_root.mkdir(parents=True, exist_ok=True)
                (outputs_root / "agent_logs").mkdir(exist_ok=True)
                
                failed_log = {
                    "problem_id": problem_id,
                    "cve_id": record.cve_id,
                    "strategy": strategy,
                    "api_provider": api_provider,
                    "stage": result["stage"],
                    "error": str(e),
                    "container_logs": container_logs
                }
                
                import json
                log_file_path.write_text(json.dumps(failed_log, indent=2, ensure_ascii=False))
        except Exception as log_e:
            logger.warning(f"{log_e}")
        
    finally:
        if container_id:
            try:
                if not keep_container:
                    force_stop = hasattr(claude, 'execution_stopped') and claude.execution_stopped
                    stop_container(f"bench.{problem_id}.work", force=force_stop)
                else:
                    pass
            except Exception as cleanup_e:
                pass
    
    return result

    