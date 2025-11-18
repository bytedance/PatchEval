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
import subprocess
import logging
import time
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from collections import defaultdict
from .script_generator import ScriptGenerator
from .dataset import CVERecord
from .stream_monitor import RealTimeStreamMonitor, ProcessStreamReader, EnhancedProcessStreamReader


class ToolUsageLimiter:
    
    def __init__(self, max_calls_per_tool: Optional[Dict[str, int]] = None, max_total_calls: Optional[int] = None):
        self.max_calls = max_calls_per_tool or {}
        self.max_total_calls = max_total_calls
        self.current_calls = defaultdict(int)
        self.total_calls = 0
        self.logger = logging.getLogger(__name__)
    
    def check_and_increment(self, tool_name: str) -> bool:
        if self.max_total_calls is not None and self.total_calls >= self.max_total_calls:
            return False
        
        
        current = self.current_calls[tool_name]
        limit = self.max_calls.get(tool_name, float('inf'))
        
        if current >= limit:
            self.logger.warning(f"tool {tool_name}  {limit}")
            return False
            
        
        self.current_calls[tool_name] += 1
        self.total_calls += 1
        
        
        if self.max_total_calls is not None:
            self.logger.debug(f"tool call: {tool_name} ({self.current_calls[tool_name]}time), all: {self.total_calls}/{self.max_total_calls}")
        else:
            self.logger.debug(f"tool {tool_name} call times: {self.current_calls[tool_name]}/{limit}")
        
        return True
    
    def get_usage_stats(self) -> Dict[str, Any]:
        stats = {
            'per_tool': dict(self.current_calls),
            'total_calls': self.total_calls
        }
        
        if self.max_total_calls is not None:
            stats['total_limit'] = self.max_total_calls
            stats['total_remaining'] = self.max_total_calls - self.total_calls
            
        return stats


class CostController:
    
    def __init__(self, max_cost_usd: float = 10.0):
        self.max_cost = max_cost_usd
        self.current_cost = 0.0
        self.token_counts = {'input': 0, 'output': 0}
        self.logger = logging.getLogger(__name__)

        self.pricing = {
            'input': 3.0,   # $3 per 1M input tokens
            'output': 15.0  # $15 per 1M output tokens
        }
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def add_usage(self, input_text: str = "", output_text: str = "") -> bool:
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        input_cost = (input_tokens / 1_000_000) * self.pricing['input']
        output_cost = (output_tokens / 1_000_000) * self.pricing['output']
        new_cost = input_cost + output_cost
        
        if self.current_cost + new_cost > self.max_cost:
            self.logger.warning(f"over budget: ${self.current_cost + new_cost:.4f} > ${self.max_cost}")
            return False
            
        self.current_cost += new_cost
        self.token_counts['input'] += input_tokens
        self.token_counts['output'] += output_tokens
        
        self.logger.info(f"accumulated cost: ${self.current_cost:.4f}/${self.max_cost} (in: {input_tokens}, out: {output_tokens} tokens)")
        return True
    
    def get_cost_summary(self) -> Dict[str, Any]:
        return {
            'current_cost_usd': self.current_cost,
            'max_cost_usd': self.max_cost,
            'budget_remaining': self.max_cost - self.current_cost,
            'budget_used_percent': (self.current_cost / self.max_cost) * 100,
            'total_tokens': sum(self.token_counts.values()),
            'input_tokens': self.token_counts['input'],
            'output_tokens': self.token_counts['output']
        }


class ClaudeRunnerEnhanced:
    
    def __init__(self, container_id: str, work_dir: str, 
                 tool_limits: Optional[Dict[str, int]] = None,
                 max_total_tool_calls: Optional[int] = None,
                 max_cost_usd: float = 10.0,
                 enable_detailed_logging: bool = True,
                 allow_git_diff_fallback: bool = False,
                 settings_file: Optional[str] = None):
        self.container_id = container_id
        self.work_dir = work_dir
        self.allow_git_diff_fallback = allow_git_diff_fallback
        self.settings_file = settings_file
        self.logger = logging.getLogger(__name__)
        
        self.stream_monitor = RealTimeStreamMonitor(
            tool_limits=tool_limits,
            max_total_tool_calls=max_total_tool_calls,
            max_cost_usd=max_cost_usd
        )
        
        self.tool_limiter = ToolUsageLimiter(
            max_calls_per_tool=tool_limits,
            max_total_calls=max_total_tool_calls
        )
        self.cost_controller = CostController(max_cost_usd)
        
        self.enable_detailed_logging = enable_detailed_logging
        
        self.process_log = []
        self.start_time = None
        self.end_time = None
        
        self.execution_stopped = False
        self.stop_reason = ""
        
        self.output_buffer = []
        self.temp_log_file = None
        self.cve_id = None
    def setup_environment(self, record: CVERecord, strategy: str, 
                         api_key: str, api_provider: str, port: str) -> bool:
        try:
            self.cve_id = record.cve_id  
            
            install_script_path = "templates/claude-code-install.sh"
            
            with open(install_script_path, 'r', encoding='utf-8') as f:
                install_script = f.read()
            install_script = install_script.replace(
                "/workspace/markdown-it", 
                self.work_dir
            )
            
            host_base_url = os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com')
            host_api_key = os.getenv('ANTHROPIC_API_KEY', '')
            
            install_script = install_script.replace("{{ANTHROPIC_BASE_URL}}", host_base_url)
            install_script = install_script.replace("{{ANTHROPIC_API_KEY}}", host_api_key)
            install_script = install_script.replace("{{ANTHROPIC_AUTH_TOKEN}}", host_api_key)
            install_script = install_script.replace("$PORT$", port)
            
            self._write_file_to_container("/tmp/install_claude.sh", install_script)
            self._exec_in_container("chmod", "+x /tmp/install_claude.sh")
            
            self._log_process_step("claude_install", "install Claude Code")
            
            
            try:
                install_cmd = f"bash /tmp/install_claude.sh >/tmp/install.log 2>&1 && echo 'INSTALL_SUCCESS' || echo 'INSTALL_FAILED'"
                install_status = self._exec_in_container_with_output("bash", f"-c '{install_cmd}'").strip()
                
                if "INSTALL_SUCCESS" in install_status:
                    self.logger.info("Claude Code install success")
                    self._log_process_step("claude_install_success", "Claude Code install success")
                else:
                    self.logger.warning("Claude install warning")
                    self._log_process_step("claude_install_warning", "Claude install warning")
            except Exception as e:
                self.logger.warning(f"Claude install error: {e}")
                self._log_process_step("claude_install_error", f"Claude install error: {e}")
            ignore_content = f'echo -e "\n\n*.png\n*.jpg\n*.jpeg\n*.gif\n*.bmp\n*.tiff\n*.webp\n*.mp3\n*.mp4\n*.avi\n*.mov\n*.flv\n*.wmv\n*.pdf\n*.psd\n*.ai\n\n\n*.zip\n*.tar\n*.tar.gz\n*.tar.bz2\n*.7z\n*.rar\n*.gz\n*.bz2\n\n\n*.exe\n*.dll\n*.so\n*.dylib\n*.bin\n*.out\n\n*.db\n*.sqlite\n*.sqlite3\n\n/build/\n/dist/\n/bin/\n/out/\n\n\n.DS_Store\nThumbs.db\n\n# Go\nmyapp\nvendor/\n*.out\n*.test\ncoverage.out\nbuild/\ndist/\n\n# JavaScript/Node.js\nnode_modules/\ndist/\nbuild/\nout/\ndist-ssr/\n*.bundle.js\n*.bundle.js.map\n*.chunk.js\n*.chunk.js.map\nnpm-debug.log*\nyarn-debug.log*\nyarn-error.log*\n.pnpm-debug.log*\n.env.local\n.env.development.local\n.env.test.local\n.env.production.local\n.node-gyp/\n*.node\n\n# Python\n__pycache__/\n*.py[cod]\n*$py.class\nvenv/\nenv/\nENV/\n*.venv\n*.egg-info/\n.installed.cfg\n*.egg\ndist/\nbuild/\nwheelhouse/\n*.so\n*.pyd\n*.dll\n.coverage\nhtmlcov/\n.pytest_cache/\n\n*.blk\n*.idx\n*.jar\n*.md\n*package-lock.json\n\n\n" >> {self.work_dir}/.gitignore'.replace('\n', '\\n')
            self._exec_in_container('find /workspace/ -maxdepth 1 -type f -name "*.patch" -exec rm -v {} +')
            self._exec_in_container("bash", f"-c '{ignore_content}'")
            
            try:
                
                self.logger.info("check git safe directory")
                self._exec_in_container("git", f"config --global --add safe.directory {self.work_dir}")
                self._log_process_step("git_safe_directory", f"add git safe directory: {self.work_dir}")
                
                
                git_check = self._exec_in_container_with_output("bash", 
                    f"-c 'cd {self.work_dir} && git status'")
                self.logger.info("Git repo exists")
                self._log_process_step("git_check", "Git repo exists")
                
                
                try:
                    
                    uncommitted_files = self._exec_in_container_with_output("bash", 
                        f"-c 'cd {self.work_dir} && git status --porcelain'")
                    if uncommitted_files.strip():
                        
     
                        self._exec_in_container("bash", 
                            f"-c 'cd {self.work_dir} && git config user.email \"cve-repair@example.com\" && git config user.name \"CVE Repair Baseline\"'")
                        

                        self._exec_in_container("bash", 
                            f"-c 'cd {self.work_dir} && git add . && git commit --no-verify -m \"Baseline commit before CVE repair - contains existing untracked/modified files\"'")
                        
                    else:
                        pass
                        
                except Exception as baseline_error:
                    pass
                    
                    
            except Exception as git_error:
                
                self.logger.info(f"Git error({git_error})")
                self._log_process_step("git_init", "begin init git repo")
                
                
                try:
                    self._exec_in_container("git", f"config --global --add safe.directory {self.work_dir}")
                except:
                    pass  
                    
                self._exec_in_container("bash", 
                    f"-c 'cd {self.work_dir} && git init && git add . && git config user.email \"cve-repair@example.com\" && git config user.name \"CVE Repair\" && git commit --no-verify -m \"Initial commit\" || true'")
            
            
            templates_dir = Path("templates")
            script_gen = ScriptGenerator(templates_dir)
            fix_command = script_gen.generate_cve_fix_command(record, strategy)
            command_file = f"{self.work_dir}/.claude/commands/{strategy}.md"
            
            self._write_file_to_container(command_file, fix_command)
            
            self._log_process_step("command_generation", f"generate file: {strategy}.md")
            
            settings_content = ScriptGenerator.generate_settings_file()
            settings_file = f"{self.work_dir}/.claude/settings.json"
            self._write_file_to_container(settings_file, settings_content)
            
            if self.settings_file:
                try:
                    source_settings_path = Path("config") / self.settings_file
                    if not source_settings_path.exists():
                        self.logger.warning(f"file not exist: {source_settings_path}")
                    else:
                        with open(source_settings_path, 'r', encoding='utf-8') as f:
                            custom_settings_content = f.read()
                        
                        container_settings_path = f"{self.work_dir}/.claude/{self.settings_file}"
                        self._write_file_to_container(container_settings_path, custom_settings_content)
                        self.logger.info(f" {self.settings_file} -> {container_settings_path}")
                        self._log_process_step("settings_copy", f"{self.settings_file}")
                except Exception as e:
                    self.logger.error(f"copy file error: {e}")
                    self._log_process_step("settings_copy_error", f"copy file error: {str(e)}")
            
            self._exec_in_container("chown", f"-R claude_user:claude_user {self.work_dir}")
        
            return True
            
        except Exception as e:
            return False
    
    def execute_cve_repair(self, strategy: str = "iterative", 
                          timeout: int = 1800) -> Tuple[bool, str, str]:
        try:
            self.start_time = time.time()
            
            
            cmd = self._build_claude_command(strategy)
            
            
            start_time = time.time()
            
            
            def on_limit_reached(reason: str):
                self.execution_stopped = True
                self.stop_reason = reason
                            
            self.stream_monitor.add_stop_callback(on_limit_reached)
            
            
            env_vars = []
            if self.enable_detailed_logging:
                env_vars.extend([
                    "ANTHROPIC_LOG=debug",
                    "CLAUDE_CODE_ENABLE_TELEMETRY=1"
                ])
            
           
            env_prefix = " ".join(env_vars) + " " if env_vars else ""
            switch_user_cmd = f"su - claude_user -c 'cd {self.work_dir}  && {env_prefix}{cmd}'"
            
            
            try:
                result = self._exec_with_real_time_monitoring(switch_user_cmd, timeout)
                
                self._update_real_time_log(status="processing")
                
                if self.execution_stopped:
                    self._log_process_step("command_stopped", f"stop execution: {self.stop_reason}")
                    self._update_real_time_log(status="stopped")
                    success = False
                    result = f"stop execution: {self.stop_reason}\n\n{result}"
                else:
                    success = True
                    
            except subprocess.TimeoutExpired:
                self._update_real_time_log(status="timeout")
                result = f"Claude execution timeout after {timeout} seconds"
                success = False
            except Exception as e:
                self._log_process_step("command_error", f"execution error: {str(e)}")
                self._update_real_time_log(status="error")
                result = str(e)
                success = False
                
            duration = time.time() - start_time
            self.end_time = time.time()
            self.logger.info(f"execution sucess: {duration:.1f}s")
            
            self._analyze_real_time_results()
            
            if not self.execution_stopped:
                success = success and self._check_repair_success(result)
                
            self._log_process_step("repair_result", f"fix{'success' if success else 'fail'}")
            
            patch_content = self._extract_patch() if success else ""
            
            self._log_final_stats()
            
            final_duration = time.time() - start_time
            self._update_real_time_log(duration=final_duration, status="analyzing_results")
            
            return success, result, patch_content
            
        except Exception as e:
            self.end_time = time.time()
            self._log_final_stats()
            
            if self.start_time:
                final_duration = time.time() - self.start_time
                self._update_real_time_log(duration=final_duration, status="failed")
            
            return False, str(e), ""
    
    def _build_claude_command(self, strategy: str) -> str:
        command_name = strategy  
        
        claude_cmd_parts = [
            f"claude /{command_name}",
            "--print",  
            "--dangerously-skip-permissions",  
            "--permission-mode bypassPermissions",  
        ]
        
        
        if self.enable_detailed_logging:
            claude_cmd_parts.extend([
                "--output-format stream-json",  
                "--verbose"  
            ])
        else:
            claude_cmd_parts.append("--output-format json")  
        
        
        if self.settings_file:
            claude_cmd_parts.extend([
                "--settings", f".claude/{self.settings_file}"
            ])
        
        
        if self.tool_limiter.max_calls:
            self.logger.debug(f"tool limit setting: {self.tool_limiter.max_calls}")
        
        claude_command = " ".join(claude_cmd_parts)
        self._log_process_step("command_build", f"build command: {claude_command}")
        
        return claude_command
    
    def _log_process_step(self, step_type: str, message: str) -> None:
        timestamp = time.time()
        elapsed_time = timestamp - (self.start_time or timestamp) if self.start_time else 0
        step = {
            'timestamp': timestamp,
            'step_type': step_type,
            'message': message,
            'elapsed': elapsed_time
        }
        self.process_log.append(step)
        
        if self.enable_detailed_logging:
            if elapsed_time > 0.01:  
                self.logger.info(f"[{step_type.upper()}] {message} (time: {elapsed_time:.2f}s)")
            else:
                self.logger.info(f"[{step_type.upper()}] {message}")
    
    def _init_real_time_log(self, record: CVERecord, strategy: str, api_provider: str) -> None:
        try:
            outputs_root = Path("./outputs")
            outputs_root.mkdir(parents=True, exist_ok=True)
            (outputs_root / "agent_logs").mkdir(exist_ok=True)
            
            self.temp_log_file = outputs_root / "agent_logs" / f"{record.problem_id}_temp.log"
            
            initial_log = {
                "problem_id": record.problem_id,
                "cve_id": record.cve_id,
                "strategy": strategy,
                "api_provider": api_provider,
                "duration": 0,
                "patch_stats": {},
                "claude_output": "",
                "container_logs": "",
                "detailed_process": None,
                "status": "running",
                "last_updated": time.time()
            }
            
            self.temp_log_file.write_text(json.dumps(initial_log, indent=2, ensure_ascii=False))
            
        except Exception as e:
            self.temp_log_file = None
    
    def _update_real_time_log(self, new_output_chunk: str = None, status: str = None, 
                             patch_stats: dict = None, duration: float = None) -> None:
        if not self.temp_log_file or not self.temp_log_file.exists():
            return
            
        try:
            current_log = json.loads(self.temp_log_file.read_text(encoding='utf-8'))
            
            if new_output_chunk:
                current_log["claude_output"] += new_output_chunk
                
            if status:
                current_log["status"] = status
                
            if patch_stats:
                current_log["patch_stats"] = patch_stats
                
            if duration is not None:
                current_log["duration"] = duration
                
            current_log["last_updated"] = time.time()
            
            if self.enable_detailed_logging and self.process_log:
                current_log["detailed_process"] = {
                    'process_steps': self.process_log,
                    'total_duration': (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0,
                    'tool_usage_stats': self.tool_limiter.get_usage_stats(),
                    'cost_summary': self.cost_controller.get_cost_summary()
                }
            
            self.temp_log_file.write_text(json.dumps(current_log, indent=2, ensure_ascii=False))
            
        except Exception as e:
            self.logger.warning(f"fail: {e}")
    
    def _finalize_real_time_log(self, success: bool, patch_content: str = "", 
                               container_logs: str = "") -> None:
        
        if not self.temp_log_file or not self.temp_log_file.exists():
            return
            
        try:
           
            current_log = json.loads(self.temp_log_file.read_text(encoding='utf-8'))
            
            
            current_log["status"] = "completed" if success else "failed"
            current_log["container_logs"] = container_logs
            
           
            if success and patch_content:
                from .patch import get_patch_stats
                current_log["patch_stats"] = get_patch_stats(patch_content)
            
            
            if self.start_time:
                current_log["duration"] = (self.end_time or time.time()) - self.start_time
            current_log["last_updated"] = time.time()
            
            
            final_log_file = self.temp_log_file.parent / f"{current_log['problem_id']}.log"
            if success:
                final_log_file.write_text(json.dumps(current_log, indent=2, ensure_ascii=False))
            else:
                
                failed_log_file = self.temp_log_file.parent / f"{current_log['problem_id']}_failed.log"
                failed_log_file.write_text(json.dumps(current_log, indent=2, ensure_ascii=False))
            
            
            self.temp_log_file.unlink()
            self.temp_log_file = None
            
            self.logger.info(f"success: {final_log_file if success else failed_log_file}")
            
        except Exception as e:
            self.logger.warning(f"fail: {e}")
    
    def get_current_output(self) -> str:
        
        if self.temp_log_file and self.temp_log_file.exists():
            try:
                current_log = json.loads(self.temp_log_file.read_text(encoding='utf-8'))
                return current_log.get("claude_output", "")
            except:
                pass
        return "".join(self.output_buffer)
    
    def _analyze_claude_output(self, output: str) -> None:
        
        try:
            
            self.cost_controller.add_usage(output_text=output)
            
            
            if "stream-json" in output or "{" in output:
                lines = output.split("\n")
                for line in lines:
                    if line.strip() and line.strip().startswith("{"):
                        try:
                            data = json.loads(line.strip())
                            if (data.get("type") == "assistant" and 
                                "message" in data and 
                                "content" in data["message"]):
                                content = data["message"]["content"]
                                if isinstance(content, list):
                                    for item in content:
                                        if (isinstance(item, dict) and 
                                            item.get("type") == "tool_use" and 
                                            "name" in item):
                                            tool_name = item["name"]
                                            if not self.tool_limiter.check_and_increment(tool_name):
                                                self.logger.warning(f"tool {tool_name} limit")
                        except json.JSONDecodeError:
                            continue
            
            
        except Exception as e:
            self.logger.warning(f"fail: {e}")
    
    def _log_final_stats(self) -> None:
        
        if not self.start_time:
            return
            
        total_duration = (self.end_time or time.time()) - self.start_time
        
        
        real_time_stats = self.stream_monitor.get_statistics()
        
        
        self._log_process_step("final_stats", f"time: {total_duration:.2f}s")
        
        
        if real_time_stats['tool_calls']:
            per_tool_str = ', '.join(f"{tool}:{count}" for tool, count in real_time_stats['tool_calls'].items())
        
        if real_time_stats.get('max_total_calls'):
            remaining = real_time_stats['max_total_calls'] - real_time_stats['total_tool_calls']
        
            
    def _generate_readable_log(self, claude_output: str) -> None:
        try:
            from .log_parser import StreamJsonLogParser
            
            temp_log_path = f"/tmp/claude_stream_{int(time.time())}.json"
            with open(temp_log_path, 'w', encoding='utf-8') as f:
                f.write(claude_output)
            
            parser = StreamJsonLogParser()
            
            readable_log = parser.generate_human_readable_log(temp_log_path)
            
            if hasattr(self, 'cve_id'):
                readable_log_path = f"./outputs/readable_logs/{self.cve_id}_readable.md"
                os.makedirs(os.path.dirname(readable_log_path), exist_ok=True)
                
                with open(readable_log_path, 'w', encoding='utf-8') as f:
                    f.write(readable_log)
                
            
            try:
                os.remove(temp_log_path)
            except:
                pass
                
        except Exception as e:
            self.logger.warning(f"fail: {e}")
    
    def get_detailed_process_log(self) -> Dict[str, Any]:
        return {
            'process_steps': self.process_log,
            'total_duration': (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0,
            'tool_usage_stats': self.tool_limiter.get_usage_stats(),
            'cost_summary': self.cost_controller.get_cost_summary()
        }
    
    def save_process_log(self, output_file: str) -> None:
        try:
            log_data = self.get_detailed_process_log()
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"record: {output_file}")
        except Exception as e:
            self.logger.error(f"fail: {e}")
    
    def _exec_in_container(self, command: str, args: str = "") -> str:
        full_cmd = f"docker exec {self.container_id} {command} {args}"
        result = subprocess.run(full_cmd, shell=True, 
                              capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"command exec fail {result.stderr}")
        return result.stdout
    
    def _exec_with_real_time_monitoring(self, command: str, timeout_seconds: int) -> str:
        full_cmd = f"docker exec {self.container_id} {command}"
        
        try:
            process = subprocess.Popen(
                full_cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  
            )
            
            reader = EnhancedProcessStreamReader(process, self.stream_monitor, self)
            
            result = reader.read_with_monitoring(timeout=timeout_seconds)
            
            return_code = process.returncode
            if return_code is not None and return_code != 0:
                if not self.execution_stopped: 
                    raise RuntimeError(f"command exec fail, return code: {return_code}")
            
            return result
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"fail: {timeout_seconds}s")
            raise
        except Exception as e:
            self.logger.error(f"fail: {e}")
            raise
    
    def _analyze_real_time_results(self) -> None:
        try:
            stats = self.stream_monitor.get_statistics()
            
            if stats['tool_calls']:
                tool_summary = ', '.join(f"{tool}:{count}" for tool, count in stats['tool_calls'].items())
                self.logger.info(f"tool: {tool_summary}")
            
            if stats.get('id_format_stats'):
                format_summary = ', '.join(f"{fmt}:{count}" for fmt, count in stats['id_format_stats'].items())
            
            
            if stats['total_tool_calls'] < 5:  
                
                
                post_analysis = self._perform_post_process_analysis()
                if post_analysis and post_analysis['detected_tool_calls'] > stats['total_tool_calls']:
                    
                    if post_analysis['tool_breakdown']:
                        post_tool_summary = ', '.join(f"{tool}:{count}" for tool, count in post_analysis['tool_breakdown'].items())
                    
                    if post_analysis['id_format_distribution']:
                        post_format_summary = ', '.join(f"{fmt}:{count}" for fmt, count in post_analysis['id_format_distribution'].items())
                    
                    stats.update({
                        'post_process_analysis': post_analysis,
                        'analysis_note': 'Real-time monitoring may have missed some tool calls, post-process analysis provides more accurate results'
                    })
            
            
            self._log_process_step("real_time_stats", stats)
            
        except Exception as e:
            self.logger.warning(f"fail: {e}")
    
    def _perform_post_process_analysis(self) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(self, '_last_claude_output') and self._last_claude_output:
                return self.stream_monitor.analyze_completed_output(self._last_claude_output)
            else:
                return None
        except Exception as e:
            self.logger.warning(f"fail: {e}")
            return None

    def _exec_in_container_with_output_timeout(self, command: str, args: str = "", timeout_seconds: int = 1800) -> str:
        import subprocess
        full_cmd = f"docker exec {self.container_id} {command} {args}"
        
        try:
            result = subprocess.run(
                full_cmd, shell=True, 
                capture_output=True, text=True, 
                timeout=timeout_seconds
            )
            
            if result.returncode != 0:
                error_msg = f"fail: {result.stderr.strip()}"
                raise RuntimeError(error_msg)
                
            return result.stdout
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"fail: {timeout_seconds}s: {full_cmd}")
        except Exception as e:
            raise RuntimeError(f"fail: {e}")

    def _exec_in_container_with_output(self, command: str, args: str) -> str:
        full_cmd = f"docker exec -i {self.container_id} {command} {args}"
        result = subprocess.run(full_cmd, shell=True, 
                              capture_output=True, text=True, timeout=3600)
        return result.stdout + "\\n--- STDERR ---\\n" + result.stderr
    
    def _write_file_to_container(self, file_path: str, content: str) -> None:
        dir_path = str(Path(file_path).parent)
        self._exec_in_container("mkdir", f"-p {dir_path}")
        
        cmd = f"docker exec -i {self.container_id} tee {file_path}"
        subprocess.run(cmd, shell=True, input=content, text=True, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        self.logger.debug(f"success: {file_path}")
    
    def _check_repair_success(self, output: str) -> bool:
        success_indicators = [
            "fix success", "Task completed", "CVE repair task completed",
            "final-cve-fix.patch", "Successfully generated patch",
            "Smart repair completed", "patch generated successfully"
        ]
        
        success = any(indicator in output for indicator in success_indicators)
        
        if success:
            self.logger.info("success")
        else:
            self.logger.warning("fail")
            
        patch_exists = self._check_patch_file_exists()
        if patch_exists:
            self.logger.info("patch exists")
            success = True
            
        return success
    
    def _check_patch_file_exists(self) -> bool:
        patch_locations = [
            "/workspace/final-cve-fix.patch",
            f"{self.work_dir}/final-cve-fix.patch"
        ]
        
        for location in patch_locations:
            try:
                result = self._exec_in_container("test", f"-f {location}")
                return True
            except:
                continue
        return False
    
    def _extract_patch(self) -> str:
        try:
            patch_locations = [
                "/workspace/final-cve-fix.patch",  
                f"{self.work_dir}/final-cve-fix.patch",
                f"{self.work_dir}/.claude/outputs/patch.diff"
            ]
            
            for location in patch_locations:
                try:
                    patch_content = self._exec_in_container("cat", location)
                    if patch_content.strip():
                        self.logger.info(f"find patch: {location}")
                        return patch_content
                except:
                    continue
            
            if self.allow_git_diff_fallback:
                try:
                    git_patch = self._exec_in_container(
                        "bash", f"-c 'cd {self.work_dir} && git diff HEAD'")
                    if git_patch.strip():
                        
                        try:
                            fallback_patch = f"""# WARNING: This is a fallback patch generated by git diff
# It may contain changes unrelated to CVE repair
# Use with caution and manual verification

{git_patch}"""
                            self._exec_in_container("bash", 
                                f"-c 'cd {self.work_dir} && echo \"{fallback_patch}\" > /workspace/final-cve-fix.patch'")
                        except:
                            pass
                            
                        return git_patch
                except Exception as e:
                    self.logger.warning(f"git diff fallback: {e}")
            else:
                pass                
            return ""
            
        except Exception as e:
            return ""
    
    def get_container_logs(self) -> str:
        try:
            result = subprocess.run(
                f"docker logs {self.container_id}", 
                shell=True, capture_output=True, text=True
            )
            return result.stdout + result.stderr
        except Exception as e:
            return ""
    
    def cleanup(self):
        try:
            if self.temp_log_file and self.temp_log_file.exists():
                self.logger.info("complete real time log...")
                success = False  
                self._finalize_real_time_log(success)
            
            self._exec_in_container("rm", "-f /tmp/install_claude.sh")
        except Exception as e:
            self.logger.warning(f"fail: {e}")
    
    def set_success_and_finalize_log(self, success: bool, patch_content: str = "", container_logs: str = ""):
        try:
            self._finalize_real_time_log(success, patch_content, container_logs)
        except Exception as e:
            self.logger.warning(f"fail: {e}")