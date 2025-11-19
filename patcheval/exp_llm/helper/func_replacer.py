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

import difflib
import os
from pathlib import Path
import re
import shutil
import subprocess
import textwrap
from typing import Dict, List, Optional, Set

class FuncReplacer:
    def __init__(self, log_manager=None):
        self.strategies = {
            "Go": self._replace_code_generic,
            "Python": self._replace_code_generic,
            "JavaScript": self._replace_code_generic,
        }
        self.logger = log_manager.get_current_logger()

    def replace(
        self,
        file_path: str,
        start: int,
        end: int,
        new_content: Optional[str],
        project_type: str,
    ) -> Dict[str, str]:
        """
        Replace code in file and return original/replaced code and unified diff.

        Return format:
        {
            'original_code': str,  # Original code
            'replaced_code': str,  # Replaced code
            'diff': str            # Diff in similar GitHub format
        }
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            self.logger.error(f"Read {file_path} failed: {e}")
            raise
        
        if start < 0 or end < start or start > len(lines) + 1:
            raise ValueError(f"Invalid line range: start={start}, end={end}, total lines={len(lines)}")
        
        self.logger.debug("Start replace")

        backup_path = f"{file_path}.bak"
        try:
            shutil.copyfile(file_path, backup_path)
            self.logger.debug(f"Create backup file: {backup_path}")
        except Exception as e:
            self.logger.error(f"Create backup file {backup_path} failed: {e}")

        if project_type not in self.strategies:
            error_msg = f"Unsupported project type: {project_type}, will use generic replace logic."
            self.logger.error(error_msg)
            replace_func = self._replace_code_generic
        else:
            replace_func = self.strategies[project_type]

        try:
            original_content = "".join(lines[start - 1 : end])

            replace_func(lines, start, end, new_content or "")

            replaced_content = "".join(
                lines[start - 1 : start - 1 + len((new_content or "").splitlines())]
            )
            self.logger.debug(f"Replace content: {replaced_content}")

            diff = "".join(
                difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    replaced_content.splitlines(keepends=True),
                    fromfile="original",
                    tofile="replaced",
                    n=3,
                )
            )

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            self.logger.debug(f"Replace {file_path} content success")

            return {
                "original_code": original_content,
                "replaced_code": replaced_content,
                "diff": diff,
            }
        except Exception as e:
            self.logger.error(f"Replace {file_path} content failed: {e}")
            try:
                shutil.copyfile(backup_path, file_path)
                self.logger.info(f"Restore {file_path} from backup {backup_path}")
            except Exception as backup_error:
                self.logger.error(
                    f"Restore {file_path} from backup {backup_path} failed: {backup_error}"
                )
            raise

    def _get_indentation(self, line: str) -> str:
        """Get indentation of a line (spaces or tabs)."""
        if not line or not isinstance(line, str):
            return ""
        match = re.match(r"^(\s*)", line)
        return match.group(1) if match else ""

    def _replace_code_generic(
        self,
        lines: List[str],
        start: int,
        end: int,
        new_content: str,
    ) -> None:
        """
        Generic code replacement logic that preserves base indentation and relative structure.
        Modifies the provided lines list in place.
        """
        if new_content is None:
            self.logger.info("Replace content is None, will be treated as empty string")
            new_content = ""

        start_index = max(0, start - 1)
        end_index = min(len(lines), end)  

        if start < 1 or end > len(lines) + 1:
            self.logger.error(f"Original file [{start}, {end}] may be out of range of file lines {len(lines)}. Adjust to valid range [{start_index + 1}, {end_index}]" )
        if start_index > end_index:
            self.logger.error(f"Calculate replace range (start_index={start_index}, end_index={end_index}) will not replace any content")

        if start_index > end_index:
            self.logger.error(f"Replace range (start_index={start_index}, end_index={end_index}) will not replace any content. Insert content at line {start_index + 1}")

        # Base indentation determination
        target_base_indentation = ""
        if 0 <= start_index < len(lines):
            target_base_indentation = self._get_indentation(lines[start_index])
        elif 0 < start_index <= len(lines):
            target_base_indentation = self._get_indentation(lines[start_index - 1])
        else:
            self.logger.error("Cannot determine context indentation. Will not add extra base indentation.")

        processed_new_content_lines: List[str] = []

        if not new_content.strip():
            self.logger.warning("Replace content is empty or only whitespace.")
            if start_index < end_index:
                self.logger.warning("Replace content is empty or only whitespace. Will delete original content.")
            else:
                self.logger.warning("Replace content is empty or only whitespace. Will not insert any line.")
        else:
            try:
                dedented_content = textwrap.dedent(new_content)
            except Exception as dedent_error:
                self.logger.error(f"textwrap.dedent failed: {dedent_error}. Falling back to original new_content.")
                dedented_content = new_content

            content_lines = dedented_content.splitlines()
            self.logger.debug(f"Apply target base indentation '{target_base_indentation}' to each line of dedent new content.")

            for i, line in enumerate(content_lines):
                processed_line = target_base_indentation + line + "\n"
                processed_new_content_lines.append(processed_line)

        final_content_str = "".join(processed_new_content_lines)

        del lines[start_index:end_index]
        lines[start_index:start_index] = processed_new_content_lines

        if start_index < len(lines):
            replaced_content_to_log = "".join(
                lines[start_index : start_index + len(processed_new_content_lines)]
            )
            self.logger.debug(f"Replace content ({len(processed_new_content_lines)}  from line {start_index + 1}):\n{replaced_content_to_log}")

    def run_cmd(self, cmd: List[str], cwd: str, timeout: int = 120) -> Optional[str]:
        """Run shell command and return stdout or None on failure."""
        try:
            result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
            return result.stdout
        except subprocess.CalledProcessError as e:
            error_msg = (
                "\n⚠️ Run shell command failed ⚠️\n"
                f"Command: {e.cmd}\n"
                f"Exit code: {e.returncode}\n"
                "------ Standard Output ------\n"
                f"{e.stdout if e.stdout else '<Empty>'}\n"
                "------ Standard Error ------\n"
                f"{e.stderr if e.stderr else '<Empty>'}\n"
                "----------------------"
            )
            self.logger.warning(error_msg)
            return None
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Git command timed out: {' '.join(e.cmd)}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while running git command: {e}")
            return None
        
    def reset_and_checkout(
        self,
        repo_path: str,
        commit_ref: str,
        dest_path: str,
        force_recreate: bool = True,
    ) -> bool:
        repo_path_abs = Path(repo_path).resolve()
        dest_path_abs = Path(dest_path).resolve()

        if not repo_path_abs.joinpath(".git").is_dir():
            self.logger.error(f"Source path '{repo_path_abs}' is not a valid Git repository.")
            return False

        if force_recreate:
            self.logger.debug(f"Force recreate is enabled. Cleaning up destination '{dest_path_abs}' before creation...")
            
            try:
                self.run_cmd(["git", "worktree", "prune"], cwd=str(repo_path_abs))
                self.logger.debug("Successfully pruned stale worktrees.")
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Failed to prune worktrees, but continuing. Error: {e}")

            if dest_path_abs.exists():
                self.logger.warning(f"Destination '{dest_path_abs}' still exists. Performing removal.")
                try:
                    self.run_cmd(["git", "worktree", "remove", "--force", str(dest_path_abs)], cwd=str(repo_path_abs))
                    self.logger.debug(f"Successfully removed existing worktree at '{dest_path_abs}'.")
                except subprocess.CalledProcessError:
                    self.logger.warning(f"Path '{dest_path_abs}' is not a registered worktree. Deleting directory using shutil.")
                    shutil.rmtree(dest_path_abs)
        
        elif dest_path_abs.exists():
            self.logger.error(f"Destination path '{dest_path_abs}' already exists and force_recreate is False.")
            return False
        
        dest_path_abs.parent.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Creating new worktree for commit '{commit_ref}' at '{dest_path_abs}'...")
        result = self.run_cmd(
            ["git", "worktree", "add", "--detach", str(dest_path_abs), commit_ref],
            cwd=str(repo_path_abs),
        )
        if result is not None:
            self.logger.debug("Successfully created clean workspace.")
            return True
        else:
            self.logger.warning(f"Failed to create worktree. Attempting to fetch '{commit_ref}' from origin...")
            try:
                self.run_cmd(["git", "fetch", "origin", commit_ref], cwd=str(repo_path_abs))
                self.logger.debug("Fetch successful. Retrying to create worktree...")
                self.run_cmd(
                    ["git", "worktree", "add", "--detach", str(dest_path_abs), commit_ref],
                    cwd=str(repo_path_abs),
                )
                self.logger.info("Successfully created clean workspace after fetch.")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to create worktree even after fetching: {e}")
                return False
    
    def generate_diff(self, dest_path: str, file_path: str) -> str:
        """Generate diff for the specified file using git diff."""
        self.logger.debug(f"Generate diff for file: {file_path}")
        diff_cmd = ["git", "-C", dest_path, "diff", file_path]
        diff_output = self.run_cmd(diff_cmd, "./") or ""
        return diff_output + "\n"

if __name__ == "__main__":
    pass
