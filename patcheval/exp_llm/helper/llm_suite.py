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
import json
import time
import re
import os
import requests
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from string import Template
from openai import OpenAI



class LLMClient:
    def __init__(self, api_url: str, api_key: str, model_name: str, timeout: int, temperature: float, max_tokens: int, log_manager=None):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_manager = log_manager

        # Decide backend style
        self.use_curl = "completions" in (api_url or "")

        # Preconfigure headers for HTTP requests
        self.headers = {"Content-Type": "application/json"}
        if "no-key" not in (api_key or ""):
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Pre-create SDK client if needed
        self.sdk_client = None
        if not self.use_curl:
            self.sdk_client = OpenAI(api_key=api_key, base_url=api_url)

    # ===== Prompt building =====
    def build_prompt(
        self,
        functions_info: List[dict],
        cve2knowledge_info: Optional[dict],
        function_last_feedbacks: Dict[str, str],
        prompt_template: str,
        use_cot: bool = False,
        cve: Optional[str] = None,
        cve_logs: Optional[list] = None,
    ) -> str:
        """Build the LLM prompt using vulnerable function and CVE context.
        """
        # 1. Process CVE knowledge
        if cve2knowledge_info is not None:
            cwe_id = cve2knowledge_info.get("cwe_id", [None])[0] if cve2knowledge_info.get("cwe_id") else None
        else:
            cwe_id = None

        cve_description = cve2knowledge_info.get("cve_description", "") if cve2knowledge_info else ""
        cwe_name = (
            cve2knowledge_info["cwe_info"][cwe_id]["name"]
            if (cwe_id and cve2knowledge_info and cve2knowledge_info.get("cwe_info"))
            else ""
        )
        cwe_description = (
            cve2knowledge_info["cwe_info"][cwe_id]["description"]
            if (cwe_id and cve2knowledge_info and cve2knowledge_info.get("cwe_info"))
            else ""
        )
        one_shot_example = cve2knowledge_info["one_shot_cot"] if use_cot else cve2knowledge_info.get("one_shot", "")

        # 2. Construct multi-function content with last feedbacks
        function_sections: List[str] = []
        for func in functions_info:
            func_id = func["id"]
            original_code = func["original_code"]
            last_feedback = function_last_feedbacks.get(func_id, "")
            section = (
                f"### Function {func_id}\n"
                f"#### Last Round Feedback: \n{last_feedback}\n"
                f"#### Vulnerable Code:\n{original_code}"
            )
            function_sections.append(section)
        function_content = "\n\n".join(function_sections)

        # 3. Fill the prompt template
        return Template(prompt_template).substitute(
            cwe_id=cwe_id if cwe_id is not None else "",
            cwe_name=cwe_name,
            cwe_description=cwe_description,
            cve_description=cve_description,
            one_shot_example=one_shot_example,
            function_content=function_content,
        )

    def _call_remote_api(
        self,
        prompt: str,
        max_retries: int = 5,
        retry_delay: int = 5,
    ) -> Tuple[Optional[str], Optional[Any], float]:
        """Call remote API using HTTP or SDK backends with retry and logging.

        Mirrors the original _call_remote_api behavior, including log messages.
        """
        api_url = self.api_url
        use_curl = self.use_curl

        if use_curl:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
        else:
            client = self.sdk_client
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]

        # perform retries
        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                if use_curl:
                    completion = requests.post(
                        api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    completion.raise_for_status()
                    completion = completion.json()
                    content = completion["choices"][0]["message"]["content"]
                    elapsed = time.time() - start_time
                else:
                    completion = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                        stream=False,
                    )
                    content = completion.choices[0].message.content
                    elapsed = time.time() - start_time
                    
                return content, completion, elapsed
            except Exception as e:  
                elapsed = time.time() - start_time
                try:
                    error_message = str(e).lower()
                    if self.log_manager is not None:
                        logger = self.log_manager.get_current_logger()
                        logger.warning(f"error_message: {error_message}")
                        if attempt < max_retries:
                            logger.warning(
                                f"API call timeout: {str(e)} Duration: {elapsed:.2f} seconds. Will retry in {retry_delay} seconds ({attempt + 1}/{max_retries})..."
                            )
                        else:
                            logger.error(
                                f"API call final failed (epoch {attempt + 1}): {str(e)} Duration: {elapsed:.2f} seconds"
                            )
                except Exception:
                    # Avoid raising from logging to preserve runtime behavior
                    pass

                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    return None, None, elapsed

        return None, None, 0.0

    def call(self, prompt: str) -> Tuple[Optional[str], Optional[Any], float]:
        """Call the remote API with retry and logging (behavior-preserving)."""
        return self._call_remote_api(prompt)

    def parse_response(self, llm_response: str, cve: str) -> Dict[str, str]:
        """Parse the raw LLM response into per-function patches.

        Delegates to PatchParser to preserve original behavior and logs using contextual logger.
        """
        parser = PatchParser(log_manager=self.log_manager)
        return parser.parse(llm_response, cve)



class FeedbackHelper:
    """Feedback templating and updates for multi-round iterations.

    This helper centralizes feedback template selection by language and
    updates per-function feedback text for the next round.
    """

    def __init__(self, log_manager):
        self.log_manager = log_manager

    def get_feedback_template(self) -> str:
        """Return feedback template string, preserving original text."""
        return """##### Last round generated code:
{last_round_code}
##### Running by test results:
{last_round_test_result}"""

    def update_feedback(self, feedbacks: Dict[str, str], vul_code_cache: Dict[str, str], test_msg: str, template: Optional[str] = None) -> None:
        """Update per-function feedback strings using provided or default template.

        Args:
            feedbacks: Mapping from function id to feedback text to be updated.
            vul_code_cache: Cache mapping function id to last generated code.
            test_msg: The test result message string.
            template: Optional explicit template; defaults to get_feedback_template().
        """
        tmpl = template if template is not None else self.get_feedback_template()
        for vul_id, code in vul_code_cache.items():
            feedbacks[vul_id] = tmpl.format(
                last_round_code=code,
                last_round_test_result=test_msg,
            )


class PatchParser:
    """Parse LLM responses to extract per-function patches.

    This class exposes a cohesive, class-based API that mirrors the original
    behavior (JSON-first, regex fallback) while enabling dependency injection.
    """

    def __init__(self, log_manager=None):
        self.log_manager = log_manager
        self.validators = Validators()
        self.encode = self.validators.escape_decode_encode
    def parse(self, llm_response: str, cve: str) -> Dict[str, str]:
        """Extract patches for each function id using contextual logger.

        Args:
            llm_response: Raw response from LLM.

        Returns:
            Mapping of function id to patch string. Empty dict on failure.
        """
        self.log_manager.bind_current_task(cve)
        self.logger = self.log_manager.get_current_logger()
        
        def extract_patches(raw: str) -> Dict[str, str]:
            results: Dict[str, str] = {}
            stack: List[int] = []
            obj_ranges: List[tuple[int, int]] = []
            for i, char in enumerate(raw):
                if char == "{":
                    stack.append(i)
                elif char == "}":
                    if stack:
                        start_index = stack.pop()
                        if not stack:
                            obj_ranges.append((start_index, i))
            for start, end in obj_ranges:
                obj_content = raw[start : end + 1]
                try:
                    obj = json.loads(obj_content)
                    if "id" in obj and "patch" in obj:
                        results[obj["id"]] = obj["patch"]
                    continue
                except json.JSONDecodeError:
                    pass
                id_match = re.search(r'"id"\s*:\s*"([^"]+)"', obj_content)
                if not id_match:
                    continue
                patch_match = re.search(r'"patch"\s*:\s*"((?:[^"\\]|\\.)*)"', obj_content)
                if patch_match:
                    raw_patch = patch_match.group(1)
                    try:
                        patch_value = json.loads(f'"{raw_patch}"')
                    except Exception as e:
                        self.logger.error(f"JSONDecodeError: {str(e)}, LLM Respoonse: {repr(raw_patch)}")
                        patch_value = raw_patch
                    results[id_match.group(1)] = patch_value
            return results

        try:
            llm_response = llm_response.strip().strip("'").strip('"')
            json_match = re.search(r"```json\n(.*?)\n```(?!.*```)", llm_response, re.DOTALL)
            if json_match:
                llm_response = json_match.group(1).strip()
            data = json.loads(llm_response)
            if isinstance(data, list):
                processed: Dict[str, str] = {}
                for item in data:
                    if isinstance(item, dict) and "id" in item and "patch" in item:
                        func_id = item["id"]
                        patch = item["patch"]
                        processed[func_id] = patch
                        self.logger.debug(f"JSON Extract Success patch: {patch}")
                return processed
            elif isinstance(data, dict):
                processed = {}
                func_id = data["id"]
                patch = data["patch"]
                processed[func_id] = patch
                self.logger.debug(f"JSON Extract Success  patch: {patch}")
                return processed
        except json.JSONDecodeError:
            try:
                processed = {}
                patches = extract_patches(llm_response)
                for func_id, patch in patches.items():
                    if patch.count("\\n ") > 2 or patch.count("\\n\\t") > 2:
                        patch = self.encode(patch)
                    processed[func_id] = patch
                    self.logger.debug(f"Regex Extract Success patch: {patch}")
                return processed
            except Exception as e:
                msg = f"Regex Extract Failed: {str(e)}, LLM Response: {repr(llm_response)}"
                self.logger.error(msg)
                return {}
        except Exception as e:
            self.logger.error(f"Postprocess Code Failed: {str(e)}")
            return {}


class Validators:
    """Validation and normalization utilities."""

    def escape_decode_encode(self, text: str) -> str:
        """Decode common escape sequences in a string."""
        import codecs
        text_bytes = text.encode("utf-8")
        decoded, _ = codecs.escape_decode(text_bytes)
        return decoded.decode("utf-8")

    def process_original_code(self, patch: Union[str, List[str]]) -> str:
        """Return the original code string from a patch value."""
        code: str = patch[0] if isinstance(patch, list) else patch
        # Decode only when escaped newline sequences are present to preserve behavior
        if code.count("\\n ") > 1 or code.count("\\n\\t") > 1:
            code = self.escape_decode_encode(code)
        return code

    def get_language_info(self, vul_id: str, language_comment_map: Dict[str, Tuple[str, str]]) -> Tuple[str, str]:
        """Return (language, comment symbol) based on the vulnerability id."""
        try:
            lang_key = vul_id.split("_")[1].lower()
        except Exception:
            return ("Unknown", "#")
        return language_comment_map.get(lang_key, ("Unknown", "#"))


class FileOps:
    """File I/O helpers for CVE knowledge and outputs."""

    def load_cve_knowledge(self, data_knowledge_path: str) -> Dict[str, Dict[str, object]]:
        """Load CVE knowledge mapping from a JSON file path."""
        with open(data_knowledge_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            entry["cve_id"]: {
                k: entry.get(k, "")
                for k in (
                    "programming_language",
                    "cwe_id",
                    "cve_description",
                    "cwe_info",
                    "patch_url",
                    "repo",
                    "fix_func",
                    "one_shot",
                    "one_shot_cot",
                )
            }
            for entry in data
        }

    def load_existing_cves(self, output_path: str) -> set[str]:
        """Load existing CVE IDs from an output JSON file if present."""
        import os
        if not os.path.exists(output_path):
            return set()
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {cve for item in data for cve in item}

    def read_from_dir(self, input_dir: str) -> Dict[str, str]:
        """Read all .txt files from the given directory and return a mapping.

        The mapping key is the filename without extension and the value is the file content.
        """
        import os
        result: Dict[str, str] = {}
        file_list = os.listdir(input_dir)
        for file in file_list:
            if file.endswith(".txt"):
                with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
                    result[file[:-4]] = f.read()
        return result

    def clean_path(self, repo_path: str) -> None:
        """Clean local repository by removing .git directory."""
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
            
class CodeTagger:
    """Utility to add template tags into source code for vulnerability localization."""

    def __init__(self, log_manager=None):
        """Initialize with the provided contextual log manager."""
        self.log_manager = log_manager

    def add_template_tags(
        self,
        code: str,
        vul_infos: Sequence[dict],
        comment_sym: str,
        cve_info: dict,
        cve: str,
        cve_logs: list,
        template_name: str,
    ) -> str:
        """Add template tags to the given code based on vulnerability info and template."""
        if not vul_infos or not code:
            return code
        vul_infos = sorted(
            vul_infos,
            key=lambda x: x.get("vul_line", [])[-1] if x.get("vul_line") else -1,
            reverse=True,
        )
        
        lines = code.split("\n")
        for vul_info in vul_infos:
            tag = vul_info.get("tag")
            vul_lines = vul_info.get("vul_line", [])
            if not tag or not vul_lines:
                continue
            if "Ablation_with_location_Approximate" in template_name:
                self._add_location_random_tag(lines, vul_lines, comment_sym, cve, cve_logs, [(0, 5)])
            elif "Ablation_with_location_Imprecise" in template_name:
                self._add_location_random_tag(lines, vul_lines, comment_sym, cve, cve_logs, [(6, None)])
            elif "Ablation_with_location_Precise" in template_name:
                self._add_location_tag(lines, vul_lines, tag, comment_sym)
        return "\n".join(lines)

    def _add_location_tag(self, lines: List[str], vul_lines: List[int], tag: str, comment_sym: str) -> None:
        """Add explicit location tag markers given vulnerable line ranges."""
        if tag == "add":
            lines.insert(vul_lines[0], f"{comment_sym} <vul> </vul>")
        elif tag == "modify":
            lines.insert(vul_lines[-1], f"{comment_sym} </vul>")
            lines.insert(vul_lines[0] - 1, f"{comment_sym} <vul>")

    def _add_location_random_tag(
        self,
        lines: List[str],
        vul_lines: List[int],
        comment_sym: str,
        cve: str,
        cve_logs: list,
        distance_ranges=None,
    ) -> List[str]:
        """Insert tag markers at random positions constrained by distance to vulnerable lines."""
        self.log_manager.bind_current_task(cve)

        import random
        if not lines or not vul_lines:
            return lines
        if distance_ranges is None:
            distance_ranges = [(0, 5), (6, 10)]
        if self.log_manager is not None:
            logger = self.log_manager.get_current_logger()
        non_vul_lines = [i for i in range(len(lines)) if i not in vul_lines]
        candidate_lines: List[int] = []
        for line_num in non_vul_lines:
            dist = self._calculate_min_distance(line_num, vul_lines)
            for min_d, max_d in distance_ranges:
                min_check = dist >= min_d
                max_check = (max_d is None) or (dist <= max_d)
                if min_check and max_check:
                    candidate_lines.append(line_num)
                    break
        logger.debug(f"Original Lines: {len(non_vul_lines)}")
        logger.debug(f"Filtered Lines: {len(candidate_lines)}")
        logger.debug(f"Candidate Lines: {candidate_lines}")
        if not candidate_lines:
            return lines
        max_insertions = max(1, len(candidate_lines) // 4)
        num_insertions = random.randint(1, max_insertions)
        selected_lines = random.sample(candidate_lines, min(num_insertions, len(candidate_lines)))
        logger.debug(f"Max Insertions: {max_insertions}")
        logger.debug(f"Selected Insertions: {num_insertions}")
        logger.debug(f"Selected Lines: {selected_lines}")
        tag_insertions = []
        for line_num in selected_lines:
            mode = random.randint(0, 1)
            if mode == 0:
                tag_insertions.append({"type": "single", "line": line_num})
            else:
                max_span = len(lines) - line_num - 1
                span = random.randint(1, min(3, max(1, max_span))) if max_span > 0 else 0
                if span > 0:
                    tag_insertions.append({"type": "pair", "start": line_num, "end": line_num + span})
                else:
                    tag_insertions.append({"type": "single", "line": line_num})
        tag_insertions = sorted(tag_insertions, key=lambda x: x['start'] if x['type'] == 'pair' else x['line'])
        for i in range(1, len(tag_insertions)):
            prev = tag_insertions[i - 1]
            curr = tag_insertions[i]
            prev_start = prev['start'] if prev['type'] == 'pair' else prev['line']
            prev_end = prev['end'] if prev['type'] == 'pair' else prev['line']
            curr_start = curr['start'] if curr['type'] == 'pair' else curr['line']
            curr_end = curr['end'] if curr['type'] == 'pair' else curr['line']
            if curr_start <= prev_end:
                if curr['type'] == 'single':
                    new_line = prev_end + 1
                    if new_line < len(lines):
                        curr['line'] = new_line
                else:
                    new_start = prev_end + 1
                    if new_start + (curr_end - curr_start) < len(lines):
                        curr['start'] = new_start
                        curr['end'] = new_start + (curr_end - curr_start)
                    else:
                        new_span = len(lines) - new_start - 1
                        if new_span > 0:
                            curr['start'] = new_start
                            curr['end'] = new_start + new_span
                        else:
                            curr['type'] = 'single'
                            curr['line'] = new_start if new_start < len(lines) else len(lines) - 1
                            del curr['start']
                            del curr['end']
        tag_insertions = sorted(tag_insertions, key=lambda x: x.get('start', x.get('line')), reverse=True)
        for tag in tag_insertions:
            if tag['type'] == 'single':
                line_idx = min(tag['line'], len(lines))
                lines.insert(line_idx, f"{comment_sym} <vul> </vul>")
            elif 'start' in tag and 'end' in tag:
                end_idx = min(tag['end'] + 1, len(lines))
                start_idx = min(tag['start'], len(lines))
                lines.insert(end_idx, f"{comment_sym} </vul>")
                lines.insert(start_idx, f"{comment_sym} <vul>")
        return lines

    def _calculate_min_distance(self, line_num: int, vul_lines: Sequence[int]) -> int:
        """Return minimum distance between line_num and any vulnerable line number."""
        if not vul_lines:
            return 10**9
        return min(abs(line_num - v_line) for v_line in vul_lines)


class CodeApplier:
    """Apply code changes and generate repository diffs using replacer.

    This class encapsulates operations previously implemented in VulFixer:
    - Apply code change to a local test repository
    - Generate CVE-level diffs by aggregating file-level diffs

    All behaviors, logs, and error messages are preserved.
    """

    def __init__(self, log_manager=None):
        self.log_manager = log_manager

    def apply_change(
        self,
        replacer,
        test_repo: str,
        vul_entry: dict,
        code: str,
        language: str,
    ) -> None:
        """Apply code change to the local test repository.

        Args:
            replacer: FuncReplacer instance used to modify files.
            test_repo: Path to the test repository.
            vul_entry: Vulnerability entry containing file_path and line ranges.
            code: The generated fixed code snippet.
            language: Programming language associated with the file.
        """
        import os

        file_path = os.path.join(test_repo, vul_entry["file_path"])
        if code != "":
            replacer.replace(
                file_path,
                int(vul_entry["start_line"]),
                int(vul_entry["end_line"]),
                code,
                language,
            )

    def generate_cve_diff(
        self,
        replacer,
        file_paths,
        cve,
    ) -> str:
        """Generate aggregated diff for each (repo_path, file_path) pair.

        Args:
            replacer: FuncReplacer instance capable of generating diffs.
            file_paths: Iterable of (repo_path, file_path) tuples.
            cache_log: Logging callback used for error messages.
            cve: CVE identifier for logging.
            cve_logs: Log record cache list.

        Returns:
            Aggregated diff string joined by newlines.
        """
        diffs = []
        for dest_path, file_path in file_paths:
            try:
                # Try modern signature first
                diffs.append(replacer.generate_diff(dest_path, file_path))
            except Exception as e:
                if self.log_manager is not None:
                    self.log_manager.bind_current_task(cve)
                    self.log_manager.get_current_logger().error(f"Generate diff failed: {file_path} - {str(e)}")
        return "\n".join(diffs)


class SuccessEvaluator:
    """Decide whether a fix is successful based on test outcomes."""

    def __init__(self):
        pass

    def is_success(self, cve: str, test_result: Any, test_msg: str, unittest_res: Any, unittest_msg: str) -> bool:
        """Return True if the fix is considered successful, mirroring original logic."""
        return bool(test_result) and bool(unittest_res)


class TestRunner:
    """Facade to run tests via injected Eval implementation."""

    def __init__(self, eval_factory, log_manager=None):
        """Initialize with a factory that produces Eval objects and optional log manager."""
        self.eval_factory = eval_factory
        self.log_manager = log_manager
    def run(self, cve: str, diff: str, language: str, test_name: str):
        """Run evaluation and return test results, preserving logging semantics."""
        try:
            eval_obj = self.eval_factory(self.log_manager, cve)
            return eval_obj.run_evaluation(cve, diff, language, test_name)
        except Exception as e:
            if self.log_manager is not None:
                self.log_manager.bind_current_task(cve)
                logger = self.log_manager.get_current_logger()
                logger.error(f"Run test failed: {cve} - {str(e)}")
            return False, f"Run test failed: {str(e)}", None, None, None
