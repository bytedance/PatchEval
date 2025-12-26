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
import os
import time
import traceback
import threading
from collections import defaultdict
import logging
from typing import Optional
from .func_replacer import FuncReplacer
from evaluation.run_evaluation import Evaluation
from .logger import get_logger, LogContextManager
from .llm_suite import (
    LLMClient,
    PatchParser,
    Validators,
    FileOps,
    CodeTagger,
    SuccessEvaluator,
    TestRunner,
    CodeApplier,
    FeedbackHelper,
)
from .task_manager import TaskManager, TaskSpec


class VulFixer:
    def __init__(
        self,
        args,
        task_manager: Optional[TaskManager] = None,
        validators: Optional[Validators] = None,
        file_ops: Optional[FileOps] = None,
        patch_parser: Optional[PatchParser] = None,
        llm_client: Optional[LLMClient] = None,
        code_applier: Optional[CodeApplier] = None,
        feedback_helper: Optional[FeedbackHelper] = None,
        log_manager: Optional[LogContextManager] = None,
    ):
        # initialize logger
        self.logger = get_logger()   
        self.args = args
        self.api_key = args.api_key
        self.api_url = args.api_url
        self.model_name = args.model
        if self.model_name in ["gpt-5-2025-08-07","o3-2025-04-16"]:
            self.temperature = 1
        else:
            self.temperature = args.temperature
            
        self.prompt_template = args.prompt_template
        if "no-key" in self.api_key:
            self.headers = {
                "Content-Type": "application/json"
            }
        else:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_time = 0

        self.cve_prompt_tokens = 0
        self.cve_completion_tokens = 0
        self.cve_total_tokens = 0
        self.logger_levels = [logging.INFO, logging.ERROR] if not self.args.debug else [logging.INFO, logging.ERROR, logging.DEBUG, logging.WARNING]
        self.logger.info(f"log level: {self.logger_levels}, debug: {self.args.debug}")
        self.log_manager = log_manager or LogContextManager(self.logger)

        self.LANGUAGE_COMMENT_MAP = {
            "py": ("Python", "#"),
            "js": ("JavaScript", "//"),
            "go": ("Go", "//")
        }
        self.PROMPT_TRUNCATE_LENGTH = 500
        self.llm_client = llm_client or LLMClient(self.api_url, self.api_key, self.model_name, self.args.timeout, self.temperature, self.args.max_tokens, log_manager=self.log_manager)
        self.replacer_factory = lambda log_manager: FuncReplacer(self.log_manager)
        self.eval_factory = lambda log_manager, cve: Evaluation(self.log_manager, cve)
        self.task_manager = task_manager or TaskManager(self.args.local_repo_path, self.args.max_workers, log_manager=self.log_manager)
        self.tagger = CodeTagger(log_manager=self.log_manager)
        self.validators = validators or Validators()
        self.file_ops = file_ops or FileOps()
        self.patch_parser = patch_parser or PatchParser(log_manager=self.log_manager)
        self.success_evaluator = SuccessEvaluator()
        self.test_runner = TestRunner(self.eval_factory, log_manager=self.log_manager)
        self.code_applier = code_applier or CodeApplier(log_manager=self.log_manager)
        self.feedback_helper = feedback_helper or FeedbackHelper(log_manager=self.log_manager)

    def process_vulnerability(self, input_data):
        """
        Process vulnerability data 
        1. task_result: single vulnerability entry processing result
        2. cve_test_result: single CVE test result (each epoch)
        3. summary: final statistics information
        """

        # initialize statistics information
        stats = defaultdict(int)
        total_tasks = sum(len(entry['vul_func']) for entry in input_data)
        stats_lock = threading.Lock()
        global_start = time.time()

        # load CVE knowledge & existing CVEs
        cve_knowledge = self.file_ops.load_cve_knowledge(self.args.input)
        existing_cves = self.file_ops.load_existing_cves(self.args.output)
        self.logger.info(f"existing_cves: {existing_cves}", extra={"cve": "GLOBAL"})

        # filter CVE entries to process (skip existing or configured)
        valid_cves = []
        
        for input_item in input_data:
            cve = input_item["cve_id"]
            vul_entries = input_item['vul_func']
            poc = input_item["is_poc"]
            if cve in existing_cves:
                self.logger.info(f"EXISTED CVE: {cve}, SKIP PROCESS", extra={"cve": "GLOBAL"})
                continue
            if poc is False:
                # self.logger.info(f"NO POC: {cve}", extra={"cve": "GLOBAL"})
                continue
            valid_cves.append((cve, vul_entries))

        self.logger.info(f"Multi-threading: {self.args.max_workers}", extra={"cve": "GLOBAL"})
        specs = [
            TaskSpec(
                key=cve,
                func=self._process_single_cve,
                args=(
                    cve,
                    vul_entries,
                    cve_knowledge,
                    stats,
                    stats_lock,
                ),
            )
            for cve, vul_entries in valid_cves
        ]

        for outcome in self.task_manager.run(specs):
            if outcome.error is not None:
                self.logger.error(f"CVE {outcome.key} Process Failed: {str(outcome.error)}", extra={"cve": outcome.key})
            else:
                for result in outcome.result:
                    yield result

        # generate final statistics
        with stats_lock:
            final_stats = stats.copy()
        yield self.task_manager.create_summary(final_stats, total_tasks, global_start)

        # clean project test path
        template_name = self.args.template.split("/")[-1].split(".txt")[0]
        test_name = f"{self.model_name.replace('/', '')}_{template_name}"
        
        if os.path.exists(f"exp_llm/projects_test/{test_name}"):
            self.file_ops.clean_path(f"exp_llm/projects_test/{test_name}")
        self.logger.info(f"Clean project test path: {f'exp_llm/projects_test/{test_name}'}")

    def _process_single_cve(
        self,
        cve,
        vul_entries,
        cve_knowledge,
        stats,
        stats_lock,
    ):
        """process single CVE"""
        cve_token_stats = {}
        results = []
        cve_logs = []  

        self.log_manager.start_task(task_id=cve, cve_id=cve, buffer_ref=cve_logs, allowed_levels=set(self.logger_levels))

        try:
            # 1. log the beginning of vulnerability processing
            self.logger.info(f"Start Processing Vulnerability (total: {len(vul_entries)})", extra={"cve": cve})
            ctx_logger = self.log_manager.get_current_logger()

            with stats_lock:
                stats["total_cves"] += 1
            ctx_logger.debug("Total CVE Count")  

            # initialize feedback template
            feedback_template = self.feedback_helper.get_feedback_template()
            last_feedbacks = defaultdict(lambda: feedback_template)
            vul_code_cache = {}  

            replacer = FuncReplacer(self.log_manager)

            # sort vulnerability entries by start line in descending order
            vul_entries.sort(key=lambda x: x['start_line'], reverse=True)
            repo_ctx = self.task_manager.init_repo_context(
                vul_entries,
                cve,
                cve_knowledge[cve]['repo'].split("/")[-1],
                self.args.template,
                self.model_name,
                self.replacer_factory,
            )  
            ctx_logger.info("▷ Processing Vulnerability")

            early_stop = False
            # multi-round iteration to fix vulnerabilities
            for current_epoch in range(self.args.epochs):
                for pass_idx in range(self.args.pass_k):
                    pass_results = []
                    ctx_logger.info(f"Epoch: {current_epoch+1}, Pass@{self.args.pass_k}: {pass_idx+1}")
                    cve_token_stats[current_epoch]={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                    
                    runtime_error = False
                    file_paths = set()

                    replacer.reset_and_checkout(repo_ctx["repo"], repo_ctx["commit"], repo_ctx["test_repo"])
                    ctx_logger.debug(f"reset and checkout target commit")
                    
                    # collect information of each vulnerability
                    vul_functions = []
                    skipped_vuls = []  
                    for vul_entry in vul_entries:
                        try:
                            original_code = self.validators.process_original_code(vul_entry["snippet"])
                            if not original_code:
                                skipped_vuls.append(vul_entry)
                                continue
                            # get programing language and comment symbol
                            language, comment_sym = self.validators.get_language_info(vul_entry["id"], self.LANGUAGE_COMMENT_MAP)
                            # add template tags to original code
                            if not "Default" in repo_ctx["template_name"] and not "Ablation_without_" in repo_ctx["template_name"]:
                                original_code = self.tagger.add_template_tags(
                                    original_code, 
                                    vul_entry.get("vul_info", []), 
                                    comment_sym, 
                                    cve_knowledge[cve],
                                    cve, 
                                    cve_logs, 
                                    repo_ctx["template_name"]
                                )
                            
                            # save information of each vulnerability
                            vul_functions.append({
                                "id": vul_entry["id"],
                                "original_code": original_code,
                                "vul_entry": vul_entry,
                                "language": language,
                                "comment_sym": comment_sym
                            })
                            ctx_logger.debug(f"original_code: \n{original_code}")

                        except Exception as e:
                            ctx_logger.error(f"Process Original Code {vul_entry['id']} failed: {str(e)}. Error Type: {type(e).__name__}. Error Info: {traceback.format_exc()}")

                            skipped_vuls.append(vul_entry)
                            with stats_lock:
                                stats["errors"] += 1

                    ctx_logger.debug("# 1. Finish Process Original Code")

                    # 2. handle skipped vulnerabilities
                    for vul_entry in skipped_vuls:
                        task_ctx = self.task_manager.init_task_context(cve, cve_knowledge[cve], vul_entry)
                        pass_results.append(self.task_manager.create_skipped_result(task_ctx, "Original Code Process Failed"))
                        self.task_manager.finalize_task_context(task_ctx)

                    # 3. handle no valid function in this round
                    if not vul_functions:
                        ctx_logger.info(f"CVE {cve} No Valid Function in this round, skip")
                        continue
                    
                    # 4. generate prompt for LLM
                    use_cot = "without_cot" not in self.args.template
                    function_last_feedbacks = {f["id"]: last_feedbacks.get(f["id"], "") for f in vul_functions}
                    prompt = self.llm_client.build_prompt(vul_functions, cve_knowledge[cve], function_last_feedbacks, self.prompt_template, use_cot, cve, cve_logs)
                    ctx_logger.debug(f"CVE {cve} Prompt:\n{prompt[:self.PROMPT_TRUNCATE_LENGTH]}...")
                    
                    # 5. call LLM to generate fix code
                    content, token_info, elapsed = self.llm_client.call(prompt)
                    ctx_logger.debug(f"resp: {content}")
                    llm_resp, token_info, api_time = content, token_info, round(elapsed, 2)

                    ctx_logger.info(f"▷ Finish Call LLM, api_time: {api_time}")
                    ctx_logger.info(f"repr(llm_resp): {repr(llm_resp)}")
                    if not llm_resp:
                        ctx_logger.error(f"CVE {cve} LLM Call Failed")

                        for func in vul_functions:
                            task_ctx = self.task_manager.init_task_context(cve, cve_knowledge[cve], func["vul_entry"])
                            pass_results.append(self.task_manager.create_api_fail_result(task_ctx, api_time))
                            self.task_manager.finalize_task_context(task_ctx)
                        continue
                    
                    if token_info:
                        if hasattr(token_info, 'usage'):  
                            usage = token_info.usage
                            prompt_tokens = usage.prompt_tokens
                            completion_tokens = usage.completion_tokens
                            total_tokens = usage.total_tokens
                        elif isinstance(token_info, dict) and 'usage' in token_info:  
                            usage = token_info['usage']
                            prompt_tokens = usage.get('prompt_tokens', 0)
                            completion_tokens = usage.get('completion_tokens', 0)
                            total_tokens = usage.get('total_tokens', 0)
                        
                        with stats_lock:
                            cve_token_stats[current_epoch]["prompt_tokens"] += prompt_tokens
                            cve_token_stats[current_epoch]["completion_tokens"] += completion_tokens
                            cve_token_stats[current_epoch]["total_tokens"] += total_tokens
                            stats["total_prompt_tokens"] += prompt_tokens
                            stats["total_completion_tokens"] += completion_tokens
                            stats["total_tokens"] += total_tokens
                            stats["total_time"] += api_time
                        
                        ctx_logger.debug(
                            f"API call finish Token Used: {cve_token_stats[current_epoch]['prompt_tokens']} prompt tokens, " +
                            f"{cve_token_stats[current_epoch]['completion_tokens']} completion tokens，{cve_token_stats[current_epoch]['total_tokens']} total tokens time={api_time:.2f} seconds\n" +
                            f"total prompt tokens={stats['total_prompt_tokens']}, " +
                            f"total completion tokens={stats['total_completion_tokens']}, " +
                            f"total tokens ={stats['total_tokens']}, " +
                            f"total time={stats['total_time']:.2f} seconds"
                        )  

                        
                    # 6. parse LLM response to get fix code
                    function_results = self.llm_client.parse_response(llm_resp, cve)
                    if not function_results:
                        ctx_logger.error(f"CVE {cve} LLM Response Parse Failed: {llm_resp}")

                        for func in vul_functions:
                            task_ctx = self.task_manager.init_task_context(cve, cve_knowledge[cve], func["vul_entry"])
                            pass_results.append(self.task_manager.create_error_result(task_ctx, "LLM Response need JSON format"))
                            with stats_lock:
                                stats["errors"] += 1
                            self.task_manager.finalize_task_context(task_ctx)
                        continue
                    ctx_logger.debug("Finish LLM response extract")
                    ctx_logger.debug(f"function_results: {function_results}")

                    
                    # 7. replace the vulnerable functions in project_test to generate a diff
                    for func in vul_functions:
                        vul_entry = func["vul_entry"]
                        vul_id = func["id"]
                        task_ctx = self.task_manager.init_task_context(cve, cve_knowledge[cve], vul_entry)
                        ctx_logger.debug(f"vul_id: {vul_id}")

                        try:
                            if vul_id not in function_results and vul_id.replace("vul", "fix") not in function_results:
                                ctx_logger.warning(f"vul_id: {vul_id} not in function_results")

                                with stats_lock:
                                    stats["errors"] += 1
                                continue

                            # get fix code for current vulnerability
                            processed_code = function_results[vul_id]
                            vul_code_cache[vul_id] = processed_code  
                            # apply fix code changes to test repository
                            ctx_logger.debug(f"Apply code diff")
                            self.code_applier.apply_change(repo_ctx["replacer"], repo_ctx["test_repo"], vul_entry, processed_code, func["language"])
                            file_paths.add((repo_ctx["test_repo"], vul_entry["file_path"]))

                            pass_results.append(self.task_manager.create_success_result(task_ctx, processed_code, api_time, current_epoch, cve_token_stats))

                            with stats_lock:
                                stats["success"] += 1

                        except Exception as e:
                            ctx_logger.error(f"Process vul_id: {vul_id} failed: {str(e)}")

                            runtime_error = True
                            pass_results.append(self.task_manager.create_error_result(task_ctx, str(e)))
                            with stats_lock:
                                stats["errors"] += 1

                        finally:
                            self.task_manager.finalize_task_context(task_ctx)
                    
                    # 8. run test for each vulnerability
                    if not runtime_error:
                        cve_diff = self.code_applier.generate_cve_diff(repo_ctx["replacer"], file_paths, cve)
                        ctx_logger.debug(f"cve_diff: {cve_diff}")
                        test_res, test_msg, unittest_res, unittest_msg, error_type = self.test_runner.run(cve, cve_diff, language, repo_ctx["test_name"])
                        evaluation_result = self.success_evaluator.is_success(cve, test_res, test_msg, unittest_res, unittest_msg)
                        
                        pass_results.append(self.task_manager.create_cve_test_result(cve, function_results, current_epoch, test_res, test_msg, unittest_res, unittest_msg, error_type, cve_diff, cve_token_stats, evaluation_result))

                        if evaluation_result:
                            ctx_logger.debug(f"Evaluation success in {current_epoch+1} epochs {pass_idx+1} pass@{self.args.pass_k}")
                            early_stop = True
                            break

                # update results
                results.extend(pass_results)  
                if early_stop:
                    break  
                # update feedback for next round
                self.feedback_helper.update_feedback(last_feedbacks, vul_code_cache, test_msg, feedback_template)

        except Exception as e:
            error_msg = f"CVE {cve} Process failed: {str(e)}. Error Type: {type(e).__name__}. Error Info: {traceback.format_exc()}"
            ctx_logger.error(error_msg)

        finally:
            ctx_logger.info(f"Finish Processing Vulnerability (total: {len(cve_logs)} logs)")
            self.file_ops.clean_path(repo_ctx["test_repo"])
            ctx_logger.info(f"Clean project test path: {repo_ctx['test_repo']}")
            # Allow TaskManager lifecycle finalize if needed (context cleanup)
            try:
                self.log_manager.finalize_task(cve)
            except Exception:
                pass
            for record in cve_logs:
                self.logger.handle(record)  
            cve_logs.clear()  # clean cache

        return results
