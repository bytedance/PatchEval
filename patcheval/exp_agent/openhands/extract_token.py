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
import os
prefix = ""
data_path = f"./evaluation/evaluation_outputs/{prefix}/output.jsonl"

results = {}
with open(data_path) as f:
    for l in f:
        data = json.loads(l)
        cve_name = data['instance_id']
        res = data["metrics"]["accumulated_token_usage"]
        # print(res)
        input_token, output_token, tool_calls = res["prompt_tokens"], res["completion_tokens"], len(data['history'])
        results[cve_name] = {
            "input_token": input_token,
            "output_token": output_token,
            "tool_calls": tool_calls
        }
print(len(results))
# json.dump(results, open(f"./evaluation/results/{prefix}/results.json", "w"))

summury = {
    "avg_input_tokens": (sum([res["input_token"] for res in results.values()]) / len(results))/1000000,
    "avg_output_tokens": (sum([res["output_token"] for res in results.values()]) / len(results))/1000000,
    "avg_tool_calls": sum([res["tool_calls"] for res in results.values()]) / len(results),
}

cost = summury['avg_input_tokens'] * 3 + summury['avg_output_tokens'] * 15
print(summury)
print(summury['avg_input_tokens']+summury['avg_output_tokens'],cost)

