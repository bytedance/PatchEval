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
prefix = "claud4_exp5"
data_dir = f"./outputs/{prefix}/process_logs"

results = {}
for cve in os.listdir(data_dir):
    cve_name = cve.split(".")[0].split("_")[0]
    log_path = f"{data_dir}/{cve}"
    log = json.load(open(log_path))
    res = log["process_steps"][-1]['message']
    # print(res)
    input_token, output_token, tool_calls = res["total_input_tokens"], res["total_output_tokens"], res["total_tool_calls"]
    results[cve_name] = {
        "input_token": input_token,
        "output_token": output_token,
        "tool_calls": tool_calls
    }

json.dump(results, open(f"./outputs/{prefix}/results.json", "w"))

summury = {
    "avg_input_tokens": (sum([res["input_token"] for res in results.values()]) / len(results))/1000000,
    "avg_output_tokens": (sum([res["output_token"] for res in results.values()]) / len(results))/1000000,
    "avg_tool_calls": sum([res["tool_calls"] for res in results.values()]) / len(results),
}

cost = summury['avg_input_tokens'] * 3 + summury['avg_output_tokens'] * 15
print(summury)
print(summury['avg_input_tokens']+summury['avg_output_tokens'],cost)
