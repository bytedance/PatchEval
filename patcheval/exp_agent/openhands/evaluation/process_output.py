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
import argparse
import utils
def main():
    patch_datas = []
    with open(args.output_file) as f:
        for l in f:
            data = json.loads(l)
            if 'git_patch' in data['test_result']:
                fix_patch = data['test_result']['git_patch']
            else:
                fix_patch = ""
            cve = data['instance_id']
            image_name = f"ghcr.io/anonymous2578-data/{cve}:0708"
            patch_datas.append(
                {
                    "image_name": image_name,
                    "fix_patch": fix_patch + '\n'
                }
            )
    utils.write_jsonl(patch_datas, args.patch_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--patch_file",
        type=str,
        required=True
    )
    args = parser.parse_args()
    main()