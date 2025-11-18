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
import utils
import os
from pathlib import Path
import argparse
def main():
    test_data = json.load(open(args.test_data_path))
    cve2language = {it['cve_id']: it["programing_language"] for it in test_data}
    dataset_path = args.dataset_path
    dataset = utils.load_jsonl_file(dataset_path)
    output_dir = args.output_dir
    process_data_path = args.process_data_path
    erro_cve = []
    process_data = []
    for data in dataset:
        cve_id, image_name = data['cve_id'], data['image_name']
        patch_path = f"{output_dir}/{cve_id}.patch"

        if not os.path.exists(patch_path) or os.path.isdir(patch_path):
            
            erro_cve.append(cve_id)
            fix_patch = ""
        else:
            with open(patch_path) as f:
                fix_patch = f.read()
        # if cve_id.upper() not in cve2language:
            # continue
        process_data.append(
            {
                "cve": cve_id.upper(),
                "language": cve2language[cve_id.upper()],
                'fix_patch': fix_patch
            }
        )
    Path(process_data_path).parent.mkdir(parents=True, exist_ok=True)
    utils.write_jsonl(process_data, process_data_path)
    print(erro_cve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--process_data_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default="dataset.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True
    )
    args = parser.parse_args()
    main()