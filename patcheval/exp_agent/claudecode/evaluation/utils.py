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
from pathlib import Path
def load_jsonl_file(path):
    datas = []
    with open(path) as f:
        for line in f:
            datas.append(
                json.loads(line)
            )
    return datas

def write_jsonl(datas, path):
    with open(path, 'w') as f:
        for data in datas:
            f.write(json.dumps(data) + '\n')

def creat_patch_file(prefix, patch):
    path = f"./evaluation/{prefix}/fix.patch"
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)

    with open(path, 'w') as f:
        f.write(patch)
    path = Path(path)
    absolute_path = path.resolve(strict=False)
    return absolute_path, parent_dir