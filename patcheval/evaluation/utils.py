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
import logging
from pathlib import Path

class CveContextFilter(logging.Filter):
    def __init__(self, cve_id: str):
        super().__init__()
        self.cve_id = cve_id

    def filter(self, record):
        record.cve = self.cve_id
        return True


def read_json(path):
    with open(path) as fr:
        datas=json.load(fr)
    return datas

def read_jsonl(path):
    datas = []
    with open(path) as f:
        for line in f:
            datas.append(
                json.loads(line)
            )
    return datas

def creat_patch_file(prefix, patch):
    path = f"{prefix}/fix.patch"
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)

    with open(path, 'w') as f:
        f.write(patch)
    path = Path(path)
    absolute_path = path.resolve(strict=False)
    return absolute_path, parent_dir

def get_logger(log_path, log_level=logging.INFO):
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    logger = logging.getLogger('main_logger')
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(cve)s] - %(message)s', defaults={'cve': 'GENERAL'})
        
        handler = logging.FileHandler(log_path, mode='w')
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def convert_json(json_file, epoch=1):
    origin_datas = read_json(json_file)
    eval_datas = []
    for origin_data in origin_datas:
        for cve, items in origin_data.items():
            for item in items:
                if item['epoch'] == epoch and item['diff_content'] is not None:
                    eval_datas.append({
                        "cve": cve,
                        "fix_patch": item['diff_content'],
                    })
                    break
    return eval_datas