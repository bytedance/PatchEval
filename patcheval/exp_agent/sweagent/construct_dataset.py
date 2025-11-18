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
import argparse

def load_jsonl_file(path):
    datas = []
    with open(path) as f:
        for line in f:
            datas.append(
                json.loads(line)
            )
    return datas

JS_CVE = [
    'CVE-2016-10548', 'CVE-2017-16042', 'CVE-2017-16083', 'CVE-2018-16482',
    'CVE-2018-3733', 'CVE-2018-3734', 'CVE-2018-3772', 'CVE-2018-3785',
    'CVE-2019-10787', 'CVE-2019-10788', 'CVE-2019-15597', 'CVE-2020-28437',
    'CVE-2020-28494', 'CVE-2020-7613', 'CVE-2020-7627', 'CVE-2020-7631',
    'CVE-2020-7640', 'CVE-2020-7674', 'CVE-2020-7675', 'CVE-2020-7687',
    'CVE-2020-7781', 'CVE-2020-7795', 'CVE-2020-8132', 'CVE-2021-23363',
    'CVE-2021-23376', 'CVE-2017-16198'
]

environment_path = {
    "CVE-2021-3583": "/workspace/PoC_env/CVE-2021-3583/bin:/root/miniconda3/bin:/root/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CVE-2020-10691": "/workspace/PoC_env/CVE-2020-10691/bin:/root/miniconda3/bin:/root/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CVE-2023-30172": "/workspace/PoC_env/CVE-2023-30172/bin:/root/miniconda3/bin:/root/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CVE-2023-39660": "/workspace/PoC_env/CVE-2023-39660/bin:/root/miniconda3/bin:/root/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "CVE-2020-10691": "/workspace/PoC_env/CVE-2020-10691/bin:/root/miniconda3/bin:/root/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
}
ignrore = """
*.png
*.jpg
*.jpeg
*.gif
*.bmp
*.tiff
*.webp
*.mp3
*.mp4
*.avi
*.mov
*.flv
*.wmv
*.pdf
*.psd
*.ai

*.zip
*.tar
*.tar.gz
*.tar.bz2
*.7z
*.rar
*.gz
*.bz2

*.exe
*.dll
*.so
*.dylib
*.bin
*.out

*.db
*.sqlite
*.sqlite3

/build/
/dist/
/bin/
/out/


.DS_Store
Thumbs.db

# Go
myapp
vendor/
*.out
*.test
coverage.out
build/
dist/

# JavaScript/Node.js
node_modules/
dist/
build/
out/
dist-ssr/
*.bundle.js
*.bundle.js.map
*.chunk.js
*.chunk.js.map
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*
.env.local
.env.development.local
.env.test.local
.env.production.local
.node-gyp/
*.node

# Python
__pycache__/
*.py[cod]
*$py.class
venv/
env/
ENV/
*.venv
*.egg-info/
.installed.cfg
*.egg
dist/
build/
wheelhouse/
*.so
*.pyd
*.dll
.coverage
htmlcov/
.pytest_cache/

*.blk
*.idx
*.jar
*.md
*package-lock.json


""".replace("\n", "\\n")
temp_ace = ["CVE-2021-32701"]

def find_error_cve():
    dir_path =f"./evaluation/results/{args.prefix}/logs"
    error_cve = []
    if not os.path.exists(dir_path):
        return []
    for cve in os.listdir(dir_path):
        if cve == "CVE-2023-32309":
            continue
        if "erro_output.log" in os.listdir(f"{dir_path}/{cve}") and "sucess_output.log" not in os.listdir(f"{dir_path}/{cve}"):
            with open(f"{dir_path}/{cve}/erro_output.log") as f:
                text = f.read()
                if "patch does not apply" in text:
                    error_cve.append(cve)
    return error_cve

def main():
    data_path = args.data_path
    template_path= args.template_path
    prefix = args.prefix
    dataset = load_jsonl_file(data_path)
    template = open(template_path).read()
    os.makedirs(f"configs/{prefix}", exist_ok=True)
    erro_cves = find_error_cve()
    
    for data in dataset:
        # if not data['cve_id'] in temp_ace:
        #     continue
        if args.only_apply_false:
            if data['cve_id'] not in erro_cves:
                continue
        new_config_path = f"configs/{prefix}/{data['cve_id']}.yaml"
        content = template \
            .replace("$IMAGE", data['image_name']) \
            .replace("$WORK_DIR", data['work_dir']) \
            .replace("$CVEID", data['cve_id']) \
            .replace("$PREFIX", prefix) \
            .replace("$JS_POC", '(The PoC test will show a "pass" result when the target system is vulnerable.)' if data['cve_id'] in JS_CVE else "") \
            .replace("$ALL_OTHER_PATH", environment_path.get(data['cve_id'], "/usr/local/go/bin"))
        
        content = content \
            .replace("$EXTRA_COMMAND", f"echo -e \"{ignrore}\" >> .gitignore")
        with open(new_config_path, 'w') as f:
            f.write(content)

        new_ps_path = f"configs/{prefix}/{data['cve_id']}_problem.md"
        with open(new_ps_path, 'w') as f:
            f.write(data['problem_statement'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="dataset.jsonl"
    )
    parser.add_argument(
        "--template_path",
        type=str,
        required=True,
        default="./configs/template.yaml"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True
    )
    parser.add_argument('--only_apply_false', action='store_true')
    args = parser.parse_args()
    main()

