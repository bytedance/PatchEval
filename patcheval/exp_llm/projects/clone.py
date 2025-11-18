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
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def _parse_repo(url):
    parts = url.strip().split('/')
    if len(parts) < 5:
        return None, None
    owner = parts[3]
    repo = parts[4]
    # repo_url = f"git@github.com:{owner}/{repo}.git"
    repo_url = f"https://github.com/{owner}/{repo}.git"
    repo_dir = repo
    return repo_url, repo_dir

def _clone(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        print(f"Cloning {repo_url} ...")
        subprocess.run(['git', 'clone', repo_url, repo_dir])
    else:
        print(f"Repo {repo_dir} already exists, skipping clone.")

def main():
    txt_file = 'repos.txt'
    print(f"Processing repos in {txt_file} !!!")
    tasks = []
    with open(txt_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            repo_url, repo_dir = _parse_repo(line)
            print(f"repo_url: {repo_url} repo_dir: {repo_dir}")
            if repo_url:
                tasks.append((repo_url, repo_dir))
    
    max_workers = min(8, len(tasks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_repo = {executor.submit(_clone, repo_url, repo_dir): repo_url for repo_url, repo_dir in tasks}
        for future in as_completed(future_to_repo):
            repo_url = future_to_repo[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {repo_url}: {e}")

if __name__ == '__main__':
    main()