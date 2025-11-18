#!/bin/bash
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
cleanup() {
    kill -TERM -$$  
    exit 1
}

trap cleanup SIGINT
models=(
    "gemini-2.5-pro"
)
#####
prefix="_exp1"
dataset="dataset"
#####
# prefix="_exp4"
# dataset="dataset_nolocation"
#####
# prefix="_exp5"
# dataset="dataset_nolocation"


run(){
    modelname=$1

    if [ "$modelname" == "xxx" ]; then
        model="xxx"
        base_url="xxx"
        api_key="xxx"
        PORT=8083
    fi
    ##########################


    PID=$(lsof -i :$PORT -t)

    if [ -z "$PID" ]; then
        :
    else
        kill $PID
        sleep 1
        if ps -p $PID > /dev/null; then
            kill -9 $PID
        fi
    fi
    echo "start $modelname", key: $api_key, url: $base_url, api_version: $api_version, port: $PORT
    cd claude-code-proxy || { echo "can't go into claude-code-proxy"; exit 1; }
    # uv run claude-code-proxy 
    mkdir ../logs
    OPENAI_API_KEY=$api_key OPENAI_BASE_URL=$base_url BIG_MODEL=$model MIDDLE_MODEL=$model SMALL_MODEL=$model AZURE_API_VERSION=$api_version PORT=$PORT uv run claude-code-proxy > ../logs/${modelname}${prefix}.log 2>&1 &
    PROCESS_PID=$!
    echo "complete claude-code-proxy $PROCESS_PID"
    cd ..
    sleep 3
    if ! ps -p $PROCESS_PID > /dev/null; then
        exit 1
    fi
    mkdir -p ./outputs/${modelname}${prefix}
    MY_MODEL=$model ANTHROPIC_API_KEY="temp_key" python -m patcheval.cli batch \
        --tool-limits "total:100" \
        --max-cost-usd 1000 \
        --dataset ${dataset}.jsonl \
        --outputs-root ./outputs/${modelname}${prefix}_test \
        --save-process-logs \
        --strategy default \
        --max-workers 4 \
        --port $PORT \
        --allow-git-diff-fallback \
        --resume \
        --save-process-logs
        #  > ./outputs/${modelname}${prefix}/output.log 2>&1 

    if ps -p $PROCESS_PID > /dev/null; then
        kill $PROCESS_PID
        sleep 2
        if ps -p $PROCESS_PID > /dev/null; then
            kill -9 $PROCESS_PID
        fi
    fi
    # bash shells/run_eval.sh ${modelname}${prefix}

}

for use_model in "${models[@]}"; do

    run $use_model &

done

wait