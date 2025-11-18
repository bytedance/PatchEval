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
MODELNAME=$1

run(){
    local local_model_name=$1
    prefix=${local_model_name}_exp1
    template_path=./configs/template_${local_model_name}_without_feedback.yaml

    target_dir=./configs/$prefix

    python construct_dataset.py --prefix $prefix --template_path $template_path --data_path dataset.jsonl 

    ### Run SWEagent
    target_dir=./configs/$prefix
    output_dir=./SWE-agent/trajectories/$prefix

    max_processes=6
    counter=0  

    # rm -rf $output_dir
    for file in "$target_dir"/*.yaml; do
        
        
        filename=$(basename "$file" .yaml)
        # echo $file
        if [[ "$filename" == CVE* ]]; then
            mkdir -p ${output_dir}/$filename

            if [ -d "${output_dir}/$filename" ] && find "${output_dir}/$filename" -name "*.patch" -print -quit | grep -q .; then
                echo skip "${output_dir}/$filename"
                continue
            fi

            echo ${output_dir}/$filename
            rm -rf ${output_dir}/$filename
            mkdir ${output_dir}/$filename
            sweagent run \
            --config SWE-agent/config/default.yaml \
            --config ${file} \
            --output_dir ${output_dir}/$filename  \
            > "${output_dir}/$filename/output.log" 2>&1 &
            
            ((counter++))
        fi
        if (( counter >= max_processes )); then
            wait -n 
            ((counter--))  
        fi

    done
    wait
    bash ./shells/run_eval.sh $prefix
}

models=(
    $MODELNAME
)

for model_name in ${models[@]}; do
    run $model_name &
done

wait