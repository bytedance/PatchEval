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
    echo "Caught Ctrl-C, killing all background processes..."
    pkill -P $$
    exit 1
}

trap cleanup SIGINT

# MODEL_CONFIG=claud37
EVAL_LIMIT=300
MAX_ITER=100
NUM_WORKERS=8

# 2. w.o. loc & w. knowledge & w. feedback & w.o test func (3)
AGENT=CodeActAgent_w_test_wo_func
custom_dataset=dataset_nolocation.json
prefix=exp3
prompt=w_test_wo_func.j2
custom_command="mkdir -p /tmp/secret && cp /workspace/test.patch /tmp/secret/test.patch && rm /workspace/test.patch"

run_and_eval(){
    local model=$1
    echo ./evaluation/evaluation_outputs/${model}_${prefix}_run_1_maxiter_${MAX_ITER}/output.jsonl
    ## run infer
    bash shells/run_infer.sh $model $AGENT $EVAL_LIMIT $MAX_ITER $NUM_WORKERS $custom_dataset $prefix $prompt "$custom_command"

    # run evaluation
    python evaluation/process_output.py \
        --output_file ./evaluation/evaluation_outputs/${model}_${prefix}_run_1_maxiter_${MAX_ITER}/output.jsonl \
        --patch_file ./evaluation/process_datas/${model}_${prefix}.jsonl
    
    python ../../evaluation/run_evaluation.py \
        --output results/${model}_${prefix} \
        --patch_file ./evaluation/process_datas/${model}_${prefix}.jsonl \
        --input_file ../../datasets/input.json


}

models=("******") #your model
for model_name in "${models[@]}"; do
    echo "$model_name", exp3
    mkdir -p "./evaluation/evaluation_outputs/${model_name}_${prefix}_run_1_maxiter_${MAX_ITER}"
    run_and_eval "$model_name" > "./evaluation/evaluation_outputs/${model_name}_${prefix}_run_1_maxiter_${MAX_ITER}/log.txt" 2>&1 
done
wait