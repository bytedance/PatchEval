# Running ClaudeCode on PatchEval

## Step 1: Set Up the Environment
Install the software dependencies:  
```bash
cd patcheval/exp_agent/claudecode
pip install -r requirements.txt
```

## Step 2: Start the LLM Service (Skip if you have access to the Claude API)
You can use [claude-code-proxy](https://github.com/fuergaosi233/claude-code-proxy) to setup proxy for OpenAI API.  
```bash
git clone https://github.com/fuergaosi233/claude-code-proxy
cd claude-code-proxy 
git checkout b4a8777035313034af76b && patch -p1 -f < ../claude-code-proxy_diff.patch
cp .env.example .env
# Edit .env to add your model configuration (e.g., BIG_MODEL, MIDDLE_MODEL, SMALL_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL)
source .env
uv run claude-code-proxy
```

## Step 3: Run ClaudeCode and Evaluate
In the `shells/run_infer.sh` file, configure your model (same as in Step 2) and select the dataset you want to test (default is `exp1`).  
```bash
# patch generation
cd patcheval/exp_agent/claudecode
bash shells/run_infer.sh
# patch evaluation
bash shells/run_eval.sh doubao_exp1_test
```
