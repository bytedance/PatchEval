
# Running OpenHands on PatchEval

## Step 1: Set Up the Environment
```bash
  git clone https://github.com/OpenHands/OpenHands.git
  cd Openhands/
  git checkout 10ae481b91716f4fa979f091cc2af22095c27e8a && patch -p1 -f < ../openhands_diff.patch
  ```
Follow the instructions in the [official OpenHands documentation](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md) to configure the environment, stopping at step 4 (`make run`).  
> Don’t forget to specify your model parameters in `config.toml` during step 3.

Next, install the evaluation dependencies listed in `pyproject.toml`:
```bash
poetry install --with evaluation
````

## Step 2: Run OpenHands and Evaluate

Execute the following command:

```bash
bash run_all_exp1.sh
```

