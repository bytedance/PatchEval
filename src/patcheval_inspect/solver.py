"""ReAct agent solver for PatchEval Inspect tasks."""

from __future__ import annotations

import shlex

from inspect_ai.agent import Agent, AgentAttempts, AgentSubmit, react
from inspect_ai.scorer import Value
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import bash, python
from inspect_ai.util import sandbox

# Path (in the agent's sandbox) where the agent writes its final patch. The
# agent submits this path rather than the diff text itself, so the answer stays
# a short, JSON-safe string and the diff never has to survive tool-call JSON
# encoding. The scorer reads this file from the agent sandbox.
DEFAULT_PATCH_PATH = "/workspace/fix.patch"
DEFAULT_SYSTEM_MESSAGE = f"""
Use the shell/Python tools to edit the target repository. Start from the target
workdir in the user prompt. When finished, submit only the path
`{DEFAULT_PATCH_PATH}` as the answer.
"""

DEFAULT_CONTINUE_MESSAGE = (
    f"If your fix is ready, write the final diff to `{DEFAULT_PATCH_PATH}` "
    f"and call `submit` with that path as the `answer`. "
    f"Otherwise, keep working with the shell tools."
)

DEFAULT_INCORRECT_MESSAGE = f"""
Your submitted patch did not pass validation. Use the validation feedback to
continue debugging, rewrite the corrected diff to `{DEFAULT_PATCH_PATH}`, then
call `submit` again with that path as the `answer`.
"""


@solver
def hide_evaluator_assets_setup() -> Solver:
    """Hide evaluator assets from the agent sandbox before repair starts."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        workdir = state.metadata["workdir"]
        script = f"""
set -e
rm -f {DEFAULT_PATCH_PATH}
hidden=/tmp/patcheval_hidden_assets
rm -rf "$hidden"
mkdir -p "$hidden"
repo_name=$(basename {shlex.quote(workdir)})
find /workspace -mindepth 1 -maxdepth 1 ! -name "$repo_name" -exec mv -t "$hidden" -- {{}} + 2>/dev/null || true
"""
        await sandbox().exec(["bash", "-lc", script], timeout=300)
        return state

    return solve


def patch_score_value(value: Value) -> float:
    """Map PatchEval scorer values onto ReAct attempt success.

    The PatchEval scorer returns "C" for a successful repair and "I" for an
    unsuccessful repair, matching Inspect's standard correct/incorrect values.
    """

    return 1.0 if value == "C" else 0.0


def patcheval_react_solver(
    *,
    max_attempts: int = 1,
    tool_timeout: int = 180,
) -> Agent:
    """Create the default PatchEval ReAct repair agent.

    The task provides a ReAct agent with shell/Python tools, and scorer
    feedback can trigger additional attempts. The task prompt is supplied by
    the dataset sample.
    """

    return react(
        prompt=DEFAULT_SYSTEM_MESSAGE,
        tools=[bash(timeout=tool_timeout), python(timeout=tool_timeout)],
        attempts=AgentAttempts(
            attempts=max_attempts,
            incorrect_message=DEFAULT_INCORRECT_MESSAGE,
            score_value=patch_score_value,
        ),
        on_continue=DEFAULT_CONTINUE_MESSAGE,
        # answer_only so state.output.completion is exactly the submitted path,
        # not the path appended to the model's chat content.
        submit=AgentSubmit(answer_only=True),
    )
