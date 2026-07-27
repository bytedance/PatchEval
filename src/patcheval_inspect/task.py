"""Inspect AI task definitions for PatchEval."""

from __future__ import annotations

from inspect_ai import Task, task

from patcheval_inspect.dataset import load_patcheval_samples
from patcheval_inspect.scorer import patcheval_patch_scorer
from patcheval_inspect.solver import hide_evaluator_assets_setup, patcheval_react_solver


@task
def patcheval_verified(
    # Solver configuration
    max_attempts: int = 1,
    tool_timeout: int = 180,
    # Scorer configuration
    command_timeout: int = 600,
) -> Task:
    """PatchEval verified Dockerized vulnerability repair subset.

    The default solver is a ReAct repair agent with shell/Python tools. The
    agent should inspect and modify the vulnerable repository, then submit a
    unified diff patch. The scorer validates submitted patches in a clean
    evaluator sandbox by running fix-run.sh.
    """

    return Task(
        dataset=load_patcheval_samples(),
        setup=hide_evaluator_assets_setup(),
        solver=patcheval_react_solver(
            max_attempts=max_attempts,
            tool_timeout=tool_timeout,
        ),
        scorer=patcheval_patch_scorer(command_timeout=command_timeout),
    )
