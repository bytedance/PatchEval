"""PatchEval scorer for Inspect AI.

The agent writes its patch to ``/workspace/fix.patch`` in the default sandbox.
The scorer reads that file first; if it is missing or empty, the scorer falls
back to ``git diff HEAD -U3`` from the target worktree, matching SecPatch patch
collection.

Validation writes the collected patch to ``/workspace/fix.patch`` in the
``evaluator`` sandbox and then runs the same in-image PoC command from the
PatchEval/SecPatch evaluator.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

# Where the patch is placed for validation.
PATCH_PATH = "/workspace/fix.patch"
TEST_CMD = "cd /workspace && bash fix-run.sh"


@dataclass
class _Cmd:
    """Outcome of a single validation command."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _format(cmd: str, res: _Cmd) -> str:
    out = f"$ {cmd}\n[exit_code] {res.returncode}\n"
    if res.stdout:
        out += f"[stdout]\n{res.stdout}\n"
    if res.stderr:
        out += f"[stderr]\n{res.stderr}\n"
    if res.returncode == 124:
        out += "[timeout]\n"
    return out


async def _read_patch() -> tuple[str | None, str]:
    """Read the fixed patch file from the sandbox."""

    try:
        contents = await sandbox().read_file(PATCH_PATH, text=True)
    except FileNotFoundError:
        return None, f"Patch file not found at {PATCH_PATH!r} in the sandbox."

    if not contents.strip():
        return None, f"Patch file {PATCH_PATH!r} is empty."
    return contents, ""


async def _collect_git_diff_patch(workdir: str, timeout: int) -> tuple[str | None, str]:
    """Collect a patch from the agent-modified worktree."""

    cmd = f"cd {shlex.quote(workdir)} && git diff HEAD -U3"
    try:
        res = await sandbox().exec(["bash", "-lc", cmd], timeout=timeout)
    except TimeoutError:
        return None, f"git diff patch collection timed out after {timeout}s"
    if res.returncode != 0:
        return None, _format(cmd, _Cmd(res.returncode, res.stdout, res.stderr))
    if not res.stdout.strip():
        return None, "git diff HEAD -U3 produced an empty patch."
    return res.stdout, ""


async def _collect_patch(workdir: str, timeout: int) -> tuple[str | None, str, str]:
    """Collect the patch with SecPatch priority: fix.patch, then git diff."""

    patch, fix_patch_error = await _read_patch()
    if patch is not None:
        return patch, "fix_patch", ""

    patch, git_diff_error = await _collect_git_diff_patch(workdir, timeout)
    if patch is not None:
        return patch, "git_diff", ""

    return (
        None,
        "none",
        "No patch collected from /workspace/fix.patch or git diff.\n"
        f"[fix.patch]\n{fix_patch_error}\n\n[git_diff]\n{git_diff_error}",
    )


async def _exec(cmd: str, timeout: int, *, sandbox_name: str = "evaluator") -> _Cmd:
    """Run a bash command in an Inspect sandbox."""

    try:
        res = await sandbox(sandbox_name).exec(["bash", "-lc", cmd], timeout=timeout)
        return _Cmd(res.returncode, res.stdout, res.stderr)
    except TimeoutError:
        return _Cmd(124, "", f"command timed out after {timeout}s")


async def _validate(
    patch: str,
    *,
    command_timeout: int,
) -> tuple[str, str, dict[str, object]]:
    """Apply and validate ``patch`` in the sandbox.

    The collected patch is written to /workspace/fix.patch in the evaluator
    sandbox, then the PoC command runs.

    Returns ``(value, explanation, metadata)`` where ``value`` is ``"C"`` for a
    successful repair and ``"I"`` otherwise.
    """

    logs: list[str] = []
    evaluator = sandbox("evaluator")
    try:
        await evaluator.write_file(PATCH_PATH, patch if patch.endswith("\n") else patch + "\n")
    except FileNotFoundError as exc:
        return (
            "I",
            f"Failed to write patch to {PATCH_PATH}: {exc}",
            {"status": "sandbox_error"},
        )

    poc = await _exec(TEST_CMD, command_timeout)
    logs.append(_format(TEST_CMD, poc))

    if not poc.ok:
        joined = "\n".join(logs)
        return (
            "I",
            "PoC validation failed.\n" + joined,
            {
                "status": "validation_fail",
                "poc_exit_code": poc.returncode,
            },
        )

    return (
        "C",
        "Patch passed PoC validation.\n" + "\n".join(logs),
        {
            "status": "repair_success",
            "poc_exit_code": poc.returncode,
        },
    )


@scorer(metrics=[accuracy()])
def patcheval_patch_scorer(
    command_timeout: int = 600,
) -> Scorer:
    """Score model patches with PatchEval validation in the evaluator sandbox.

    Returns ``C`` for a successful repair and ``I`` otherwise so Inspect's
    built-in accuracy metric can treat success as the correct answer.
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = str(state.output.completion).strip()

        patch, patch_source, read_error = await _collect_patch(
            state.metadata["workdir"],
            command_timeout,
        )
        if patch is None:
            return Score(
                value="I",
                answer=answer,
                explanation=read_error,
                metadata={"status": "no_patch", "patch_source": patch_source},
            )

        value, explanation, score_metadata = await _validate(
            patch,
            command_timeout=command_timeout,
        )
        return Score(
            value=value,
            answer=answer,
            explanation=explanation,
            metadata=score_metadata | {"patch_source": patch_source},
        )

    return score
