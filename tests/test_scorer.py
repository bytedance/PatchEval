"""Tests for the PatchEval scorer validation logic.

These tests mock the Inspect sandbox so they can exercise the scorer's
control flow without a running Docker daemon.
"""

from __future__ import annotations

import patcheval_inspect.scorer as scorer_mod
from patcheval_inspect.scorer import (
    _Cmd,
    _collect_patch,
    _read_patch,
    _validate,
)


class FakeSandbox:
    """Records writes/reads and replays scripted command results by substring."""

    def __init__(
        self,
        results: dict[str, _Cmd],
        default: _Cmd,
        files: dict[str, str] | None = None,
    ) -> None:
        self._results = results
        self._default = default
        self._files = files or {}
        self.written: dict[str, str] = {}
        self.commands: list[str] = []

    async def write_file(self, path: str, contents: str) -> None:
        self.written[path] = contents

    async def read_file(self, path: str, text: bool = True):  # noqa: ANN001
        if path not in self._files:
            raise FileNotFoundError(path)
        return self._files[path]

    async def exec(self, cmd, timeout=None, **kwargs):  # noqa: ANN001 - test shim
        command = cmd[-1]
        self.commands.append(command)
        for needle, result in self._results.items():
            if needle in command:
                return result
        return self._default


def _install(monkeypatch, default: FakeSandbox, evaluator: FakeSandbox | None = None) -> None:
    evaluator = evaluator or default
    monkeypatch.setattr(
        scorer_mod,
        "sandbox",
        lambda name=None: evaluator if name == "evaluator" else default,
    )


def _ok() -> _Cmd:
    return _Cmd(0, "", "")


async def _run(**kwargs):
    defaults = dict(
        command_timeout=60,
    )
    defaults.update(kwargs)
    return await _validate("diff --git a b", **defaults)


async def test_read_patch_reads_from_agent_sandbox(monkeypatch):
    fake = FakeSandbox({}, _ok(), files={"/workspace/fix.patch": "diff --git a b\n"})
    _install(monkeypatch, fake)

    patch, err = await _read_patch()

    assert err == ""
    assert patch is not None and patch.startswith("diff --git")


async def test_read_patch_missing_file(monkeypatch):
    fake = FakeSandbox({}, _ok(), files={})
    _install(monkeypatch, fake)

    patch, err = await _read_patch()

    assert patch is None
    assert "not found" in err


async def test_read_patch_empty_file(monkeypatch):
    fake = FakeSandbox({}, _ok(), files={"/workspace/fix.patch": "   \n"})
    _install(monkeypatch, fake)

    patch, err = await _read_patch()

    assert patch is None
    assert "empty" in err


async def test_collect_patch_prefers_fix_patch(monkeypatch):
    fake = FakeSandbox({}, _ok(), files={"/workspace/fix.patch": "diff --git a/fix b/fix\n"})
    _install(monkeypatch, fake)

    patch, source, err = await _collect_patch("/workspace/repo", 60)

    assert err == ""
    assert source == "fix_patch"
    assert patch is not None and "a/fix" in patch


async def test_collect_patch_falls_back_to_git_diff(monkeypatch):
    fake = FakeSandbox({}, _Cmd(0, "diff --git a/git b/git\n", ""), files={})
    _install(monkeypatch, fake)

    patch, source, err = await _collect_patch("/workspace/repo", 60)

    assert err == ""
    assert source == "git_diff"
    assert patch is not None and "a/git" in patch
    assert any("git diff HEAD -U3" in c for c in fake.commands)


async def test_successful_repair(monkeypatch):
    default = FakeSandbox({}, _ok())
    evaluator = FakeSandbox({}, _ok())
    _install(monkeypatch, default, evaluator)

    value, explanation, metadata = await _run()

    assert value == "C"
    assert metadata["status"] == "repair_success"
    assert evaluator.written[scorer_mod.PATCH_PATH].startswith("diff --git")
    commands = evaluator.commands
    assert any("fix-run.sh" in c for c in commands)


async def test_evaluator_write_failure(monkeypatch):
    default = FakeSandbox({}, _ok())
    evaluator = FakeSandbox({}, _ok())

    async def fail_write(path: str, contents: str) -> None:
        raise FileNotFoundError(path)

    evaluator.write_file = fail_write
    _install(monkeypatch, default, evaluator)

    value, _explanation, metadata = await _run()

    assert value == "I"
    assert metadata["status"] == "sandbox_error"


async def test_poc_failure_classified(monkeypatch):
    default = FakeSandbox({}, _ok())
    evaluator = FakeSandbox({"fix-run.sh": _Cmd(1, "", "error: patch does not apply")}, _ok())
    _install(monkeypatch, default, evaluator)

    value, _explanation, metadata = await _run()

    assert value == "I"
    assert metadata["status"] == "validation_fail"


async def test_unit_tests_are_not_run(monkeypatch):
    # Unit-test logic has been removed: validation stops at the PoC command and
    # unit_test.sh must never be invoked.
    default = FakeSandbox({}, _ok())
    evaluator = FakeSandbox({}, _ok())
    _install(monkeypatch, default, evaluator)

    value, _explanation, metadata = await _run()

    assert value == "C"
    assert metadata["status"] == "repair_success"
    assert "unit_exit_code" not in metadata
    assert not any("unit_test.sh" in c for c in evaluator.commands)
