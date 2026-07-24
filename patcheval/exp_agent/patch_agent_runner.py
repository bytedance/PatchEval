#!/usr/bin/env python3
"""Generate patches for PatchEval cases with a configurable CLI agent.

This is a patch-generation-only runner. Evaluation is handled separately by
../evaluation/run_evaluation.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_AGENT_TIMEOUT_S = 3600
STREAM_READER_LIMIT = 16 * 1024 * 1024


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


@dataclass
class GenerationResult:
    index: int
    cve: str
    instance_id: str
    image: str
    workdir: str
    container_name: str
    status: str
    patch_generated: bool
    agent_exit_code: Optional[int]
    timed_out: bool
    duration_s: float
    patch_path: str
    error: str = ""


def _safe_name(value: str, max_len: int = 100) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return (value.strip(".-") or "sample")[:max_len]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _require_dataset_images(samples: list[dict[str, Any]]) -> None:
    missing: list[str] = []
    for sample in samples:
        cve = str(sample["cve_id"])
        if not sample.get("image_url"):
            missing.append(cve)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(f"dataset samples missing image_url: {preview}{suffix}")


def _image_url(sample: dict[str, Any]) -> str:
    image = str(sample.get("image_url") or "").strip()
    if not image:
        raise ValueError(f"dataset sample {sample.get('cve_id', '<unknown>')} missing image_url")
    return image


def _repo_basename(repo_url: str) -> str:
    name = repo_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def _prompt(sample: dict[str, Any], workdir_hint: str) -> str:
    return (
        "## USER\n\n"
        "Please fix the vulnerabilities in the code repository based on the following information:"
        + str(sample.get("cve_description") or "").strip()
        + "\n\n"
        + "Task runtime information:\n"
        + "- Target workdir: "
        + workdir_hint
        + "\n"
        + "- All tool path arguments must stay under this directory.\n"
        + "- Start exploration from this workspace root instead of guessing a path under /workspace.\n"
        + "- Before stopping, write the final repository diff to `/workspace/fix.patch` from the target workdir."
        + "\n\n"
        + "Repair-source restrictions:\n"
        + "- Do not search the web for this vulnerability, CVE, advisory, GHSA, release note, issue, pull request, or upstream patch.\n"
        + "- Do not run network commands such as curl, wget, git fetch, git pull, git ls-remote, npm view, pip index, or package/advisory lookups to find the fix.\n"
    )


async def _run(args: list[str], *, timeout_s: Optional[int] = None) -> CommandResult:
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=STREAM_READER_LIMIT,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        timed_out = False
    except asyncio.TimeoutError:
        proc.kill()
        out_b, err_b = await proc.communicate()
        timed_out = True
    return CommandResult(
        command=args,
        exit_code=int(proc.returncode if proc.returncode is not None else 124),
        stdout=(out_b or b"").decode("utf-8", errors="replace"),
        stderr=(err_b or b"").decode("utf-8", errors="replace"),
        duration_s=time.monotonic() - started,
        timed_out=timed_out,
    )


async def _docker_exec(container: str, command: str, *, workdir: str = "/", timeout_s: int = 600) -> CommandResult:
    args = ["docker", "exec", "-w", workdir, container, "bash", "-lc", command]
    return await _run(args, timeout_s=timeout_s)


def _parse_mounts(items: list[str]) -> list[tuple[str, str, str]]:
    mounts = []
    for item in items:
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"expected HOST:CONTAINER[:ro|rw], got: {item}")
        mode = parts[2] if len(parts) == 3 else "rw"
        if mode not in {"ro", "rw"}:
            raise ValueError(f"invalid mount mode: {item}")
        mounts.append((str(Path(parts[0]).expanduser().resolve()), parts[1], mode))
    return mounts


def _docker_run_args(container: str, image: str, result_dir: Path, mounts: list[tuple[str, str, str]]) -> list[str]:
    cmd = ["docker", "run", "-d", "--name", container, "-v", f"{result_dir.resolve()}:/results:rw"]
    for host, dst, mode in mounts:
        cmd.extend(["-v", f"{host}:{dst}:{mode}"])
    cmd.extend([image, "bash", "-lc", "tail -f /dev/null"])
    return cmd


async def _detect_workdir(container: str, sample: dict[str, Any]) -> str:
    workdir_hint = str(sample.get("workdir") or "").rstrip("/")
    if workdir_hint:
        check = await _docker_exec(container, f"test -d {shlex.quote(workdir_hint)}/.git", timeout_s=60)
        if check.exit_code == 0:
            return workdir_hint
        raise RuntimeError(f"dataset workdir is not a git repository in image: {workdir_hint}")

    repo_name = _repo_basename(str(sample.get("repo") or ""))
    if repo_name:
        check = await _docker_exec(container, f"test -d /workspace/{shlex.quote(repo_name)}/.git", timeout_s=60)
        if check.exit_code == 0:
            return f"/workspace/{repo_name}"
        lower_repo_name = repo_name.lower()
        if lower_repo_name != repo_name:
            check = await _docker_exec(container, f"test -d /workspace/{shlex.quote(lower_repo_name)}/.git", timeout_s=60)
            if check.exit_code == 0:
                return f"/workspace/{lower_repo_name}"

    check = await _docker_exec(container, "test -d /workspace/.git", timeout_s=60)
    if check.exit_code == 0:
        return "/workspace"

    if repo_name:
        raise RuntimeError(
            f"could not locate git workdir for repo {repo_name!r}; "
            "check that dataset image_url matches this CVE or add dataset workdir"
        )

    find_repo = await _docker_exec(
        container,
        "find /workspace -mindepth 2 -maxdepth 3 -type d -name .git 2>/dev/null | head -n 2",
        timeout_s=60,
    )
    candidates = [line.rsplit("/.git", 1)[0] for line in find_repo.stdout.splitlines() if line.strip()]
    if len(candidates) == 1:
        return candidates[0]

    return "/workspace"


async def _hide_workspace_payload(container: str, workdir: str, session_key: str) -> None:
    if workdir.rstrip("/") == "/workspace":
        result = await _docker_exec(container, "rm -f /workspace/fix.patch", timeout_s=300)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or result.stdout)
        return

    if not workdir.startswith("/workspace/"):
        raise RuntimeError(f"target workdir is outside /workspace: {workdir}")

    script = f"""
set -e
rm -f /workspace/fix.patch
mkdir -p /tmp/{_safe_name(session_key)}
workdir={shlex.quote(workdir)}
top_name=${{workdir#/workspace/}}
top_name=${{top_name%%/*}}
find /workspace -mindepth 1 -maxdepth 1 ! -name "$top_name" -exec mv -t /tmp/{_safe_name(session_key)} -- {{}} + 2>/dev/null || true
"""
    result = await _docker_exec(container, script, timeout_s=300)
    if result.exit_code != 0:
        raise RuntimeError(result.stderr or result.stdout)


async def _run_agent(container: str, workdir: str, command_template: str, result_dir: Path, env: dict[str, str], timeout_s: int) -> CommandResult:
    values = {
        "prompt_file": "/results/prompt.txt",
        "workdir": workdir,
    }
    command = command_template.format(**values)
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "-w", workdir,
        *sum((["-e", f"{k}={v}"] for k, v in env.items()), []),
        container, "bash", "-lc", command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=STREAM_READER_LIMIT,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        timed_out = False
    except asyncio.TimeoutError:
        proc.kill()
        out_b, err_b = await proc.communicate()
        timed_out = True
    stdout = (out_b or b"").decode("utf-8", errors="replace")
    stderr = (err_b or b"").decode("utf-8", errors="replace")
    (result_dir / "agent_stdout.txt").write_text(stdout, encoding="utf-8")
    (result_dir / "agent_stderr.txt").write_text(stderr, encoding="utf-8")
    return CommandResult(["docker", "exec", container, "bash", "-lc", command], int(proc.returncode if proc.returncode is not None else 124), stdout, stderr, time.monotonic() - started, timed_out)


async def _collect_patch(container: str, workdir: str, result_dir: Path) -> CommandResult:
    script = f"""
set -e
cd {shlex.quote(workdir)}
if [ -s /workspace/fix.patch ]; then
  cp /workspace/fix.patch /results/llm.patch
elif git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff HEAD -U3 > /results/llm.patch
else
  echo "target workdir is not a git repository: $(pwd)" >&2
  exit 1
fi
test -s /results/llm.patch
"""
    return await _docker_exec(container, script, workdir=workdir, timeout_s=300)


async def _remove_container(container: str) -> None:
    await _run(["docker", "rm", "-f", container], timeout_s=120)


async def _run_one(sample: dict[str, Any], index: int, args: argparse.Namespace, sem: asyncio.Semaphore, mounts: list[tuple[str, str, str]]) -> GenerationResult:
    async with sem:
        started = time.monotonic()
        cve = str(sample["cve_id"])
        instance_id = f"patcheval_{cve}"
        image = _image_url(sample)
        run_id = f"{index:05d}-{instance_id}"
        container = f"{args.container_prefix}-{_safe_name(run_id)}-{os.getpid()}"
        run_root = Path(args.run_root)
        work = run_root / ".work" / run_id
        work.mkdir(parents=True, exist_ok=True)
        patch_path = run_root / "patches" / f"{cve}.patch"
        status = "failed"
        error = ""
        workdir = "/workspace"
        agent_result: Optional[CommandResult] = None
        timed_out = False
        try:
            env = {"PATCHAGENT_SESSION_ID": container}
            run_result = await _run(_docker_run_args(container, image, work, mounts), timeout_s=1200)
            if run_result.exit_code != 0:
                raise RuntimeError(f"docker run failed: {run_result.stderr or run_result.stdout}")
            workdir = await _detect_workdir(container, sample)
            await _hide_workspace_payload(container, workdir, container)
            prompt = _prompt(sample, workdir)
            (work / "prompt.txt").write_text(prompt, encoding="utf-8")
            agent_result = await _run_agent(container, workdir, args.agent_command, work, env, args.agent_timeout)
            timed_out = agent_result.timed_out
            if agent_result.exit_code != 0:
                raise RuntimeError(f"agent failed with exit_code={agent_result.exit_code}")
            collect = await _collect_patch(container, workdir, work)
            if collect.exit_code != 0:
                raise RuntimeError(f"collect patch failed: {collect.stderr or collect.stdout}")
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "llm.patch", patch_path)
            status = "generated"
        except Exception as exc:
            error = str(exc)
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text("", encoding="utf-8")
            _log(f"{run_id}: failed: {error}")
        finally:
            await _remove_container(container)
        result = GenerationResult(index, cve, instance_id, image, workdir, container, status, status == "generated", agent_result.exit_code if agent_result else None, timed_out, time.monotonic() - started, str(patch_path), error)
        return result


async def _main(args: argparse.Namespace) -> int:
    samples = _read_json(Path(args.input))
    indexed = list(enumerate(samples))
    selected = indexed if args.limit < 0 else indexed[:args.limit]
    _require_dataset_images([sample for _, sample in selected])
    run_root = Path(args.output_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}-{_safe_name(args.run_label or 'run')}"
    for sub in ["patches", ".work"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    args.run_root = str(run_root)
    _write_json(run_root / "run_metadata.json", vars(args) | {"run_root": str(run_root), "total_cases": len(selected)})
    mounts = _parse_mounts(args.mount)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(_run_one(sample, idx, args, sem, mounts)) for idx, sample in selected]
    results_path = run_root / "results.jsonl"
    results = []
    with results_path.open("w", encoding="utf-8") as f:
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            results.append(result)
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            f.flush()
            _log(f"progress {i}/{len(tasks)}: {result.cve} {result.status}")
    generated = sum(r.patch_generated for r in results)
    _write_json(run_root / "summary.json", {"total": len(results), "generated": generated, "failed": len(results) - generated})
    print(f"Run directory: {run_root}")
    print(f"Generated patches: {generated}/{len(results)}")
    return 0 if generated == len(results) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="../datasets/patcheval_verified.json")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--run-label", default="")
    p.add_argument("--agent-command", required=True)
    p.add_argument("--mount", action="append", default=[])
    p.add_argument("--agent-timeout", type=int, default=DEFAULT_AGENT_TIMEOUT_S)
    p.add_argument("--container-prefix", default="patcheval-agent")
    return p


def main() -> int:
    return asyncio.run(_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
