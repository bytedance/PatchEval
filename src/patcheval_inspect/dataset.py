"""Dataset loading utilities for the PatchEval Inspect evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai.dataset import Sample

DATA_PATH = Path(__file__).resolve().parent / "data" / "patcheval_verified.json"
TEMPLATE_DIR = Path(__file__).resolve().parent / "task_template"


def load_patcheval_records() -> list[dict[str, str]]:
    """Load PatchEval 230-CVE records from a JSON file."""

    with DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def record_to_sample(
    record: dict[str, str],
) -> Sample:
    """Convert one PatchEval record into an Inspect ``Sample``."""

    cve_id = record["cve_id"]
    description = record["description"]
    image_url = record["image_url"]
    workdir = record["workdir"]

    prompt = f"""## USER

Please fix the vulnerabilities in the code repository based on the following information:{description.strip()}

Task runtime information:
- Target workdir: {workdir}
- All tool path arguments must stay under this directory.
- Start exploration from this workspace root instead of guessing a path under /workspace.
- Before stopping, write the final repository diff to `/workspace/fix.patch` from the target workdir.

Repair-source restrictions:
- Do not search the web for this vulnerability, CVE, advisory, GHSA, release note, issue, pull request, or upstream patch.
- Do not run network commands such as curl, wget, git fetch, git pull, git ls-remote, npm view, pip index, or package/advisory lookups to find the fix.
"""

    return Sample(
        id=cve_id,
        input=prompt,
        # Inspect accuracy treats a score of "C" as correct. The scorer
        # returns "C" only when Docker validation succeeds.
        target="C",
        metadata={
            "cve_id": cve_id,
            "description": description,
            "image_url": image_url,
            "workdir": workdir,
        },
        sandbox=("docker", str((TEMPLATE_DIR / "compose.yml").resolve())),
    )


def load_patcheval_samples() -> list[Sample]:
    """Load all PatchEval records as Inspect samples."""

    return [record_to_sample(record) for record in load_patcheval_records()]
