"""Tests for PatchEval dataset loading and metadata."""

from __future__ import annotations

from patcheval_inspect.dataset import (
    load_patcheval_records,
    load_patcheval_samples,
    record_to_sample,
)


def test_records_load_and_count() -> None:
    records = load_patcheval_records()
    assert isinstance(records, list)
    assert len(records) == 230
    for record in records:
        assert set(record) == {"cve_id", "description", "image_url", "workdir"}
        assert all(record.values())


def test_samples_match_record_count() -> None:
    samples = load_patcheval_samples()
    assert len(samples) == 230


def test_record_to_sample_basic() -> None:
    record = {
        "cve_id": "CVE-2021-23376",
        "description": "example vulnerability",
        "image_url": "ghcr.io/patcheval-cve/patcheval-cve:cve-2021-23376",
        "workdir": "/workspace/ffmpegdotjs",
    }

    sample = record_to_sample(record)

    # The scorer maps a successful repair onto "C", which Inspect treats as the
    # correct answer.
    assert sample.target == "C"
    assert sample.id == "CVE-2021-23376"
    assert sample.metadata["cve_id"] == "CVE-2021-23376"
    assert sample.metadata["description"] == record["description"]
    assert sample.metadata["image_url"] == record["image_url"]
    assert sample.metadata["workdir"] == record["workdir"]
    # Inspect uppercases metadata keys for compose interpolation, so
    # image_url resolves as ${SAMPLE_METADATA_IMAGE_URL}.
    assert "IMAGE_URL" not in sample.metadata
    assert "test_cmd" not in sample.metadata
    assert "oracle_cmd" not in sample.metadata
    assert "programming_language" not in sample.metadata
    assert "repo" not in sample.metadata
    assert "cwe_ids" not in sample.metadata
    # The sample runs in the docker sandbox defined by task_template/compose.yml.
    assert sample.sandbox is not None
    assert sample.sandbox.type == "docker"


def test_user_prompt_matches_runtime_prompt() -> None:
    record = {
        "cve_id": "CVE-2021-23376",
        "image_url": "ghcr.io/patcheval-cve/patcheval-cve:cve-2021-23376",
        "workdir": "/workspace/ffmpegdotjs",
        "description": "example vulnerability",
    }
    sample = record_to_sample(record)
    prompt = sample.input if isinstance(sample.input, str) else str(sample.input)
    assert prompt.startswith("## USER\n\nPlease fix")
    assert "CVE-2021-23376" not in prompt
    assert "JavaScript" not in prompt
    assert "CWE-78" not in prompt
    assert "/workspace/ffmpegdotjs" in prompt
    assert "example vulnerability" in prompt
    assert "Task runtime information:" in prompt
    assert "Before stopping, write the final repository diff" in prompt
    assert "Do not search the web" in prompt


def test_system_prompt_has_minimal_inspect_rules() -> None:
    from patcheval_inspect.solver import DEFAULT_SYSTEM_MESSAGE

    # The user prompt carries the PatchEval task text; the system prompt only
    # keeps Inspect-specific mechanics.
    assert "submit" in DEFAULT_SYSTEM_MESSAGE
    assert "/workspace/fix.patch" in DEFAULT_SYSTEM_MESSAGE
    assert "Do not search the web" not in DEFAULT_SYSTEM_MESSAGE
