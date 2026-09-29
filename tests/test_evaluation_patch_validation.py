import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patcheval"))
_validate_patch_cves = importlib.import_module(
    "evaluation.run_evaluation"
)._validate_patch_cves


class PatchCveValidationTest(unittest.TestCase):
    def test_subset_submissions_remain_supported(self):
        _validate_patch_cves(
            [{"cve": "CVE-2025-0001"}],
            {"CVE-2025-0001": "Python", "CVE-2025-0002": "Go"},
        )

    def test_duplicate_normalized_cves_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate CVEs: CVE-2025-0001"):
            _validate_patch_cves(
                [
                    {"cve": "CVE-2025-0001"},
                    {"cve": "ghcr.io/anonymous2578-data/CVE-2025-0001:latest"},
                ],
                {"CVE-2025-0001": "Python"},
            )

    def test_unknown_cves_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown CVEs: CVE-2025-9999"):
            _validate_patch_cves(
                [{"cve": "CVE-2025-9999"}],
                {"CVE-2025-0001": "Python"},
            )


if __name__ == "__main__":
    unittest.main()
