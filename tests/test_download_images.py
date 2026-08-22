import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import download_images


class DownloadImagesMainTest(unittest.TestCase):
    @mock.patch.object(download_images, "batch_pull_images", return_value=(230, 0))
    def test_success_exits_zero(self, _batch_pull_images):
        self.assertEqual(download_images.main(), 0)

    @mock.patch.object(download_images, "batch_pull_images", return_value=(229, 1))
    def test_partial_failure_exits_nonzero(self, _batch_pull_images):
        self.assertEqual(download_images.main(), 1)

    @mock.patch.object(download_images, "batch_pull_images", return_value=(0, 0))
    def test_setup_failure_exits_nonzero(self, _batch_pull_images):
        self.assertEqual(download_images.main(), 1)

    def test_missing_image_list_command_exits_nonzero(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "download_images.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
