from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cgr_exp.secret_safety import audit_secret_safety


def _fake_project_key() -> str:
    return "sk-" + "proj-" + "abcdefghijklmnopqrstuvwxyz123456"


class SecretSafetyTest(unittest.TestCase):
    def test_current_tracked_files_do_not_contain_real_openai_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = audit_secret_safety(Path(tmp), root=Path("."))
            self.assertTrue(manifest["secrets_ready"])
            self.assertEqual(manifest["finding_count"], 0)

    def test_audit_redacts_detected_project_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "leak.txt").write_text(f"CGR_LLM_API_KEY={_fake_project_key()}\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "add", "leak.txt"], cwd=root, check=True, capture_output=True)
            out = root / "audit"
            manifest = audit_secret_safety(out, root=root)
            self.assertFalse(manifest["secrets_ready"])
            self.assertEqual(manifest["finding_count"], 1)
            row = manifest["rows"][0]
            self.assertEqual(row["path"], "leak.txt")
            self.assertEqual(row["pattern"], "openai_project_key")
            self.assertNotIn("abcdefghijklmnopqrstuvwxyz", row["redacted_match"])
            saved = json.loads((out / "secret_safety_audit.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["secrets_ready"])

    def test_cli_fails_after_writing_artifacts_when_secret_is_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "leak.txt").write_text(f"{_fake_project_key()}\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "add", "leak.txt"], cwd=root, check=True, capture_output=True)
            out = root / "audit"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-secret-safety",
                    "--output-dir",
                    str(out),
                    "--root",
                    str(root),
                    "--overwrite",
                ],
                cwd=Path(__file__).resolve().parents[2],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("secret safety audit failed", result.stderr)
            saved = json.loads((out / "secret_safety_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["finding_count"], 1)
            self.assertTrue((out / "command_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
