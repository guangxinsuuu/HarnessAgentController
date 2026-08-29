from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.q2_local_primitive_audit import Q2_LOCAL_PRIMITIVE_CHECKS, audit_q2_local_primitives
from cgr_exp.table5_q2 import TABLE5_METHODS


class Q2LocalPrimitiveAuditTest(unittest.TestCase):
    def test_primitive_checks_cover_every_table_row(self) -> None:
        table_methods = {method for method, _setting, _source, _kind in TABLE5_METHODS}
        self.assertEqual(table_methods - set(Q2_LOCAL_PRIMITIVE_CHECKS), set())

    def test_local_primitive_audit_writes_ready_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = audit_q2_local_primitives(out)
            self.assertTrue(manifest["all_local_primitives_ready"])
            self.assertTrue((out / "q2_local_primitive_audit.csv").exists())
            saved = json.loads((out / "q2_local_primitive_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(saved["all_local_primitives_ready"])
            by_method = {row["Method"]: row for row in saved["rows"]}
            self.assertIn("method_generate", by_method["Tree of Thoughts"]["required_anchors"])
            self.assertIn("PRMGuidedMCTSStylePolicy", by_method["PRM-guided MCTS (ReST-MCTS-style)"]["required_anchors"])
            self.assertIn("NewOptionalIntervalVar", by_method["CP-SAT"]["required_anchors"])
            self.assertIn("cgr_exp/scheduling_tot_q2.py:", by_method["Tree of Thoughts"]["anchor_locations"])
            self.assertIn("cgr_exp/scheduling_transfer.py:", by_method["PRM-guided MCTS (ReST-MCTS-style)"]["anchor_locations"])
            self.assertIn("cgr_exp/scheduling_structured_q2.py:", by_method["CP-SAT"]["anchor_locations"])
            for row in saved["rows"]:
                self.assertEqual(row["missing_paths"], "", row["Method"])
                self.assertEqual(row["missing_anchors"], "", row["Method"])
                self.assertTrue(row["anchor_locations"], row["Method"])
                locations = json.loads(row["anchor_locations_json"])
                for anchor in row["required_anchors"].split("; "):
                    self.assertTrue(locations[anchor], f"{row['Method']}: {anchor}")


if __name__ == "__main__":
    unittest.main()
