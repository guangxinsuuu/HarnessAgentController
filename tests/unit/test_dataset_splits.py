from __future__ import annotations

import unittest

from cgr_exp.datasets.splits import build_family_split_manifest, build_split_manifest


class DatasetSplitTest(unittest.TestCase):
    def test_split_manifest_has_required_disjoint_splits(self) -> None:
        manifest = build_split_manifest(seed=3, family="assignment", base_count=4)
        required = {"train", "validation", "test_iid", "test_size_ood", "test_structural_ood", "semantic_paraphrase_ood", "entity_renaming_ood", "proposal_quality_ood"}
        self.assertTrue(required.issubset(manifest))
        all_ids = []
        for split in required:
            ids = manifest[split]
            self.assertTrue(ids, split)
            all_ids.extend(ids)
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_split_manifest_is_reproducible(self) -> None:
        self.assertEqual(
            build_split_manifest(seed=5, family="assignment", base_count=3),
            build_split_manifest(seed=5, family="assignment", base_count=3),
        )

    def test_family_split_manifest_covers_all_requested_families_without_overlap(self) -> None:
        manifest = build_family_split_manifest(seed=7, families=("assignment", "scheduling", "setcover", "level3_tool"), base_count=3)
        self.assertEqual(set(manifest), {"assignment", "scheduling", "setcover", "level3_tool"})
        all_ids = []
        for family_manifest in manifest.values():
            for ids in family_manifest.values():
                all_ids.extend(ids)
        self.assertEqual(len(all_ids), len(set(all_ids)))


if __name__ == "__main__":
    unittest.main()
