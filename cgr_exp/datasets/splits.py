from __future__ import annotations

import random


def assert_disjoint_splits(*splits):
    seen = set()
    for split in splits:
        overlap = seen.intersection(split)
        if overlap:
            raise ValueError(f"split overlap: {sorted(overlap)[:3]}")
        seen.update(split)


def build_split_manifest(seed: int, family: str, base_count: int):
    rng = random.Random(seed)
    split_names = [
        "train",
        "validation",
        "test_iid",
        "test_size_ood",
        "test_structural_ood",
        "semantic_paraphrase_ood",
        "entity_renaming_ood",
        "proposal_quality_ood",
    ]
    manifest = {}
    for split_index, split in enumerate(split_names):
        ids = [f"{family}:{split}:{seed}:{i:04d}" for i in range(base_count)]
        rng.shuffle(ids)
        manifest[split] = ids
    assert_disjoint_splits(*manifest.values())
    return manifest


def build_family_split_manifest(seed: int, families, base_count: int):
    manifest = {
        family: build_split_manifest(seed + family_index, family, base_count)
        for family_index, family in enumerate(families)
    }
    all_splits = []
    for family_manifest in manifest.values():
        all_splits.extend(family_manifest.values())
    assert_disjoint_splits(*all_splits)
    return manifest
