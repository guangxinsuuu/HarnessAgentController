from __future__ import annotations


def semantic_perturbations(text: str):
    return {
        "canonical": text,
        "paraphrase": text,
        "entity_rename": text,
        "irrelevant_context": text + " Extra context is irrelevant.",
        "mild_ambiguity": text,
    }

