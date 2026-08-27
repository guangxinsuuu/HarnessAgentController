from __future__ import annotations


def node_feature_dict(node):
    return {**node.universal_features, **node.verifier_features}

