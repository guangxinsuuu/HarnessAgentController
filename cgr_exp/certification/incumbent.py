from __future__ import annotations


def update_incumbent(current, candidate):
    if candidate is None:
        return current
    if current is None or candidate < current:
        return candidate
    return current

