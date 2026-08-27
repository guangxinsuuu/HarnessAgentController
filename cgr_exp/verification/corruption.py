from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationCorruptionConfig:
    false_negative_rate: float = 0.0
    false_positive_rate: float = 0.0
    uncertain_noise_rate: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("false_negative_rate", self.false_negative_rate),
            ("false_positive_rate", self.false_positive_rate),
            ("uncertain_noise_rate", self.uncertain_noise_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
