from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cgr_exp.schemas import CostLedger


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetState:
    proposal_call_budget: Optional[int] = None
    evaluation_call_budget: Optional[int] = None
    generated_token_budget: Optional[int] = None
    cost: CostLedger = CostLedger()

    def _check(self) -> None:
        if self.proposal_call_budget is not None and self.cost.proposal_calls > self.proposal_call_budget:
            raise BudgetExceeded("proposal call budget exceeded")
        if self.evaluation_call_budget is not None and self.cost.evaluation_calls > self.evaluation_call_budget:
            raise BudgetExceeded("evaluation call budget exceeded")
        if self.generated_token_budget is not None and (self.cost.input_tokens + self.cost.output_tokens) > self.generated_token_budget:
            raise BudgetExceeded("generated token budget exceeded")

    def can_charge_proposal(self) -> bool:
        return self.proposal_call_budget is None or self.cost.proposal_calls < self.proposal_call_budget

    def charge_proposal(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.cost = self.cost + CostLedger(proposal_calls=1, input_tokens=input_tokens, output_tokens=output_tokens)
        self._check()

    def charge_evaluation(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.cost = self.cost + CostLedger(evaluation_calls=1, input_tokens=input_tokens, output_tokens=output_tokens)
        self._check()

    def charge_verifier(self) -> None:
        self.cost = self.cost + CostLedger(verifier_calls=1)
        self._check()


__all__ = ["CostLedger", "BudgetState", "BudgetExceeded"]
