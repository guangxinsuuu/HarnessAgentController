from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cgr_exp.datasets.game24 import Game24Action
from cgr_exp.game24_llm_pilot import ScriptedGame24Client, SharedProposalCacheProposer, run_game24_llm_pilot
from cgr_exp.proposers.llm_adapter import RealLLMProposerAdapter


class Game24LLMPilotTest(unittest.TestCase):
    def test_mock_pilot_uses_shared_proposal_cache_and_writes_tables(self) -> None:
        client = ScriptedGame24Client()
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_game24_llm_pilot(
                out_dir=Path(tmp),
                client=client,
                model="mock-game24",
                seed=0,
                start=900,
                count=3,
                proposal_k=4,
                node_budget=40,
                batch_size=2,
            )
            self.assertFalse(metrics["live_network_required"])
            self.assertEqual(metrics["batch_size"], 2)
            self.assertEqual(metrics["proposal_cache"]["shared"], True)
            self.assertGreater(metrics["proposal_cache"]["state_count"], 0)
            self.assertTrue((Path(tmp) / "game24_llm_pilot_table.csv").exists())
            self.assertTrue((Path(tmp) / "game24_proposal_cache.jsonl").exists())
            self.assertTrue((Path(tmp) / "progress.json").exists())
            self.assertTrue((Path(tmp) / "game24_llm_pilot_metrics.json").exists())
            methods = set(metrics["methods"])
            self.assertTrue({"CoT", "ToT-style", "MLP priority", "Oracle priority"}.issubset(methods))
            progress = (Path(tmp) / "progress.json").read_text(encoding="utf-8")
            self.assertIn("proposal_cache_states", progress)

    def test_tot_style_counts_value_calls_separately_from_proposal_calls(self) -> None:
        client = ScriptedGame24Client()
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_game24_llm_pilot(
                out_dir=Path(tmp),
                client=client,
                model="mock-game24",
                seed=1,
                start=900,
                count=2,
                proposal_k=4,
                node_budget=30,
            )
            tot = metrics["methods"]["ToT-style"]
            mlp = metrics["methods"]["MLP priority"]
            self.assertGreater(tot["value_calls"], 0)
            self.assertEqual(mlp["value_calls"], 0)
            self.assertGreaterEqual(tot["total_tokens"], tot["proposal_tokens"])

    def test_batch_replay_preserves_existing_cache_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            baseline = run_game24_llm_pilot(
                out_dir=out,
                client=ScriptedGame24Client(),
                model="mock-game24",
                seed=2,
                start=900,
                count=5,
                proposal_k=4,
                node_budget=35,
                batch_size=1,
                resume=False,
            )
            cache_before = (out / "game24_proposal_cache.jsonl").read_text(encoding="utf-8")
            replayed = run_game24_llm_pilot(
                out_dir=out,
                client=ScriptedGame24Client(),
                model="mock-game24",
                seed=2,
                start=900,
                count=5,
                proposal_k=4,
                node_budget=35,
                batch_size=8,
                resume=True,
            )
            self.assertEqual(baseline["proposal_cache"], replayed["proposal_cache"])
            self.assertEqual(baseline["methods"], replayed["methods"])
            self.assertEqual(cache_before, (out / "game24_proposal_cache.jsonl").read_text(encoding="utf-8"))

    def test_shared_proposal_cache_resumes_existing_jsonl(self) -> None:
        client = ScriptedGame24Client()
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.jsonl"
            progress_path = Path(tmp) / "progress.json"
            cache_path.write_text(
                '{"task_id":"t","node_id":"n","actions":[{"action_type":"game24_op","left":10,"right":4,"operator":"-"}],"input_tokens":2,"output_tokens":3,"proposal_recall_at_k":1.0}\n',
                encoding="utf-8",
            )
            proposer = SharedProposalCacheProposer(
                RealLLMProposerAdapter(client=client, model="mock", prompt_template_version="v1"),
                proposal_k=4,
                cache_path=cache_path,
                progress_path=progress_path,
                resume=True,
            )
            self.assertEqual(proposer.cache["n"], [Game24Action(left=10, right=4, operator="-")])
            self.assertEqual(proposer.cost.input_tokens, 2)
            self.assertIn("proposal_cache_states", progress_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
