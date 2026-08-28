from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.datasets.level1_scheduling import ScheduleAction
from cgr_exp.datasets.level2_renderer import render_scheduling_level2_variants
from cgr_exp.scheduling_micro import generate_scheduling_micro_instances
from cgr_exp.scheduling_transfer import (
    ORACLE_LABEL_FIELDS,
    SharedSchedulingProposalCache,
    PRMGuidedMCTSStylePolicy,
    ScriptedSchedulingClient,
    canonical_scheduling_state,
    canonical_state_hash,
    make_level2_scheduling_episodes,
    preflight_scheduling_llm_pilot,
    run_scheduling_fewshot_curve,
    run_scheduling_llm_pilot,
    run_scheduling_zero_shot_transfer,
)


class SchedulingTransferTest(unittest.TestCase):
    def test_scheduling_level2_renderings_keep_hidden_instance(self) -> None:
        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=11)["train"][0]
        variants = render_scheduling_level2_variants(task)
        self.assertGreaterEqual(len(variants), 3)
        self.assertTrue(all(variant.hidden_task is task for variant in variants))
        self.assertTrue(any(variant.perturbation == "entity_rename" for variant in variants))
        self.assertIn("duration", variants[0].text)

    def test_make_level2_scheduling_episodes_uses_same_hidden_tasks(self) -> None:
        tasks = generate_scheduling_micro_instances(train_count=0, validation_count=0, test_count=3, seed=12)["test"]
        episodes = make_level2_scheduling_episodes(tasks, seed=13, perturbation="canonical")
        self.assertEqual(len(episodes), 3)
        self.assertTrue(all(episode.hidden_task in tasks for episode in episodes))
        self.assertTrue(all(episode.perturbation == "canonical" for episode in episodes))

    def test_canonical_state_excludes_oracle_label_fields(self) -> None:
        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=16)["train"][0]
        state = canonical_scheduling_state(task, task.root_node(), remaining_budget=70)
        for label_field in ORACLE_LABEL_FIELDS:
            self.assertNotIn(label_field, state)
        self.assertIn("remaining_jobs", state)
        self.assertIn("eligible_jobs", state)
        self.assertIn("remaining_budget", state)

    def test_shared_proposal_cache_reuses_canonical_state(self) -> None:
        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=17)["train"][0]
        episode = make_level2_scheduling_episodes([task], seed=18, perturbation="canonical")[0]
        cache = SharedSchedulingProposalCache(episode, seed=19, proposal_k=4, node_budget=70, useful_action_recall=0.9)
        root = task.root_node()
        first = cache.propose(task, root, 4)
        second = cache.propose(task, root, 4)
        self.assertEqual([action.to_dict() for action in first], [action.to_dict() for action in second])
        self.assertEqual(cache.stats()["proposal_calls"], 1)
        self.assertEqual(cache.stats()["cache_hits"], 1)

    def test_shared_proposal_cache_filters_invalid_and_duplicate_actions(self) -> None:
        class NoisyAdapter:
            def propose_with_record(self, task, node, k):
                valid = task.candidate_actions(node, k)[0]
                return [
                    valid,
                    valid,
                    ScheduleAction(job="missing", machine="M0"),
                    ScheduleAction(job=valid.job, machine="missing"),
                ], {"input_tokens": 1, "output_tokens": 1, "latency": 0.0, "parse_failures": 0}

        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=22)["train"][0]
        episode = make_level2_scheduling_episodes([task], seed=23, perturbation="canonical")[0]
        cache = SharedSchedulingProposalCache(episode, seed=24, proposal_k=4, node_budget=70, useful_action_recall=0.9, adapter=NoisyAdapter())
        actions = cache.propose(task, task.root_node(), 4)
        self.assertEqual(len(actions), 1)
        stats = cache.stats()
        self.assertEqual(stats["invalid_actions"], 2)
        self.assertEqual(stats["duplicate_actions"], 1)
        self.assertEqual(stats["empty_valid_action_rate"], 0.0)

    def test_shared_proposal_cache_can_preload_frozen_state_rows(self) -> None:
        class FailingAdapter:
            def propose_with_record(self, task, node, k):
                raise AssertionError("preloaded cache should avoid adapter calls")

        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=25)["train"][0]
        episode = make_level2_scheduling_episodes([task], seed=26, perturbation="canonical")[0]
        node = task.root_node()
        action = task.candidate_actions(node, 1)[0]
        key = canonical_state_hash(task, node, 70, "mock", "scheduling-level2-v1", 4, 0.0)
        cache = SharedSchedulingProposalCache(episode, seed=27, proposal_k=4, node_budget=70, useful_action_recall=0.9, model="mock", adapter=FailingAdapter())
        cache.load_state_rows(
            [
                {
                    "cache_key": key,
                    "instance_id": task.task_id,
                    "actions": [action.to_dict()],
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "latency": 0.0,
                }
            ]
        )
        self.assertEqual(cache.propose(task, node, 4), [action])
        self.assertEqual(cache.stats()["proposal_calls"], 1)
        self.assertEqual(cache.stats()["total_tokens"], 18)
        self.assertEqual(cache.stats()["preloaded_state_count"], 1)
        self.assertEqual(cache.stats()["preloaded_tokens"], 18)
        self.assertEqual(cache.stats()["new_proposal_calls"], 0)
        self.assertEqual(cache.stats()["new_proposal_tokens"], 0)

    def test_zero_shot_transfer_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_zero_shot_transfer(
                Path(tmp),
                seed=14,
                train_count=8,
                validation_count=3,
                test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                useful_action_recall=0.9,
                perturbation="canonical",
                batch_size=2,
            )
            self.assertEqual(metrics["training_source"], "level1_solver_traces_only")
            self.assertEqual(metrics["target_training_trace_count"], 0)
            self.assertEqual(metrics["batch_size"], 2)
            self.assertGreater(metrics["train_pair_count"], 0)
            for name in (
                "config.json",
                "split_manifest.json",
                "feature_schema.json",
                "checkpoint_metadata.json",
                "label_manifest.json",
                "source_ranking_metrics.csv",
                "source_domain_table.csv",
                "target_zero_shot_table.csv",
                "per_instance_results.csv",
                "proposal_cache.jsonl",
                "proposal_quality.json",
                "failure_analysis.csv",
                "domain_shift_metrics.csv",
                "token_accounting.csv",
                "timing_metrics.csv",
                "scheduling_transfer_audit.json",
                "scheduling_transfer_table.csv",
                "scheduling_transfer_per_instance.csv",
                "scheduling_transfer_proposal_cache.csv",
                "scheduling_transfer_leakage_manifest.json",
                "scheduling_level2_episodes.jsonl",
            ):
                self.assertTrue((Path(tmp) / name).exists(), name)
            saved = json.loads((Path(tmp) / "scheduling_transfer_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("Solver-trained MLP", saved["methods"])
            self.assertIn("LLM value evaluator", saved["methods"])
            self.assertTrue(saved["shared_proposal_replay"]["shared"])
            self.assertFalse(saved["certified_pruning_enabled"])
            self.assertFalse(saved["semantic_state_schema"]["oracle_labels_in_inference_features"])
            table_text = (Path(tmp) / "scheduling_transfer_table.csv").read_text(encoding="utf-8")
            self.assertIn("frontier_top1_accuracy", table_text)
            self.assertIn("pairwise_accuracy", table_text)
            self.assertIn("selection_regret_mean", table_text)
            source_domain_text = (Path(tmp) / "source_domain_table.csv").read_text(encoding="utf-8")
            self.assertIn("Random", source_domain_text)
            self.assertIn("MLP", source_domain_text)
            source_ranking_text = (Path(tmp) / "source_ranking_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("source_validation", source_ranking_text)
            self.assertIn("source_test", source_ranking_text)
            feature_schema = json.loads((Path(tmp) / "feature_schema.json").read_text(encoding="utf-8"))
            self.assertFalse(feature_schema["oracle_labels_in_inference_features"])
            self.assertIn("machine_loads", feature_schema["common_inference_features"])
            self.assertIn("observable_history", feature_schema["common_inference_features"])
            label_manifest = json.loads((Path(tmp) / "label_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(label_manifest["supervision"], "pairwise_frontier_ranking")
            self.assertEqual(label_manifest["residual_solver_status"], "exact_exhaustive")
            self.assertFalse(label_manifest["pseudo_oracle_labels_used"])
            self.assertFalse(label_manifest["oracle_labels_in_inference_features"])
            self.assertEqual(label_manifest["unknown_pairs_excluded"], 0)
            self.assertGreater(label_manifest["train_pair_count"], 0)
            split_manifest = json.loads((Path(tmp) / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(set(split_manifest["train"]) & set(split_manifest["test"]))
            self.assertIn("Solver-trained MLP", (Path(tmp) / "target_zero_shot_table.csv").read_text(encoding="utf-8"))
            self.assertIn("total_tokens", (Path(tmp) / "token_accounting.csv").read_text(encoding="utf-8"))
            self.assertIn("proposal_support", (Path(tmp) / "failure_analysis.csv").read_text(encoding="utf-8"))
            self.assertIn("proposal_recall_at_k", (Path(tmp) / "proposal_quality.json").read_text(encoding="utf-8"))
            domain_shift_text = (Path(tmp) / "domain_shift_metrics.csv").read_text(encoding="utf-8")
            self.assertIn("overall_l2_standardized_mean_shift", domain_shift_text)
            self.assertIn("feature_standardized_mean_difference", domain_shift_text)
            proposal_lines = (Path(tmp) / "proposal_cache.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertGreater(len(proposal_lines), len(split_manifest["test"]))
            proposal_row = json.loads(proposal_lines[0])
            self.assertIn("cache_key", proposal_row)
            self.assertIn("canonical_state", proposal_row)
            self.assertIn("machine_loads", proposal_row["canonical_state"])
            self.assertIn("observable_history", proposal_row["canonical_state"])
            self.assertIn("features", proposal_row)
            self.assertIn("feature_names", proposal_row)
            self.assertIn("actions", proposal_row)
            audit = json.loads((Path(tmp) / "scheduling_transfer_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["all_core_harness_requirements_met"])
            self.assertFalse(audit["real_llm_pilot_complete"])
            leakage_items = [item for item in audit["requirements"] if item["requirement"] == "oracle label fields excluded from inference features"]
            self.assertIn("label_manifest.json", leakage_items[0]["evidence"])
            domain_shift_items = [item for item in audit["requirements"] if item["requirement"] == "source-target domain shift metrics reported"]
            self.assertEqual(domain_shift_items[0]["status"], "met")
            real_llm_items = [item for item in audit["requirements"] if item["requirement"] == "real Scheduling GPT pilot API run"]
            self.assertEqual(real_llm_items[0]["status"], "missing")

    def test_zero_shot_transfer_resume_reuses_completed_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = run_scheduling_zero_shot_transfer(
                root,
                seed=26,
                train_count=8,
                validation_count=3,
                test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                useful_action_recall=0.9,
                perturbation="canonical",
                batch_size=2,
            )
            second = run_scheduling_zero_shot_transfer(
                root,
                seed=26,
                train_count=8,
                validation_count=3,
                test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                useful_action_recall=0.9,
                perturbation="canonical",
                batch_size=2,
                resume=True,
            )
            self.assertEqual(first["methods"], second["methods"])
            self.assertTrue(second["resume"])
            lines = (root / "scheduling_transfer_per_instance.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1 + 5 * 4)

    def test_zero_shot_batch_size_does_not_change_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            common = {
                "seed": 27,
                "train_count": 8,
                "validation_count": 3,
                "test_count": 4,
                "job_count": 5,
                "machine_count": 2,
                "node_budget": 70,
                "action_budget": 6,
                "records_per_instance": 12,
                "rank_epochs": 5,
                "useful_action_recall": 0.9,
                "perturbation": "canonical",
            }
            first = run_scheduling_zero_shot_transfer(Path(tmp_a), batch_size=1, **common)
            second = run_scheduling_zero_shot_transfer(Path(tmp_b), batch_size=2, **common)
            self.assertEqual(first["methods"], second["methods"])
            self.assertEqual(
                (Path(tmp_a) / "scheduling_transfer_per_instance.csv").read_text(encoding="utf-8"),
                (Path(tmp_b) / "scheduling_transfer_per_instance.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (Path(tmp_a) / "proposal_cache.jsonl").read_text(encoding="utf-8"),
                (Path(tmp_b) / "proposal_cache.jsonl").read_text(encoding="utf-8"),
            )

    def test_zero_shot_parallel_episodes_do_not_change_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            common = {
                "seed": 31,
                "train_count": 8,
                "validation_count": 3,
                "test_count": 4,
                "job_count": 5,
                "machine_count": 2,
                "node_budget": 70,
                "action_budget": 6,
                "records_per_instance": 12,
                "rank_epochs": 5,
                "useful_action_recall": 0.9,
                "perturbation": "canonical",
                "batch_size": 2,
            }
            first = run_scheduling_zero_shot_transfer(Path(tmp_a), parallel_episodes=1, **common)
            second = run_scheduling_zero_shot_transfer(Path(tmp_b), parallel_episodes=2, **common)
            self.assertEqual(first["methods"], second["methods"])
            self.assertEqual(
                (Path(tmp_a) / "scheduling_transfer_per_instance.csv").read_text(encoding="utf-8"),
                (Path(tmp_b) / "scheduling_transfer_per_instance.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (Path(tmp_a) / "proposal_cache.jsonl").read_text(encoding="utf-8"),
                (Path(tmp_b) / "proposal_cache.jsonl").read_text(encoding="utf-8"),
            )

    def test_scheduling_llm_pilot_runs_with_scripted_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_llm_pilot(
                Path(tmp),
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=20,
                train_count=8,
                validation_count=3,
                test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                perturbation="canonical",
                batch_size=2,
                parallel_episodes=2,
            )
            root = Path(tmp)
            self.assertEqual(metrics["benchmark"], "scheduling_real_llm_pilot")
            self.assertEqual(metrics["batch_size"], 2)
            self.assertEqual(metrics["parallel_episodes"], 2)
            self.assertFalse(metrics["live_network_required"])
            self.assertTrue(metrics["shared_proposal_replay"]["shared"])
            for name in (
                "config.json",
                "split_manifest.json",
                "feature_schema.json",
                "checkpoint_metadata.json",
                "target_zero_shot_table.csv",
                "per_instance_results.csv",
                "proposal_cache.jsonl",
                "proposal_quality.json",
                "token_accounting.csv",
                "timing_metrics.csv",
                "scheduling_transfer_audit.json",
                "scheduling_llm_pilot_table.csv",
                "scheduling_llm_pilot_proposal_cache.csv",
                "scheduling_llm_pilot_leakage_manifest.json",
            ):
                self.assertTrue((root / name).exists(), name)
            saved = json.loads((root / "scheduling_llm_pilot_metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["certified_pruning_enabled"])
            self.assertFalse(saved["semantic_state_schema"]["oracle_labels_in_inference_features"])
            table_text = (root / "scheduling_llm_pilot_table.csv").read_text(encoding="utf-8")
            self.assertIn("frontier_top1_accuracy", table_text)
            self.assertIn("pairwise_accuracy", table_text)
            self.assertIn("selection_regret_mean", table_text)

    def test_scheduling_llm_pilot_resume_reuses_completed_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = run_scheduling_llm_pilot(
                root,
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=21,
                train_count=8,
                validation_count=3,
                test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                perturbation="canonical",
                batch_size=2,
            )
            second = run_scheduling_llm_pilot(
                root,
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=21,
                train_count=8,
                validation_count=3,
                test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                perturbation="canonical",
                batch_size=2,
                resume=True,
            )
            self.assertEqual(first["methods"], second["methods"])
            self.assertTrue(second["resume"])
            lines = (root / "scheduling_llm_pilot_per_instance.csv").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1 + 4 * 4)
            self.assertGreater(len((root / "proposal_cache.jsonl").read_text(encoding="utf-8").strip().splitlines()), 0)

    def test_scheduling_llm_pilot_preflight_is_key_safe_and_resume_aware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_scheduling_llm_pilot(
                root,
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=25,
                train_count=8,
                validation_count=3,
                test_count=2,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                perturbation="canonical",
                batch_size=1,
            )
            preflight = preflight_scheduling_llm_pilot(
                root,
                seed=25,
                train_count=8,
                validation_count=3,
                test_count=2,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                model="gpt-4o-mini",
                batch_size=2,
                parallel_episodes=3,
                perturbation="canonical",
                env={"CGR_LLM_API_KEY": "secret-value"},
            )
            self.assertTrue(preflight["api_key_present"])
            self.assertNotIn("secret-value", json.dumps(preflight))
            self.assertEqual(preflight["completed_episode_count"], 2)
            self.assertEqual(preflight["remaining_episode_count"], 0)
            self.assertEqual(preflight["parallel_episodes"], 3)
            self.assertTrue((root / "scheduling_llm_pilot_preflight.json").exists())

    def test_scheduling_llm_pilot_can_include_gpt_value_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_llm_pilot(
                Path(tmp),
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=27,
                train_count=8,
                validation_count=3,
                test_count=3,
                job_count=5,
                machine_count=2,
                node_budget=50,
                action_budget=5,
                records_per_instance=10,
                rank_epochs=4,
                perturbation="canonical",
                batch_size=1,
                include_gpt_value=True,
            )
            self.assertIn("GPT-value / ToT-style", metrics["methods"])
            value_row = metrics["methods"]["GPT-value / ToT-style"]
            self.assertGreater(value_row["value_calls"], 0)
            self.assertGreater(value_row["value_tokens"], 0)
            self.assertGreater(value_row["total_tokens"], value_row["proposal_tokens"])
            saved = (Path(tmp) / "scheduling_llm_pilot_table.csv").read_text(encoding="utf-8")
            self.assertIn("GPT-value / ToT-style", saved)

    def test_scheduling_llm_pilot_can_include_gpt_frontier_ranking_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_llm_pilot(
                Path(tmp),
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=28,
                train_count=8,
                validation_count=3,
                test_count=3,
                job_count=5,
                machine_count=2,
                node_budget=50,
                action_budget=5,
                records_per_instance=10,
                rank_epochs=4,
                perturbation="canonical",
                batch_size=1,
                include_gpt_frontier_ranking=True,
            )
            self.assertIn("GPT-frontier-rank / ToT-style", metrics["methods"])
            value_row = metrics["methods"]["GPT-frontier-rank / ToT-style"]
            self.assertGreater(value_row["value_calls"], 0)
            self.assertGreater(value_row["value_tokens"], 0)
            self.assertGreater(value_row["total_tokens"], value_row["proposal_tokens"])
            self.assertLessEqual(value_row["value_calls"], 3 * 50)

    def test_scheduling_llm_pilot_can_include_target_prm_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = run_scheduling_llm_pilot(
                root,
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=29,
                train_count=8,
                validation_count=3,
                test_count=3,
                job_count=5,
                machine_count=2,
                node_budget=50,
                action_budget=5,
                records_per_instance=10,
                rank_epochs=4,
                perturbation="canonical",
                batch_size=1,
                include_target_prm=True,
                target_prm_train_count=4,
            )
            self.assertIn("Target-PRM", metrics["methods"])
            self.assertEqual(metrics["target_prm_train_count"], 4)
            self.assertGreater(metrics["target_prm_trace_record_count"], 0)
            self.assertGreater(metrics["target_prm_pair_count"], 0)
            target_protocol = metrics["learned_search_protocol"]["policies"]["Target-PRM"]
            self.assertEqual(target_protocol["training_distribution"], "target LLM-induced Scheduling trajectories from disjoint training instances")
            self.assertFalse(target_protocol["zero_shot_transfer"])
            self.assertIn("Target-PRM priority", metrics["q2_method_alignment"])
            self.assertIn(
                "process supervision",
                metrics["q2_method_alignment"]["Target-PRM priority"]["source_primitives"],
            )
            for name in (
                "target_prm_ranker.json",
                "target_prm_trace_records.jsonl",
                "target_prm_train_pairs.jsonl",
                "target_prm_training_proposal_cache.csv",
                "target_prm_training_proposal_states.jsonl",
            ):
                self.assertTrue((root / name).exists(), name)

    def test_scheduling_llm_pilot_can_include_rest_mcts_and_solver_certification_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = run_scheduling_llm_pilot(
                root,
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=31,
                train_count=8,
                validation_count=3,
                test_count=3,
                job_count=5,
                machine_count=2,
                node_budget=50,
                action_budget=5,
                records_per_instance=10,
                rank_epochs=4,
                perturbation="canonical",
                batch_size=1,
                include_rest_mcts=True,
                include_solver_certification=True,
                target_prm_train_count=4,
            )
            self.assertIn("PRM-guided MCTS (ReST-MCTS-style)", metrics["methods"])
            self.assertIn("Solver-trained MLP + certified pruning", metrics["methods"])
            self.assertEqual(metrics["target_prm_train_count"], 4)
            self.assertGreater(metrics["target_prm_trace_record_count"], 0)
            rest_protocol = metrics["learned_search_protocol"]["policies"]["PRM-guided MCTS (ReST-MCTS-style)"]
            self.assertFalse(rest_protocol["full_iterative_self_training"])
            rest_row = metrics["methods"]["PRM-guided MCTS (ReST-MCTS-style)"]
            self.assertGreater(rest_row["mcts_selection_calls"], 0)
            self.assertGreater(rest_row["mcts_expansion_observations"], 0)
            self.assertGreater(rest_row["mcts_prior_evaluations"], 0)
            self.assertGreater(rest_row["mcts_leaf_value_evaluations"], 0)
            self.assertGreater(rest_row["mcts_backup_updates"], 0)
            cert_protocol = metrics["learned_search_protocol"]["policies"]["Solver-trained MLP + certified pruning"]
            self.assertEqual(cert_protocol["pruning_authority"], "verified lower-bound/feasibility logic only")
            self.assertFalse(cert_protocol["learned_score_authorizes_pruning"])
            self.assertIn("PRM-guided MCTS (ReST-MCTS-style)", metrics["q2_method_alignment"])
            self.assertIn("Solver-trained priority + certified pruning", metrics["q2_method_alignment"])
            self.assertIn(
                "MCTS backpropagation",
                metrics["q2_method_alignment"]["PRM-guided MCTS (ReST-MCTS-style)"]["source_primitives"],
            )
            self.assertIn(
                "verified pruning authority",
                metrics["q2_method_alignment"]["Solver-trained priority + certified pruning"]["source_primitives"],
            )
            certified = metrics["methods"]["Solver-trained MLP + certified pruning"]
            self.assertGreaterEqual(certified["false_prunes"], 0)
            self.assertIn("on_demand_total_tokens", certified)
            self.assertLessEqual(certified["on_demand_total_tokens"], certified["total_tokens"])
            for name in (
                "target_prm_ranker.json",
                "target_prm_trace_records.jsonl",
                "target_prm_train_pairs.jsonl",
            ):
                self.assertTrue((root / name).exists(), name)

    def test_scheduling_llm_pilot_resume_requires_each_requested_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = dict(
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=36,
                train_count=8,
                validation_count=3,
                test_count=2,
                job_count=5,
                machine_count=2,
                node_budget=35,
                action_budget=5,
                records_per_instance=10,
                rank_epochs=4,
                perturbation="canonical",
                batch_size=1,
            )
            run_scheduling_llm_pilot(root, **kwargs)
            per_instance_path = root / "scheduling_llm_pilot_per_instance.csv"
            with per_instance_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            first_rendering = rows[0]["rendering_id"]
            first_random = next(row for row in rows if row["rendering_id"] == first_rendering and row["Policy"] == "Random")
            corrupted_rows = [
                row
                for row in rows
                if not (row["rendering_id"] == first_rendering and row["Policy"] == "Oracle")
            ]
            corrupted_rows.append(dict(first_random))
            with per_instance_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(corrupted_rows)

            run_scheduling_llm_pilot(root, **kwargs, resume=True)
            with per_instance_path.open(newline="", encoding="utf-8") as handle:
                resumed_rows = list(csv.DictReader(handle))
            policies_by_rendering = {}
            for row in resumed_rows:
                policies_by_rendering.setdefault(row["rendering_id"], []).append(row["Policy"])
            self.assertEqual(set(policies_by_rendering), {row["rendering_id"] for row in rows})
            for policy_names in policies_by_rendering.values():
                self.assertEqual(sorted(policy_names), ["Hand heuristic", "Oracle", "Random", "Solver-trained MLP"])

    def test_scheduling_llm_pilot_resume_preserves_unrequested_policy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = dict(
                client=ScriptedSchedulingClient(model="mock-scheduling"),
                model="mock-scheduling",
                seed=37,
                train_count=8,
                validation_count=3,
                test_count=2,
                job_count=5,
                machine_count=2,
                node_budget=35,
                action_budget=5,
                records_per_instance=10,
                rank_epochs=4,
                perturbation="canonical",
                batch_size=1,
                target_prm_train_count=4,
            )
            run_scheduling_llm_pilot(root, **kwargs, include_target_prm=True)
            resumed = run_scheduling_llm_pilot(root, **kwargs, include_solver_certification=True, resume=True)
            self.assertIn("Target-PRM", resumed["methods"])
            self.assertIn("Solver-trained MLP + certified pruning", resumed["methods"])
            self.assertIn("Target-PRM priority", resumed["q2_method_alignment"])
            with (root / "scheduling_llm_pilot_per_instance.csv").open(newline="", encoding="utf-8") as handle:
                per_instance = list(csv.DictReader(handle))
            policies = {row["Policy"] for row in per_instance}
            self.assertIn("Target-PRM", policies)
            self.assertIn("Solver-trained MLP + certified pruning", policies)
            with (root / "scheduling_llm_pilot_proposal_cache.csv").open(newline="", encoding="utf-8") as handle:
                proposal_rows = list(csv.DictReader(handle))
            proposal_renderings = [row["rendering_id"] for row in proposal_rows]
            self.assertEqual(len(proposal_renderings), len(set(proposal_renderings)))

    def test_prm_guided_mcts_policy_tracks_visits_and_backups(self) -> None:
        class ConstantRanker:
            def score(self, features):
                return 0.25 + 0.01 * features[0]

        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=41, job_count=4, machine_count=2)["train"][0]
        root = task.root_node()
        children = [task.apply(root, action, idx) for idx, action in enumerate(task.candidate_actions(root, 2))]
        policy = PRMGuidedMCTSStylePolicy(task, ConstantRanker())
        first_scores = policy.score_frontier(children)
        self.assertEqual(len(first_scores), len(children))
        self.assertEqual(policy.total_backups, 0)
        second_scores = policy.score_frontier(children)
        self.assertEqual(len(second_scores), len(children))
        self.assertGreater(policy.total_backups, 0)
        self.assertGreater(sum(policy.visits.values()), 0)
        diagnostics = policy.diagnostics()
        self.assertTrue(diagnostics["mcts_components_observed"]["selection"])
        self.assertTrue(diagnostics["mcts_components_observed"]["process_reward_guidance"])
        self.assertGreater(diagnostics["selection_calls"], 0)
        self.assertGreater(diagnostics["prior_evaluations"], 0)

    def test_prm_guided_mcts_policy_backs_up_observed_expansion(self) -> None:
        class ConstantRanker:
            def score(self, features):
                return 0.0

        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=43, job_count=4, machine_count=2)["train"][0]
        root = task.root_node()
        children = [task.apply(root, action, idx) for idx, action in enumerate(task.candidate_actions(root, 2))]
        policy = PRMGuidedMCTSStylePolicy(task, ConstantRanker())
        policy.score_frontier([root])
        policy.observe_expansion(root, children)
        self.assertIsNone(policy.last_selected)
        self.assertEqual(policy.visits[root.node_id], 1)
        self.assertGreater(policy.value_sum[root.node_id], 0.0)
        diagnostics = policy.diagnostics()
        self.assertTrue(diagnostics["mcts_components_observed"]["expansion"])
        self.assertTrue(diagnostics["mcts_components_observed"]["simulation/evaluation"])
        self.assertTrue(diagnostics["mcts_components_observed"]["backpropagation"])
        for child in children:
            self.assertIn(child.node_id, policy.node_cache)

    def test_fewshot_curve_writes_regime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_fewshot_curve(
                Path(tmp),
                seed=15,
                solver_train_count=8,
                target_train_count=12,
                validation_count=3,
                target_test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                fewshot_counts=(3, 6),
                useful_action_recall=0.9,
                perturbation="canonical",
            )
            root = Path(tmp)
            self.assertEqual(metrics["benchmark"], "scheduling_fewshot_curve")
            self.assertFalse(metrics["target_train_eval_overlap"])
            self.assertGreater(metrics["solver_pair_count"], 0)
            self.assertGreater(metrics["target_trace_record_count"], 0)
            for name in (
                "scheduling_fewshot_curve.csv",
                "scheduling_fewshot_metrics.json",
                "scheduling_fewshot_target_train_episodes.jsonl",
                "scheduling_fewshot_target_test_episodes.jsonl",
                "scheduling_level2_target_trace_records.jsonl",
                "scheduling_solver_train_pairs.jsonl",
            ):
                self.assertTrue((root / name).exists(), name)
            curve_text = (root / "scheduling_fewshot_curve.csv").read_text(encoding="utf-8")
            self.assertIn("Solver-only", curve_text)
            self.assertIn("Solver pretrain + target adapt", curve_text)
            self.assertIn("Target-trajectory trained", curve_text)
            self.assertIn("source_replay_pair_count", curve_text)
            self.assertNotIn("LLM-only", curve_text)
            saved = json.loads((root / "scheduling_fewshot_metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(saved["target_train_eval_overlap"])
            self.assertIn("Solver-only:0", saved["training_regimes"])
            self.assertIn("target_only_baseline_definition", saved)
            self.assertIn("solver_pretrain_adaptation", saved)


if __name__ == "__main__":
    unittest.main()
