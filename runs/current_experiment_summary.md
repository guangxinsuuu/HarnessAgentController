# Current Experiment Summary

Date: 2026-08-27

This file collects the currently trustworthy experiment results from the
local run artifacts. It distinguishes stable findings from hypotheses that
are not yet supported.

## Game24 Real-LLM Pilot

Trusted runs:

- `runs/game24_llm_pilot_20_prompt_v2_k8`
- `runs/game24_llm_pilot_50_prompt_v2_k8_batch8`
- `runs/game24_llm_pilot_100_prompt_v2_k8_batch8`

All use shared proposal replay with `gpt-4o-mini`, prompt v2, `K=8`,
and node budget 80.

| Count | Method | Success | Mean expanded nodes | Total tokens |
| ---: | --- | ---: | ---: | ---: |
| 20 | CoT | 0.85 | 18.35 | 586533 |
| 20 | ToT-style | 0.85 | 22.85 | 1054799 |
| 20 | MLP priority | 0.85 | 14.35 | 586533 |
| 20 | Oracle priority | 0.85 | 10.75 | 586533 |
| 50 | CoT | 0.70 | 26.80 | 1458809 |
| 50 | ToT-style | 0.70 | 27.20 | 2700327 |
| 50 | MLP priority | 0.70 | 22.16 | 1458809 |
| 50 | Oracle priority | 0.70 | 17.00 | 1458809 |
| 100 | CoT | 0.62 | 29.17 | 2857668 |
| 100 | ToT-style | 0.62 | 29.20 | 5401695 |
| 100 | MLP priority | 0.62 | 23.93 | 2857668 |
| 100 | Oracle priority | 0.62 | 20.08 | 2857668 |

Consistent finding:

- MLP priority matches ToT-style success on 20, 50, and 100 cases.
- MLP priority uses the same proposal tokens as CoT and avoids ToT value
  calls, saving about 44-47% total tokens versus ToT-style.
- MLP priority expands fewer nodes than CoT and ToT-style at every scale.
- Oracle has the same success ceiling as the learned policy under the shared
  proposal cache, but fewer expanded nodes. The remaining gap is search
  efficiency, not reachable-solution success.

What this supports:

- Learned frontier priority can replace expensive LLM value evaluation while
  preserving success on Game24 under matched proposal support.

What this does not support:

- It does not show higher final success than CoT or ToT.
- It does not show that Graph-GNN structure is necessary.

## Scheduling Zero-Shot Transfer

### Source-Domain Sanity

Trusted run:

- `runs/scheduling_micro_200_50_100_seed0_mlp_v2`

Setup:

- Source train: 200 solver-induced semantic scheduling instances.
- Source validation: 50.
- Source test: 100.
- `records_per_instance=60`, `rank_epochs=25`.
- No GPT or target LLM trajectories are used.

| Policy | Success | Gap | Mean nodes | Top-1 | Pairwise | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.80 | 0.0266 | 118.47 | 0.7055 | 0.5076 | 1.2904 |
| Hand heuristic | 0.10 | 0.8843 | 119.87 | 0.8252 | 0.5417 | 0.5235 |
| MLP | 0.96 | 0.0228 | 113.06 | 0.9008 | 0.6888 | 0.2331 |
| Oracle | 1.00 | 0.0000 | 112.33 | 1.0000 | 1.0000 | 0.0000 |

Consistent finding:

- Q1 source-domain sanity passes: solver-induced semantic trajectories train a
  useful ranker.
- MLP improves success over Random and Hand heuristic, reduces objective gap
  relative to Random, and has the best non-oracle ranking diagnostics.
- This result is source-only. It proves that the solver-side supervision has
  signal before asking whether that signal transfers to LLM-induced states.

### Target Zero-Shot

Trusted run:

- `runs/scheduling_transfer_zero_shot_shared_200_50_100_seed0_v2`

Setup:

- Source train: 200 solver-induced semantic scheduling instances.
- Source validation: 50.
- Target test: 100 LLM-induced semantic scheduling episodes.
- Shared proposal replay enabled.
- Certified pruning disabled.
- Solver-trained MLP uses no target training trajectories.

| Policy | Success | Gap | Mean nodes | Top-1 | Pairwise | Regret | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.73 | 0.0474 | 119.04 | 0.2906 | 0.5001 | 4.6266 | 7999058 |
| Hand heuristic | 0.18 | 0.7884 | 119.04 | 0.5123 | 0.6255 | 1.8163 | 7999058 |
| LLM value evaluator | 0.54 | 0.3575 | 119.04 | 0.4664 | 0.6631 | 2.0545 | 181120664 |
| Solver-trained MLP | 0.95 | 0.0248 | 119.26 | 0.4949 | 0.6426 | 1.7142 | 7999058 |
| Oracle | 1.00 | 0.0000 | 118.99 | 1.0000 | 1.0000 | 0.0000 | 7999058 |

Shared proposal replay diagnostics:

- State count: 45342.
- Request count: 59537.
- Cache hits: 14195.
- ProposalRecall@K: 0.9547.
- Empty-valid-action rate: 0.0212.
- Invalid actions filtered: 3190.
- Duplicate actions filtered: 3881.

Audit artifacts now present in the trusted run:

- `config.json`
- `split_manifest.json`
- `feature_schema.json`
- `checkpoint_metadata.json`
- `label_manifest.json`
- `source_ranking_metrics.csv`
- `source_domain_table.csv`
- `target_zero_shot_table.csv`
- `per_instance_results.csv`
- `proposal_cache.jsonl`
- `proposal_quality.json`
- `failure_analysis.csv`
- `domain_shift_metrics.csv`
- `token_accounting.csv`
- `timing_metrics.csv`
- `scheduling_transfer_audit.json`

Verified audit invariants:

- Train, validation, and target test instance IDs are disjoint.
- `feature_schema.json` records `oracle_labels_in_inference_features=false`.
  Its common semantic state fields include `scheduled_jobs`,
  `remaining_jobs`, `machine_loads`, `eligible_jobs`, current
  objective/incumbent fields, `depth`, `remaining_budget`, and
  `observable_history`.
- `config.json` records `certified_pruning_enabled=false`.
- `checkpoint_metadata.json` records SHA256 hashes for both ranker and feature
  schema.
- `label_manifest.json` records pairwise frontier-ranking supervision from the
  exact residual scheduling oracle, `pseudo_oracle_labels_used=false`,
  `uncertain_interval_pairs=0`, `unknown_pairs_excluded=0`, and
  `oracle_labels_in_inference_features=false`.
- `source_domain_table.csv` records held-out Level-I Scheduling source-domain
  policy evaluation for Random, Hand heuristic, MLP, and Oracle with success,
  gap, nodes, top-1, pairwise accuracy, and selection regret.
- `source_ranking_metrics.csv` now contains both validation pairwise ranker
  accuracy and source-test policy ranking diagnostics.
- `proposal_cache.jsonl` is state-level: it has one JSONL row per cached
  canonical state, including cache key, canonical state, valid unique actions,
  ranker feature vector, feature names, and invalid/duplicate action counts.
- `domain_shift_metrics.csv` is computed from source solver-trace feature
  vectors and target proposal-cache state feature vectors. It records 14982
  source states, 45342 target states, per-feature standardized mean
  differences, and overall L2 standardized mean shift 2.4029.
- `run-scheduling-transfer` now supports episode-level `--batch-size`
  checkpoint flushing and `--resume`; the trusted v2 run records
  `batch_size=10`.
- Batch-size invariance is tested: batch size 1 and 2 produce identical
  zero-shot method metrics, per-instance rows, and state-level proposal cache
  under the same seed.
- `scheduling_transfer_audit.json` records requirement-by-requirement status:
  core transfer harness requirements are met for this simulated zero-shot run.

Consistent finding:

- Solver-trained MLP transfers zero-shot from solver-induced semantic states
  to LLM-induced semantic states.
- It strongly improves success over Random and Hand heuristic.
- It also beats the online LLM-value evaluator while using about 4.4% of the
  LLM tokens used by LLM-value evaluation.
- Oracle is only slightly above MLP in success, so the learned ranker is near
  the proposal-support success ceiling in this setup.
- With certification disabled, mean nodes are close to the node budget for all
  policies, so this table supports success/gap transfer more strongly than a
  node-reduction claim.
- Ranking diagnostics are mixed but informative: MLP has lower regret than the
  hand heuristic and LLM-value evaluator, but lower pairwise accuracy than
  LLM-value. End-to-end success is the stronger evidence for this policy.

What this supports:

- Exact solver traces can act as scalable supervision for transferable
  test-time search priors.

## Scheduling Representation Ablation

Trusted run:

- `runs/scheduling_representation_200_50_100_seed0`

| Policy | Success | Gap | Mean nodes |
| --- | ---: | ---: | ---: |
| Random | 0.80 | 0.0266 | 118.47 |
| Hand heuristic | 0.10 | 0.8843 | 119.87 |
| MLP | 0.93 | 0.0269 | 113.14 |
| Tree-GNN | 0.93 | 0.0219 | 113.53 |
| Graph-GNN | 0.92 | 0.0096 | 114.70 |
| Oracle | 1.00 | 0.0000 | 112.33 |

Consistent finding:

- Learned priority helps.
- Graph-GNN improves objective gap in this run, but it does not dominate MLP
  on success or nodes.

What this supports:

- Graph structure remains a representation ablation, not the main proven
  mechanism.

## Scheduling Certification

Trusted run:

- `runs/scheduling_certification_200_50_100_seed0_budget200`

| Policy | Success | Gap | Mean nodes | Total tokens | Certificates | Wrong prunes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned priority only | 0.97 | 0.0036 | 197.87 | 3401121 | 0 | 0 |
| learned priority + certified pruning | 0.97 | 0.0036 | 180.61 | 3150662 | 17507 | 0 |

Consistent finding:

- With a looser node budget, certified pruning preserves success and objective
  gap while reducing nodes and tokens.
- Wrong-prune rate is zero in this run.

What this supports:

- Certification is useful as a full-system add-on, but it should remain
  disabled in the pure zero-shot transfer table.

## Scheduling Few-Shot Adaptation With Source Replay

Trusted run:

- `runs/scheduling_fewshot_curve_replay_seed0`

| Regime | Target traces | Source replay pairs | Success | Gap | Mean nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Solver-only | 0 | 0 | 0.91 | 0.0446 | 115.34 |
| Solver pretrain + target adapt | 10 | 400 | 0.94 | 0.0238 | 114.42 |
| Target-trajectory trained | 10 | 0 | 0.86 | 0.0244 | 116.18 |
| Solver pretrain + target adapt | 50 | 2000 | 0.70 | 0.2075 | 117.09 |
| Target-trajectory trained | 50 | 0 | 0.74 | 0.1624 | 117.88 |
| Solver pretrain + target adapt | 100 | 4000 | 0.66 | 0.2575 | 116.74 |
| Target-trajectory trained | 100 | 0 | 0.63 | 0.2573 | 116.65 |
| Solver pretrain + target adapt | 500 | 20000 | 0.88 | 0.0491 | 116.33 |
| Target-trajectory trained | 500 | 0 | 0.86 | 0.0756 | 116.91 |

Consistent finding:

- 10-shot adaptation improves over solver-only.
- Larger target adaptation is unstable even with equal-count source replay.
- 500-shot recovers much of the lost performance but does not beat the
  10-shot setting.

What this supports:

- Few-shot target adaptation is not yet a clean positive result.
- It should be treated as diagnostic or appendix material unless the
  adaptation procedure is improved and made stable.

## Current Paper-Level Claims

Supported:

- Learned search priority works in exact/offline and real-LLM settings.
- Game24 shows a clean quality-compute result: MLP matches ToT success while
  saving substantial LLM tokens.
- Scheduling shows the stronger transfer result: solver-trained priority
  transfers zero-shot to LLM-induced semantic states.
- Certification can reduce search cost without observed wrong pruning when
  added after the ranking result is isolated.

Not yet supported:

- Graph structure is necessary.
- Few-shot target adaptation reliably shifts the sample-efficiency curve left.

## Scheduling Real-LLM Pilot

Trusted run:

- `runs/scheduling_llm_pilot_real_20_seed0`

Setup:

- `gpt-4o-mini`
- 20 target Scheduling episodes.
- Source train: 200 solver-induced semantic scheduling instances.
- Source validation: 50.
- Node budget 80, action budget 6.
- Shared real GPT proposal replay.
- Certified pruning disabled.
- Solver-trained MLP uses no target training trajectories.

| Policy | Success | Gap | Mean nodes | Top-1 | Pairwise | Regret | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.70 | 0.0565 | 80.00 | 0.3219 | 0.4993 | 4.4563 | 4062669 |
| Hand heuristic | 0.10 | 0.9000 | 80.00 | 0.5119 | 0.5972 | 1.8019 | 4062669 |
| Solver-trained MLP | 0.90 | 0.0091 | 80.00 | 0.5050 | 0.5928 | 2.0444 | 4062669 |
| Oracle | 1.00 | 0.0000 | 80.00 | 1.0000 | 1.0000 | 0.0000 | 4062669 |

Real proposal diagnostics:

- Request count: 6400.
- State count: 5605.
- Cache hits: 795.
- ProposalRecall@K: 0.9964.
- Empty-valid-action rate: 0.0036.
- Invalid actions filtered: 3548.
- Duplicate actions filtered: 622.
- Parse failures: 0.
- Source-target domain shift: 14977 source states, 5605 real target states,
  overall L2 standardized mean shift 2.4131.

Consistent finding:

- The real Scheduling GPT pilot supports the same transfer story as the
  simulated Level-II transfer: source-only solver-trained MLP improves success
  and objective gap over Random and Hand heuristic under shared proposal
  replay.
- Oracle reaches 1.00 success while MLP reaches 0.90, so there is still a
  ranking/search-efficiency gap, but proposal support is strong.
- All non-oracle methods hit the node budget, so this pilot primarily supports
  success/gap transfer, not node reduction.

Runner support:

- `preflight-scheduling-llm-pilot` checks API-key presence, existing resume
  progress, split sizes, and worst-case request/token upper bounds without
  calling the network or printing secrets.
- `--batch-size` controls episode-level checkpoint flushing and progress
  printing.
- `--resume` reloads completed per-instance rows and proposal-cache rows, then
  skips completed renderings without duplicating rows.
- Metrics and command manifest record `batch_size` and `resume`.

Verified with:

- `python3 -m cgr_exp.cli preflight-scheduling-llm-pilot ...`
- `python3 -m cgr_exp.cli run-scheduling-llm-pilot ... --batch-size 2 --mock --overwrite`

Current verification:

- `python3 -m unittest tests.unit.test_scheduling_transfer`
- `python3 -m unittest discover`
- Full suite result on 2026-08-27: 128 tests passed.
- `python3 -m cgr_exp.cli run-scheduling-llm-pilot ... --batch-size 2 --mock --resume`
- `python3 -m unittest discover`

The full test suite passed with 128 tests.
