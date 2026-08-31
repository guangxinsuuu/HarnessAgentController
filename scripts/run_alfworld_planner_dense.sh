#!/usr/bin/env zsh
set -euo pipefail

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/cgr-alfworld/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
SEED="${SEED:-0}"
if [[ "${MODE}" == "smoke" ]]; then
  TRAIN_EPISODES="${TRAIN_EPISODES:-1}"
  EVAL_EPISODES="${EVAL_EPISODES:-1}"
  MAX_STEPS="${MAX_STEPS:-8}"
elif [[ "${MODE}" == "pilot" ]]; then
  TRAIN_EPISODES="${TRAIN_EPISODES:-50}"
  EVAL_EPISODES="${EVAL_EPISODES:-20}"
  MAX_STEPS="${MAX_STEPS:-40}"
else
  TRAIN_EPISODES="${TRAIN_EPISODES:-50}"
  EVAL_EPISODES="${EVAL_EPISODES:-20}"
  MAX_STEPS="${MAX_STEPS:-40}"
fi
EPOCHS="${EPOCHS:-10}"
MAX_PAIRS_PER_STATE="${MAX_PAIRS_PER_STATE:-16}"
if [[ "${MODE}" == "smoke" ]]; then
  MAX_FRONTIER_ACTIONS="${MAX_FRONTIER_ACTIONS:-4}"
else
  MAX_FRONTIER_ACTIONS="${MAX_FRONTIER_ACTIONS:-6}"
fi
FRONTIER_STRATEGY="${FRONTIER_STRATEGY:-priority}"
COUNTERFACTUAL_ROLLOUT_STEPS="${COUNTERFACTUAL_ROLLOUT_STEPS:-30}"
PLANNER_TIMEOUT_SECONDS="${PLANNER_TIMEOUT_SECONDS:-3}"
MAX_EPISODE_LABEL_SECONDS="${MAX_EPISODE_LABEL_SECONDS:-60}"
PERTURBATION_PROB="${PERTURBATION_PROB:-0.0}"
WARMUP_RANDOM_STEPS="${WARMUP_RANDOM_STEPS:-0}"
LABEL_SOURCE="${LABEL_SOURCE:-planner}"
EVAL_SPLIT="${EVAL_SPLIT:-eval_in_distribution}"
OUT_ROOT="${OUT_ROOT:-runs/alfworld_planner_dense_seed${SEED}}"
TRAIN_DIR="${TRAIN_DIR:-${OUT_ROOT}_train}"
AUDIT_DIR="${AUDIT_DIR:-${OUT_ROOT}_audit}"
EVAL_DIR="${EVAL_DIR:-${OUT_ROOT}_eval}"
RANKER_CHECKPOINT="${RANKER_CHECKPOINT:-${TRAIN_DIR}/alfworld_dense_ranker.json}"
POLICIES="${POLICIES:-Random,Hand heuristic,ALFWorld dense-ranker,ALFWorld planner diagnostic}"
RESUME="${RESUME:-1}"

export ALFWORLD_DATA
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_alfworld_planner_dense.sh [audit|train|eval|smoke|pilot|all]

This script uses ALFWorld/TextWorld PDDL planner cost-to-go labels.
It does not use an LLM or API key.
`smoke` only checks that the pipeline runs; use `pilot` for performance.

Common environment overrides:
  TRAIN_EPISODES=50 EVAL_EPISODES=20 MAX_STEPS=40 EPOCHS=20
  MAX_FRONTIER_ACTIONS=6 PLANNER_TIMEOUT_SECONDS=3
  FRONTIER_STRATEGY=score-diverse
  MAX_EPISODE_LABEL_SECONDS=60
  PERTURBATION_PROB=0.15 WARMUP_RANDOM_STEPS=2
  LABEL_SOURCE=planner     # exact TextWorld/PDDL cost-to-go labels
  LABEL_SOURCE=walkthrough # fast proxy labels; manifest marks planner labels false
  RESUME=1       # default; reuses existing dense label cache if present
  RESUME=0       # clears an existing training cache and starts over
  EVAL_SPLIT=eval_in_distribution
EOF
}

run_audit() {
  local cache_args=()
  if [[ "${AUDIT_CACHE_DIR:-}" != "" ]]; then
    cache_args=("--cache-dir" "${AUDIT_CACHE_DIR}")
  fi
  "${PYTHON_BIN}" -m cgr_exp.cli audit-alfworld-dense-labels \
    --output-dir "${AUDIT_DIR}" \
    --seed "${SEED}" \
    "${cache_args[@]}" \
    --episodes "${TRAIN_EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --max-pairs-per-state "${MAX_PAIRS_PER_STATE}" \
    --max-frontier-actions "${MAX_FRONTIER_ACTIONS}" \
    --frontier-strategy "${FRONTIER_STRATEGY}" \
    --counterfactual-rollout-steps "${COUNTERFACTUAL_ROLLOUT_STEPS}" \
    --planner-timeout-seconds "${PLANNER_TIMEOUT_SECONDS}" \
    --max-episode-label-seconds "${MAX_EPISODE_LABEL_SECONDS}" \
    --label-source "${LABEL_SOURCE}" \
    --warmup-random-steps "${WARMUP_RANDOM_STEPS}" \
    --perturbation-prob "${PERTURBATION_PROB}" \
    --overwrite
}

run_train() {
  local resume_args=("--overwrite")
  if [[ "${RESUME}" == "1" ]]; then
    resume_args=("--resume")
  fi
  "${PYTHON_BIN}" -m cgr_exp.cli run-alfworld-dense-train \
    --output-dir "${TRAIN_DIR}" \
    --seed "${SEED}" \
    --train-episodes "${TRAIN_EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --epochs "${EPOCHS}" \
    --max-pairs-per-state "${MAX_PAIRS_PER_STATE}" \
    --max-frontier-actions "${MAX_FRONTIER_ACTIONS}" \
    --frontier-strategy "${FRONTIER_STRATEGY}" \
    --counterfactual-rollout-steps "${COUNTERFACTUAL_ROLLOUT_STEPS}" \
    --planner-timeout-seconds "${PLANNER_TIMEOUT_SECONDS}" \
    --max-episode-label-seconds "${MAX_EPISODE_LABEL_SECONDS}" \
    --label-source "${LABEL_SOURCE}" \
    --warmup-random-steps "${WARMUP_RANDOM_STEPS}" \
    --perturbation-prob "${PERTURBATION_PROB}" \
    "${resume_args[@]}"
}

run_eval() {
  if [[ ! -f "${RANKER_CHECKPOINT}" ]]; then
    print -u2 "missing ranker checkpoint: ${RANKER_CHECKPOINT}"
    print -u2 "run train first or set RANKER_CHECKPOINT=/path/to/alfworld_dense_ranker.json"
    exit 2
  fi
  "${PYTHON_BIN}" -m cgr_exp.cli run-alfworld-dense-transfer \
    --output-dir "${EVAL_DIR}" \
    --seed "${SEED}" \
    --eval-split "${EVAL_SPLIT}" \
    --train-episodes 0 \
    --eval-episodes "${EVAL_EPISODES}" \
    --max-steps "${MAX_STEPS}" \
    --epochs 0 \
    --max-pairs-per-state "${MAX_PAIRS_PER_STATE}" \
    --max-frontier-actions "${MAX_FRONTIER_ACTIONS}" \
    --frontier-strategy "${FRONTIER_STRATEGY}" \
    --counterfactual-rollout-steps "${COUNTERFACTUAL_ROLLOUT_STEPS}" \
    --planner-timeout-seconds "${PLANNER_TIMEOUT_SECONDS}" \
    --max-episode-label-seconds "${MAX_EPISODE_LABEL_SECONDS}" \
    --label-source "${LABEL_SOURCE}" \
    --warmup-random-steps "${WARMUP_RANDOM_STEPS}" \
    --perturbation-prob "${PERTURBATION_PROB}" \
    --ranker-checkpoint "${RANKER_CHECKPOINT}" \
    --policies "${POLICIES}" \
    --overwrite
}

case "${MODE}" in
  audit)
    run_audit
    ;;
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  smoke)
    run_train
    AUDIT_CACHE_DIR="${TRAIN_DIR}" run_audit
    run_eval
    ;;
  pilot)
    run_train
    AUDIT_CACHE_DIR="${TRAIN_DIR}" run_audit
    run_eval
    ;;
  all)
    run_train
    AUDIT_CACHE_DIR="${TRAIN_DIR}" run_audit
    run_eval
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
