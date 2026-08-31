#!/usr/bin/env zsh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/cgr-alfworld/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
SEED="${SEED:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-eval_in_distribution}"
EPISODES="${EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-40}"
SHORTLIST_K="${SHORTLIST_K:-4}"
MODEL="${MODEL:-${CGR_LLM_MODEL:-gpt-4o-mini}}"
TIMEOUT="${TIMEOUT:-120}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
RANKER_CHECKPOINT="${RANKER_CHECKPOINT:-runs/alfworld_dense_v17_agg_train20_agg20_eval10_seed0_train/alfworld_dense_ranker.json}"
OUT_DIR="${OUT_DIR:-runs/alfworld_llm_shortlist_${EPISODES}_seed${SEED}}"
SHORTLIST_CACHE_PATH="${SHORTLIST_CACHE_PATH:-${OUT_DIR}/alfworld_llm_shortlist_cache.jsonl}"
POLICIES="${POLICIES:-Frozen LLM top-1,Frozen LLM shortlist + Random,Frozen LLM shortlist + Dense-ranker,Frozen LLM shortlist + Planner diagnostic}"
MOCK="${MOCK:-0}"

export ALFWORLD_DATA
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

args=()
if [[ "${MOCK}" == "1" ]]; then
  args+=("--mock")
fi

"${PYTHON_BIN}" -m cgr_exp.cli run-alfworld-llm-shortlist-transfer \
  --output-dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --eval-split "${EVAL_SPLIT}" \
  --ranker-checkpoint "${RANKER_CHECKPOINT}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --shortlist-k "${SHORTLIST_K}" \
  --model "${MODEL}" \
  --timeout "${TIMEOUT}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --shortlist-cache-path "${SHORTLIST_CACHE_PATH}" \
  --policies "${POLICIES}" \
  "${args[@]}" \
  --overwrite
