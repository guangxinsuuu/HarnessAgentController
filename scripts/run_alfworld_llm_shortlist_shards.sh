#!/usr/bin/env zsh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/cgr-alfworld/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_ROOT}/data/alfworld}"
SEED="${SEED:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-eval_in_distribution}"
CONFIG_PATH="${CONFIG_PATH:-}"
TOTAL_EPISODES="${TOTAL_EPISODES:-20}"
SHARDS="${SHARDS:-2}"
SHARD_SIZE="${SHARD_SIZE:-$(( (TOTAL_EPISODES + SHARDS - 1) / SHARDS ))}"
OUT_ROOT="${OUT_ROOT:-runs/alfworld_llm_shortlist_sharded_seed${SEED}}"
MERGED_DIR="${MERGED_DIR:-${OUT_ROOT}_merged}"
MERGED_DIAGNOSIS_DIR="${MERGED_DIAGNOSIS_DIR:-${MERGED_DIR}_diagnosis}"
MAX_STEPS="${MAX_STEPS:-40}"
SHORTLIST_K="${SHORTLIST_K:-4}"
MODEL="${MODEL:-${CGR_LLM_MODEL:-gpt-4o-mini}}"
TIMEOUT="${TIMEOUT:-120}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
RANKER_CHECKPOINT="${RANKER_CHECKPOINT:-runs/alfworld_dense_v17_agg_train20_agg20_eval10_seed0_train/alfworld_dense_ranker.json}"
POLICIES="${POLICIES:-Frozen LLM top-1,Frozen LLM shortlist + Random,Frozen LLM shortlist + Dense-ranker,Frozen LLM shortlist + Planner diagnostic}"
MOCK="${MOCK:-1}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"

export PYTHON_BIN
export PROJECT_ROOT
export SEED

shard_dirs=()
pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
  start=$((shard * SHARD_SIZE))
  count="${SHARD_SIZE}"
  if (( start >= TOTAL_EPISODES )); then
    continue
  fi
  if (( start + count > TOTAL_EPISODES )); then
    count=$((TOTAL_EPISODES - start))
  fi
  shard_dir="${OUT_ROOT}_shard${shard}"
  shard_dirs+=("${shard_dir}")
  (
    EPISODE_START="${start}" \
    EPISODES="${count}" \
    ALFWORLD_DATA="${ALFWORLD_DATA}" \
    EVAL_SPLIT="${EVAL_SPLIT}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    MAX_STEPS="${MAX_STEPS}" \
    SHORTLIST_K="${SHORTLIST_K}" \
    MODEL="${MODEL}" \
    TIMEOUT="${TIMEOUT}" \
    TEMPERATURE="${TEMPERATURE}" \
    TOP_P="${TOP_P}" \
    RANKER_CHECKPOINT="${RANKER_CHECKPOINT}" \
    POLICIES="${POLICIES}" \
    OUT_DIR="${shard_dir}" \
    SUMMARY_DIR="${shard_dir}_diagnosis" \
    SHORTLIST_CACHE_PATH="${shard_dir}/alfworld_llm_shortlist_cache.jsonl" \
    MOCK="${MOCK}" \
    RESUME="${RESUME}" \
    OVERWRITE="${OVERWRITE}" \
    SUMMARIZE=1 \
    scripts/run_alfworld_llm_shortlist.sh
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  print -u2 "one or more ALFWorld LLM shortlist shards failed"
  exit 1
fi

"${PYTHON_BIN}" -m cgr_exp.cli merge-alfworld-llm-shortlist-shards \
  --output-dir "${MERGED_DIR}" \
  --source-dirs "${shard_dirs[@]}" \
  --overwrite

"${PYTHON_BIN}" -m cgr_exp.cli summarize-alfworld-llm-shortlist \
  --output-dir "${MERGED_DIAGNOSIS_DIR}" \
  --source-dir "${MERGED_DIR}" \
  --overwrite
