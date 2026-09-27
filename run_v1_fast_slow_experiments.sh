#!/usr/bin/env bash
set -euo pipefail

# Server-side v1 quality experiment. Override paths and scale through environment variables.
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-screen}"
EVENTS="${EVENTS:-data/raw/retailrocket/events.csv}"
SID_REGISTRY="${SID_REGISTRY:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
QWEN_MODEL="${QWEN_MODEL:-}"
SOURCE_RESULT_ROOT="${SOURCE_RESULT_ROOT:-artifacts/v1_fast_slow_mainline}"

# refine profiles reuse the expensive Base/Qwen artifacts from the first screen.
REFINE_PROFILE=false
REFINE2_PROFILE=false
case "${RUN_MODE}" in
  refine|refine_confirm|refine_summary) REFINE_PROFILE=true ;;
  refine2|refine2_confirm|refine2_summary)
    REFINE_PROFILE=true
    REFINE2_PROFILE=true
    ;;
esac
if [[ "${REFINE2_PROFILE}" == true ]]; then
  RESULT_ROOT="${RESULT_ROOT:-artifacts/v1_fast_slow_v12}"
elif [[ "${REFINE_PROFILE}" == true ]]; then
  RESULT_ROOT="${RESULT_ROOT:-artifacts/v1_fast_slow_v11}"
else
  RESULT_ROOT="${RESULT_ROOT:-artifacts/v1_fast_slow_mainline}"
fi

SAMPLE_SEED="${SAMPLE_SEED:-17}"
PRETRAIN_SAMPLES="${PRETRAIN_SAMPLES:-100000}"
FINETUNE_SAMPLES="${FINETUNE_SAMPLES:-2000}"
VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-500}"
TEST_SAMPLES="${TEST_SAMPLES:-500}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-3}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-8}"
QWEN_MAX_INPUT_LENGTH="${QWEN_MAX_INPUT_LENGTH:-1024}"
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-512}"
QWEN_RETRY_MAX_NEW_TOKENS="${QWEN_RETRY_MAX_NEW_TOKENS:-1024}"
QWEN_GENERATION_SEED="${QWEN_GENERATION_SEED:-17}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
if [[ "${REFINE2_PROFILE}" == true ]]; then
  Q2I_WEIGHT="${Q2I_WEIGHT:-0.02}"
  Q2I_DECODER_WEIGHT="${Q2I_DECODER_WEIGHT:-0.10}"
  Q2I_CONTRASTIVE_WEIGHT="${Q2I_CONTRASTIVE_WEIGHT:-0.5}"
  Q2I_TEMPERATURE="${Q2I_TEMPERATURE:-0.1}"
  IGR_TOP_K="${IGR_TOP_K:-5}"
  IGR_RECENCY_WEIGHT="${IGR_RECENCY_WEIGHT:-0.75}"
  COHORT_MIN_LONG_HISTORY_ITEMS="${COHORT_MIN_LONG_HISTORY_ITEMS:-10}"
  INSTRUCTION_FEATURE_RESIDUAL=true
  INSTRUCTION_FEATURE_WEIGHT="${INSTRUCTION_FEATURE_WEIGHT:-0.1}"
  ZERO_INIT_INSTRUCTION_ADAPTER=true
  DISABLE_QWEN_TRIGGER_ITEM=true
elif [[ "${REFINE_PROFILE}" == true ]]; then
  Q2I_WEIGHT="${Q2I_WEIGHT:-0.05}"
  Q2I_DECODER_WEIGHT="${Q2I_DECODER_WEIGHT:-0.25}"
  Q2I_CONTRASTIVE_WEIGHT="${Q2I_CONTRASTIVE_WEIGHT:-1.0}"
  Q2I_TEMPERATURE="${Q2I_TEMPERATURE:-0.1}"
  IGR_TOP_K="${IGR_TOP_K:-5}"
  IGR_RECENCY_WEIGHT="${IGR_RECENCY_WEIGHT:-0.5}"
  COHORT_MIN_LONG_HISTORY_ITEMS="${COHORT_MIN_LONG_HISTORY_ITEMS:-10}"
  INSTRUCTION_FEATURE_RESIDUAL=false
  INSTRUCTION_FEATURE_WEIGHT="${INSTRUCTION_FEATURE_WEIGHT:-1.0}"
  ZERO_INIT_INSTRUCTION_ADAPTER=false
  DISABLE_QWEN_TRIGGER_ITEM=false
else
  Q2I_WEIGHT="${Q2I_WEIGHT:-0.2}"
  Q2I_DECODER_WEIGHT="${Q2I_DECODER_WEIGHT:-0.0}"
  Q2I_CONTRASTIVE_WEIGHT="${Q2I_CONTRASTIVE_WEIGHT:-0.0}"
  Q2I_TEMPERATURE="${Q2I_TEMPERATURE:-0.1}"
  IGR_TOP_K="${IGR_TOP_K:-10}"
  IGR_RECENCY_WEIGHT="${IGR_RECENCY_WEIGHT:-0.0}"
  COHORT_MIN_LONG_HISTORY_ITEMS="${COHORT_MIN_LONG_HISTORY_ITEMS:-10}"
  INSTRUCTION_FEATURE_RESIDUAL=false
  INSTRUCTION_FEATURE_WEIGHT="${INSTRUCTION_FEATURE_WEIGHT:-1.0}"
  ZERO_INIT_INSTRUCTION_ADAPTER=false
  DISABLE_QWEN_TRIGGER_ITEM=false
fi
DEVICE="${DEVICE:-cuda}"

CACHE_DIR="${RESULT_ROOT}/cache"
if [[ "${REFINE_PROFILE}" == true ]]; then
  BASE_CHECKPOINT_17="${BASE_CHECKPOINT_17:-${SOURCE_RESULT_ROOT}/pretrain/seed-17/base/best.pt}"
  CACHE_FILE="${CACHE_FILE:-${SOURCE_RESULT_ROOT}/cache/qwen_instruction_features.pt}"
  REASONING_FILE="${REASONING_FILE:-${SOURCE_RESULT_ROOT}/cache/qwen_instruction_reasoning.jsonl}"
else
  CACHE_FILE="${CACHE_FILE:-${CACHE_DIR}/qwen_instruction_features.pt}"
  REASONING_FILE="${REASONING_FILE:-${CACHE_DIR}/qwen_instruction_reasoning.jsonl}"
fi
LOG_DIR="${RESULT_ROOT}/logs"

mkdir -p "${CACHE_DIR}" "${LOG_DIR}"

if [[ ! -f "${EVENTS}" ]]; then
  echo "ERROR events not found: ${EVENTS}" >&2
  exit 2
fi
if [[ ! -f "${SID_REGISTRY}" ]]; then
  echo "ERROR SID registry not found: ${SID_REGISTRY}" >&2
  exit 2
fi

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  "$@" 2>&1 | tee "${log_file}"
}

base_checkpoint_for() {
  local seed="$1"
  local override=""
  case "${seed}" in
    17) override="${BASE_CHECKPOINT_17:-}" ;;
    23) override="${BASE_CHECKPOINT_23:-}" ;;
    42) override="${BASE_CHECKPOINT_42:-}" ;;
  esac
  if [[ -n "${override}" ]]; then
    printf '%s\n' "${override}"
  else
    printf '%s\n' "${RESULT_ROOT}/pretrain/seed-${seed}/base/best.pt"
  fi
}

pretrain_seed() {
  local seed="$1"
  local checkpoint
  checkpoint="$(base_checkpoint_for "${seed}")"
  if [[ "${checkpoint}" != "${RESULT_ROOT}/pretrain/seed-${seed}/base/best.pt" ]]; then
    if [[ ! -f "${checkpoint}" ]]; then
      echo "ERROR external Base checkpoint not found for seed=${seed}: ${checkpoint}" >&2
      exit 2
    fi
    echo "SKIP pretrain seed=${seed}; using external checkpoint: ${checkpoint}"
    return
  fi
  local output_dir="${RESULT_ROOT}/pretrain/seed-${seed}/base"
  if [[ -f "${output_dir}/best.pt" && -f "${output_dir}/result.json" ]]; then
    echo "SKIP completed pretrain seed=${seed}: ${output_dir}"
    return
  fi
  run_logged "${LOG_DIR}/pretrain-seed-${seed}.log" \
    "${PYTHON_BIN}" scripts/train_retailrocket.py \
      --events "${EVENTS}" \
      --sid-registry "${SID_REGISTRY}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
      --sample-seed "${SAMPLE_SEED}" \
      --variant base \
      --matched-igr-cohort \
      --max-history 20 \
      --long-history 100 \
      --igr-top-k "${IGR_TOP_K}" \
      --cohort-min-long-history-items "${COHORT_MIN_LONG_HISTORY_ITEMS}" \
      --max-train-samples "${PRETRAIN_SAMPLES}" \
      --max-validation-samples "${VALIDATION_SAMPLES}" \
      --max-test-samples "${TEST_SAMPLES}" \
      --batch-size "${BATCH_SIZE}" \
      --epochs "${PRETRAIN_EPOCHS}" \
      --learning-rate 0.0003 \
      --beam-width 10 \
      --output-dir "${output_dir}"
}

build_cache() {
  local checkpoint
  checkpoint="$(base_checkpoint_for 17)"
  if [[ -f "${CACHE_FILE}" && -f "${REASONING_FILE}" ]]; then
    echo "SKIP completed Qwen cache: ${CACHE_FILE}"
    return
  fi
  if [[ -e "${CACHE_FILE}" || -e "${REASONING_FILE}" ]]; then
    echo "ERROR incomplete cache pair; move the existing partial file(s) aside before retrying" >&2
    echo "CACHE_FILE=${CACHE_FILE}" >&2
    echo "REASONING_FILE=${REASONING_FILE}" >&2
    exit 2
  fi
  if [[ -z "${QWEN_MODEL}" || ! -d "${QWEN_MODEL}" ]]; then
    echo "ERROR set QWEN_MODEL to the local Qwen3-4B-Instruct-2507 directory" >&2
    exit 2
  fi
  run_logged "${LOG_DIR}/qwen-cache.log" \
    "${PYTHON_BIN}" scripts/cache_qwen_instructions_retailrocket.py \
      --events "${EVENTS}" \
      --checkpoint "${checkpoint}" \
      --sid-registry "${SID_REGISTRY}" \
      --model-path "${QWEN_MODEL}" \
      --output "${CACHE_FILE}" \
      --reasoning-output "${REASONING_FILE}" \
      --max-train-samples "${FINETUNE_SAMPLES}" \
      --max-validation-samples "${VALIDATION_SAMPLES}" \
      --max-test-samples "${TEST_SAMPLES}" \
      --short-history 20 \
      --long-history 100 \
      --igr-top-k "${COHORT_MIN_LONG_HISTORY_ITEMS}" \
      --sample-seed "${SAMPLE_SEED}" \
      --batch-size "${QWEN_BATCH_SIZE}" \
      --max-input-length "${QWEN_MAX_INPUT_LENGTH}" \
      --max-new-tokens "${QWEN_MAX_NEW_TOKENS}" \
      --retry-max-new-tokens "${QWEN_RETRY_MAX_NEW_TOKENS}" \
      --max-generation-retries 2 \
      --generation-seed "${QWEN_GENERATION_SEED}" \
      --progress-every-batches 10 \
      --device "${DEVICE}" \
      --dtype bfloat16
}

run_variant() {
  local phase="$1"
  local seed="$2"
  local variant="$3"
  local evaluate_test="$4"
  local output_dir="${RESULT_ROOT}/${phase}/seed-${seed}/${variant}"
  local base_checkpoint
  base_checkpoint="$(base_checkpoint_for "${seed}")"
  if [[ -f "${output_dir}/result.json" && -f "${output_dir}/best.pt" ]]; then
    echo "SKIP completed phase=${phase} seed=${seed} variant=${variant}: ${output_dir}"
    return
  fi
  local command=(
    "${PYTHON_BIN}" scripts/train_retailrocket.py
    --events "${EVENTS}"
    --sid-registry "${SID_REGISTRY}"
    --device "${DEVICE}"
    --seed "${seed}"
    --sample-seed "${SAMPLE_SEED}"
    --variant "${variant}"
    --matched-igr-cohort
    --max-history 20
    --long-history 100
    --igr-top-k "${IGR_TOP_K}"
    --cohort-min-long-history-items "${COHORT_MIN_LONG_HISTORY_ITEMS}"
    --igr-recency-weight "${IGR_RECENCY_WEIGHT}"
    --max-train-samples "${FINETUNE_SAMPLES}"
    --max-validation-samples "${VALIDATION_SAMPLES}"
    --max-test-samples "${TEST_SAMPLES}"
    --batch-size "${BATCH_SIZE}"
    --epochs "${FINETUNE_EPOCHS}"
    --learning-rate "${LEARNING_RATE}"
    --q2i-weight "${Q2I_WEIGHT}"
    --q2i-decoder-weight "${Q2I_DECODER_WEIGHT}"
    --q2i-contrastive-weight "${Q2I_CONTRASTIVE_WEIGHT}"
    --q2i-temperature "${Q2I_TEMPERATURE}"
    --instruction-feature-weight "${INSTRUCTION_FEATURE_WEIGHT}"
    --beam-width 10
    --init-checkpoint "${base_checkpoint}"
    --output-dir "${output_dir}"
  )
  if [[ "${variant}" == qwen_instruction || "${variant}" == qwen_q2i || "${variant}" == igr_qwen_q2i ]]; then
    command+=(--instruction-feature-cache "${CACHE_FILE}")
  fi
  if [[ "${INSTRUCTION_FEATURE_RESIDUAL}" == true ]]; then
    command+=(--instruction-feature-residual)
  fi
  if [[ "${ZERO_INIT_INSTRUCTION_ADAPTER}" == true ]]; then
    command+=(--zero-init-instruction-adapter)
  fi
  if [[ "${DISABLE_QWEN_TRIGGER_ITEM}" == true ]]; then
    command+=(--disable-qwen-trigger-item)
  fi
  if [[ "${evaluate_test}" == true ]]; then
    command+=(--evaluate-test)
  fi
  run_logged "${LOG_DIR}/${phase}-seed-${seed}-${variant}.log" "${command[@]}"
}

summarize_screen() {
  "${PYTHON_BIN}" scripts/summarize_v1_fast_slow.py \
    --root "${RESULT_ROOT}/screen" \
    --split validation \
    --expected-variants base qwen_instruction qwen_q2i igr_qwen_q2i
}

summarize_confirm() {
  "${PYTHON_BIN}" scripts/summarize_v1_fast_slow.py \
    --root "${RESULT_ROOT}/confirm" \
    --split test \
    --expected-variants base "${CONFIRM_VARIANT}"
}

summarize_fusion_screen() {
  "${PYTHON_BIN}" scripts/evaluate_v1_rank_fusion.py \
    --root "${RESULT_ROOT}/screen" \
    --split validation \
    --candidates qwen_instruction qwen_q2i igr_qwen_q2i
}

summarize_fusion_confirm() {
  local fusion_alpha="${FUSION_ALPHA:-}"
  if [[ -z "${fusion_alpha}" ]]; then
    echo "ERROR refine2_confirm fusion requires FUSION_ALPHA from the screen summary" >&2
    exit 2
  fi
  "${PYTHON_BIN}" scripts/evaluate_v1_rank_fusion.py \
    --root "${RESULT_ROOT}/confirm" \
    --split test \
    --candidates "${CONFIRM_VARIANT}" \
    --alpha "${fusion_alpha}"
}

if [[ "${REFINE2_PROFILE}" == true ]]; then
  echo "V12_CONFIG q2i_weight=${Q2I_WEIGHT} q2i_decoder=${Q2I_DECODER_WEIGHT} contrastive=${Q2I_CONTRASTIVE_WEIGHT} igr_top_k=${IGR_TOP_K} igr_recency=${IGR_RECENCY_WEIGHT} slow_residual_weight=${INSTRUCTION_FEATURE_WEIGHT}"
fi

case "${RUN_MODE}" in
  prepare)
    pretrain_seed 17
    build_cache
    ;;
  screen)
    pretrain_seed 17
    build_cache
    for variant in base qwen_instruction qwen_q2i igr_qwen_q2i; do
      run_variant screen 17 "${variant}" false
    done
    summarize_screen
    ;;
  refine)
    pretrain_seed 17
    build_cache
    for variant in base qwen_instruction qwen_q2i igr_qwen_q2i; do
      run_variant screen 17 "${variant}" false
    done
    summarize_screen
    ;;
  refine2)
    pretrain_seed 17
    build_cache
    for variant in base qwen_instruction qwen_q2i igr_qwen_q2i; do
      run_variant screen 17 "${variant}" false
    done
    summarize_screen
    summarize_fusion_screen
    ;;
  confirm)
    CONFIRM_VARIANT="${CONFIRM_VARIANT:-}"
    if [[ "${CONFIRM_VARIANT}" != qwen_instruction && "${CONFIRM_VARIANT}" != qwen_q2i && "${CONFIRM_VARIANT}" != igr_qwen_q2i ]]; then
      echo "ERROR set CONFIRM_VARIANT to qwen_instruction, qwen_q2i, or igr_qwen_q2i" >&2
      exit 2
    fi
    pretrain_seed 17
    build_cache
    for seed in 17 23 42; do
      pretrain_seed "${seed}"
      run_variant confirm "${seed}" base true
      run_variant confirm "${seed}" "${CONFIRM_VARIANT}" true
    done
    summarize_confirm
    ;;
  refine_confirm)
    CONFIRM_VARIANT="${CONFIRM_VARIANT:-}"
    if [[ "${CONFIRM_VARIANT}" != qwen_q2i && "${CONFIRM_VARIANT}" != igr_qwen_q2i ]]; then
      echo "ERROR refine_confirm requires CONFIRM_VARIANT=qwen_q2i or igr_qwen_q2i" >&2
      exit 2
    fi
    pretrain_seed 17
    build_cache
    for seed in 17 23 42; do
      pretrain_seed "${seed}"
      run_variant confirm "${seed}" base true
      run_variant confirm "${seed}" "${CONFIRM_VARIANT}" true
    done
    summarize_confirm
    ;;
  refine2_confirm)
    CONFIRM_VARIANT="${CONFIRM_VARIANT:-}"
    if [[ "${CONFIRM_VARIANT}" != qwen_instruction && "${CONFIRM_VARIANT}" != qwen_q2i && "${CONFIRM_VARIANT}" != igr_qwen_q2i ]]; then
      echo "ERROR refine2_confirm requires a Qwen candidate variant" >&2
      exit 2
    fi
    pretrain_seed 17
    build_cache
    for seed in 17 23 42; do
      pretrain_seed "${seed}"
      run_variant confirm "${seed}" base true
      run_variant confirm "${seed}" "${CONFIRM_VARIANT}" true
    done
    summarize_confirm
    summarize_fusion_confirm
    ;;
  summary)
    if [[ -d "${RESULT_ROOT}/screen" ]]; then
      summarize_screen
    fi
    if [[ -d "${RESULT_ROOT}/confirm" ]]; then
      CONFIRM_VARIANT="${CONFIRM_VARIANT:-}"
      if [[ -z "${CONFIRM_VARIANT}" ]]; then
        echo "ERROR summary for confirm requires CONFIRM_VARIANT" >&2
        exit 2
      fi
      summarize_confirm
    fi
    ;;
  refine_summary)
    if [[ -d "${RESULT_ROOT}/screen" ]]; then
      summarize_screen
    fi
    if [[ -d "${RESULT_ROOT}/confirm" ]]; then
      CONFIRM_VARIANT="${CONFIRM_VARIANT:-}"
      if [[ -z "${CONFIRM_VARIANT}" ]]; then
        echo "ERROR refine_summary for confirm requires CONFIRM_VARIANT" >&2
        exit 2
      fi
      summarize_confirm
    fi
    ;;
  refine2_summary)
    if [[ -d "${RESULT_ROOT}/screen" ]]; then
      summarize_screen
      summarize_fusion_screen
    fi
    if [[ -d "${RESULT_ROOT}/confirm" ]]; then
      CONFIRM_VARIANT="${CONFIRM_VARIANT:-}"
      if [[ -z "${CONFIRM_VARIANT}" ]]; then
        echo "ERROR refine2_summary for confirm requires CONFIRM_VARIANT" >&2
        exit 2
      fi
      summarize_confirm
      summarize_fusion_confirm
    fi
    ;;
  *)
    echo "ERROR unsupported RUN_MODE=${RUN_MODE}" >&2
    exit 2
    ;;
esac

echo "DONE RUN_MODE=${RUN_MODE}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "SCREEN_SUMMARY=${RESULT_ROOT}/screen/summary.md"
echo "CONFIRM_SUMMARY=${RESULT_ROOT}/confirm/summary.md"
