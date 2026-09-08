#!/usr/bin/env bash
set -uo pipefail

# Roll the hybrid (HRM-PET LoRA as first-session adaptation + routing-free
# second-order head) across every dataset with trained rank-8 checkpoints.
# Datasets whose checkpoints are missing are skipped, not fatal.
#
# Usage: bash training_scripts/rollout_hybrid_4090.sh [SEED]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_ROOT}/hrm-pet-output}"
SEED="${1:-${SEED:-42}}"

# prefix | dataset string | config module
ENTRIES=(
  "imr|Split-Imagenet-R|imr_lora"
  "cifar100|Split-CIFAR100|cifar100_lora"
  "cub200|Split-CUB200|imr_lora"
)

complete() { local d="$1" t; for t in $(seq 1 10); do [[ -s "${d}/checkpoint/task${t}_checkpoint.pth" ]] || return 1; done; }

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r prefix dataset config <<< "${entry}"
  run_dir="${OUTPUT_ROOT}/${prefix}_lora_rank8_baseline_10tasks_seed${SEED}"
  tii_dir="${OUTPUT_ROOT}/${prefix}_tii_original_10tasks_seed${SEED}"
  echo "==================== ${dataset} (seed ${SEED}) ===================="
  complete "${run_dir}" || { echo "  skip: incomplete baseline $(basename "${run_dir}")"; continue; }
  complete "${tii_dir}" || { echo "  skip: incomplete TII $(basename "${tii_dir}")"; continue; }
  echo "  baseline: $(basename "${run_dir}")"
  echo "  TII:      $(basename "${tii_dir}")"
  DATASET="${dataset}" TII_DIR="${tii_dir}" SEED="${SEED}" CONFIG="${config}" \
  CALIBRATE="${CALIBRATE:-1}" RP_DIM="${RP_DIM:-10000}" RP_LAMBDA="${RP_LAMBDA:-10000}" \
    bash "${SCRIPT_DIR}/eval_rp_head_any_4090.sh" "${run_dir}" 2>&1 \
    | grep -E --line-buffered "till task10|delta=|GATE|LoRA/sample|temperature|Refusing|Missing|rror"
done
echo "==================== rollout done ===================="
