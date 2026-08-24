#!/usr/bin/env bash
set -euo pipefail

# Dynamic multi-GPU seed launcher with metrics extraction.
#
# Usage:
#   GPUS="0 1 2" SEEDS="1000 1001 1002 1003" \
#   CONFIG=scalable_safe_rl/config.yaml ENV_ID=ant_goal_headless \
#   ./run_seeds_dynamic.sh --critic_loss_type scaling_crl
#
# Arguments passed to this script are forwarded to scalable_safe_rl.train.
#
# Outputs default to results/<env_id>/<method>/<run_name>/:
#   ${RUN_DIR}/<seed>/seed_<seed>.log         raw training stdout/stderr
#   ${RUN_DIR}/metrics_long.tsv               one metric per row, all epochs/seeds
#   ${RUN_DIR}/metrics_final.tsv              final observed value per metric/seed
#   ${RUN_DIR}/runs.tsv                       per-seed status and log path
#   ${RUN_DIR}/policies.tsv                   per-seed final policy path
#   ${RUN_DIR}/<seed>/final_policy/           saved actor-only policy bundle
#   ${RUN_DIR}/commands.txt                   exact launcher settings and forwarded args

if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  echo "This script requires bash >= 5.1 for 'wait -n -p'." >&2
  exit 2
fi

read -r -a GPUS_ARRAY <<< "${GPUS:-0 1 2}"
read -r -a SEEDS_ARRAY <<< "${SEEDS:-1000 1001 1002 1003 1004 1005 1006 1007 1008 1009}"

CONFIG="${CONFIG:-scalable_safe_rl/config.yaml}"
ENV_ID="${ENV_ID:-point_goal_headless}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

cli_value() {
  local name_a="$1"
  local name_b="${2:-}"
  shift || true
  shift || true
  local arg
  while (( $# > 0 )); do
    arg="$1"
    if [[ "${arg}" == "${name_a}" || ( -n "${name_b}" && "${arg}" == "${name_b}" ) ]]; then
      shift
      if (( $# > 0 )); then
        printf "%s" "$1"
        return 0
      fi
      return 1
    fi
    if [[ "${arg}" == "${name_a}="* ]]; then
      printf "%s" "${arg#*=}"
      return 0
    fi
    if [[ -n "${name_b}" && "${arg}" == "${name_b}="* ]]; then
      printf "%s" "${arg#*=}"
      return 0
    fi
    shift
  done
  return 1
}

config_value() {
  local key="$1"
  "${PYTHON_BIN}" - "${CONFIG}" "${key}" <<'PY'
import sys
try:
    import yaml
except Exception:
    yaml = None

path, key = sys.argv[1:]
if yaml is None:
    print("")
    raise SystemExit(0)
try:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
except FileNotFoundError:
    data = {}
value = data.get(key, "")
print(value if value is not None else "")
PY
}

METHOD="${METHOD:-}"
if [[ -z "${METHOD}" ]]; then
  METHOD="$(cli_value "--critic-loss-type" "--critic_loss_type" "$@" || true)"
fi
if [[ -z "${METHOD}" ]]; then
  METHOD="$(config_value critic_loss_type)"
fi
METHOD="${METHOD:-unknown_method}"
RUN_NAME="${RUN_NAME:-${RUN_ID:-run_$(date +%Y%m%d-%H%M%S)}}"
RUN_ID="${RUN_ID:-${RUN_NAME}}"
LOG_ROOT="${LOG_DIR:-${RESULTS_ROOT}/${ENV_ID}/${METHOD}}"
requested_run_dir="${RUN_DIR:-${LOG_ROOT}/${RUN_NAME}}"
requested_run_name="${RUN_NAME}"
requested_run_id="${RUN_ID}"
RUN_DIR="${requested_run_dir}"
run_suffix=0
while [[ -e "${RUN_DIR}" ]]; do
  run_suffix=$((run_suffix + 1))
  RUN_DIR="${requested_run_dir}_${run_suffix}"
done
if (( run_suffix > 0 )); then
  RUN_NAME="${requested_run_name}_${run_suffix}"
  RUN_ID="${requested_run_id}_${run_suffix}"
fi
METRICS_LONG_TSV="${METRICS_LONG_TSV:-${RUN_DIR}/metrics_long.tsv}"
METRICS_FINAL_TSV="${METRICS_FINAL_TSV:-${RUN_DIR}/metrics_final.tsv}"
RUNS_TSV="${RUNS_TSV:-${RUN_DIR}/runs.tsv}"
POLICY_ROOT="${POLICY_ROOT:-${RUN_DIR}}"
POLICIES_TSV="${POLICIES_TSV:-${RUN_DIR}/policies.tsv}"
POLICY_VIS_COMMANDS_FILE="${POLICY_VIS_COMMANDS_FILE:-${RUN_DIR}/policy_visualization_commands.txt}"
COMMANDS_FILE="${COMMANDS_FILE:-${RUN_DIR}/commands.txt}"

if (( ${#GPUS_ARRAY[@]} == 0 )); then
  echo "No GPUs specified. Set GPUS, e.g. GPUS=\"0 1 2\"." >&2
  exit 2
fi

if (( ${#SEEDS_ARRAY[@]} == 0 )); then
  echo "No seeds specified. Set SEEDS, e.g. SEEDS=\"1000 1001 1002\"." >&2
  exit 2
fi

mkdir -p "${RUN_DIR}" "${POLICY_ROOT}"

printf "run_id\tseed\tgpu\tenv_id\tlog_file\tepoch\ttotal_epochs\tenv_steps\tepoch_seconds\telapsed_hours\tcategory\tmetric\tvalue\n" > "${METRICS_LONG_TSV}"
printf "run_id\tseed\tgpu\tenv_id\tlog_file\tcategory\tmetric\tvalue\tfinal_epoch\tfinal_env_steps\n" > "${METRICS_FINAL_TSV}"
printf "run_id\tseed\tgpu\tenv_id\tmethod\tstatus\texit_code\tlog_file\trun_dir\tstarted_at\tfinished_at\n" > "${RUNS_TSV}"
printf "run_id\tseed\tgpu\tenv_id\tpolicy_dir\tpolicy_file\tlog_file\n" > "${POLICIES_TSV}"
printf "# One deterministic rollout HTML per saved final policy.\n" > "${POLICY_VIS_COMMANDS_FILE}"
{
  printf "run_id=%s\n" "${RUN_ID}"
  printf "run_name=%s\n" "${RUN_NAME}"
  printf "config=%s\n" "${CONFIG}"
  printf "env_id=%s\n" "${ENV_ID}"
  printf "method=%s\n" "${METHOD}"
  printf "gpus=%s\n" "${GPUS_ARRAY[*]}"
  printf "seeds=%s\n" "${SEEDS_ARRAY[*]}"
  printf "python_bin=%s\n" "${PYTHON_BIN}"
  printf "policy_root=%s\n" "${POLICY_ROOT}"
  printf "policy_visualization_commands=%s\n" "${POLICY_VIS_COMMANDS_FILE}"
  printf "forwarded_args="
  printf "%q " "$@"
  printf "\n"
} > "${COMMANDS_FILE}"

echo "Run directory: ${RUN_DIR}"
echo "Combined metrics: ${METRICS_LONG_TSV}"
echo "Final policies: ${POLICY_ROOT}"

has_cli_arg() {
  local wanted_a="$1"
  local wanted_b="${2:-}"
  shift || true
  shift || true
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${wanted_a}" || ( -n "${wanted_b}" && "${arg}" == "${wanted_b}" ) ]]; then
      return 0
    fi
    if [[ "${arg}" == "${wanted_a}="* || ( -n "${wanted_b}" && "${arg}" == "${wanted_b}="* ) ]]; then
      return 0
    fi
  done
  return 1
}

declare -a EXTRA_TRAIN_ARGS=()
if ! has_cli_arg "--save-final-policy" "--save_final_policy" "$@"; then
  EXTRA_TRAIN_ARGS+=(--save-final-policy true)
fi
if ! has_cli_arg "--wandb-dir" "--wandb_dir" "$@"; then
  EXTRA_TRAIN_ARGS+=(--wandb-dir "${RUN_DIR}")
fi
if ! has_cli_arg "--results-root" "--results_root" "$@"; then
  EXTRA_TRAIN_ARGS+=(--results-root "${RESULTS_ROOT}")
fi
if ! has_cli_arg "--run-name" "--run_name" "$@"; then
  EXTRA_TRAIN_ARGS+=(--run-name "${RUN_NAME}")
fi

declare -A PID_TO_GPU
declare -A PID_TO_SEED
declare -A PID_TO_LOG
declare -A PID_TO_STARTED_AT

cleanup_children() {
  local pids=("${!PID_TO_GPU[@]}")
  if (( ${#pids[@]} > 0 )); then
    echo "Stopping ${#pids[@]} active training process(es)..." >&2
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM

next_seed_idx=0
exit_code=0

parse_log_metrics() {
  local log_file="$1"
  local seed="$2"
  local gpu="$3"
  local seed_dir="${RUN_DIR}/${seed}"
  local per_seed_long="${seed_dir}/metrics_long.tsv"
  local per_seed_final="${seed_dir}/metrics_final.tsv"

  "${PYTHON_BIN}" - "${log_file}" "${RUN_ID}" "${seed}" "${gpu}" "${ENV_ID}" "${per_seed_long}" "${per_seed_final}" <<'PY'
import math
import json
import re
import sys
from pathlib import Path

log_file, run_id, seed, gpu, env_id, long_path, final_path = sys.argv[1:]
epoch_re = re.compile(
    r"^Epoch\s+(\d+)/(\d+)\s+env_steps\s+([0-9,]+)\s+epoch\s+([0-9.eE+-]+)s\s+elapsed\s+([0-9.eE+-]+)h"
)

records = []
current = None
last_category = ""

def parse_value(text: str) -> float:
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return math.nan

lines = Path(log_file).read_text(encoding="utf-8", errors="replace").splitlines()

for raw_line in lines:
    if not raw_line.startswith("METRICS_JSON\t"):
        continue
    try:
        payload = json.loads(raw_line.split("\t", 1)[1])
    except (IndexError, json.JSONDecodeError):
        continue
    current = {
        "epoch": int(payload.get("epoch", 0)),
        "total_epochs": int(payload.get("total_epochs", 0)),
        "env_steps": int(payload.get("env_steps", 0)),
        "epoch_seconds": float(payload.get("epoch_seconds", math.nan)),
        "elapsed_hours": float(payload.get("elapsed_hours", math.nan)),
    }
    for item in payload.get("metrics", []):
        value = item.get("value", math.nan)
        if not isinstance(value, (int, float)):
            value = parse_value(str(value))
        records.append(
            {
                **current,
                "category": str(item.get("category", "")),
                "metric": str(item.get("metric", "")),
                "value": float(value),
            }
        )

if not records:
    for raw_line in lines:
        line = raw_line.rstrip()
        match = epoch_re.match(line)
        if match:
            current = {
                "epoch": int(match.group(1)),
                "total_epochs": int(match.group(2)),
                "env_steps": int(match.group(3).replace(",", "")),
                "epoch_seconds": float(match.group(4)),
                "elapsed_hours": float(match.group(5)),
            }
            last_category = ""
            continue

        if current is None or "|" not in line:
            continue

        left, rest = line.split("|", 1)
        category = left.strip()
        if category:
            last_category = category
        else:
            category = last_category
        if not category:
            continue

        for cell in rest.split("|"):
            parts = cell.strip().split()
            if len(parts) < 2:
                continue
            metric = " ".join(parts[:-1])
            value = parse_value(parts[-1])
            records.append(
                {
                    **current,
                    "category": category,
                    "metric": metric,
                    "value": value,
                }
            )

with Path(long_path).open("w", encoding="utf-8") as f:
    f.write(
        "run_id\tseed\tgpu\tenv_id\tlog_file\tepoch\ttotal_epochs\tenv_steps\t"
        "epoch_seconds\telapsed_hours\tcategory\tmetric\tvalue\n"
    )
    for r in records:
        f.write(
            f"{run_id}\t{seed}\t{gpu}\t{env_id}\t{log_file}\t"
            f"{r['epoch']}\t{r['total_epochs']}\t{r['env_steps']}\t"
            f"{r['epoch_seconds']}\t{r['elapsed_hours']}\t"
            f"{r['category']}\t{r['metric']}\t{r['value']}\n"
        )

latest_by_metric = {}
for r in records:
    latest_by_metric[(r["category"], r["metric"])] = r

with Path(final_path).open("w", encoding="utf-8") as f:
    f.write(
        "run_id\tseed\tgpu\tenv_id\tlog_file\tcategory\tmetric\tvalue\t"
        "final_epoch\tfinal_env_steps\n"
    )
    for (category, metric), r in sorted(latest_by_metric.items()):
        f.write(
            f"{run_id}\t{seed}\t{gpu}\t{env_id}\t{log_file}\t"
            f"{category}\t{metric}\t{r['value']}\t{r['epoch']}\t{r['env_steps']}\n"
        )
PY

  tail -n +2 "${per_seed_long}" >> "${METRICS_LONG_TSV}"
  tail -n +2 "${per_seed_final}" >> "${METRICS_FINAL_TSV}"
}

record_policy_path() {
  local log_file="$1"
  local seed="$2"
  local gpu="$3"
  local policy_dir=""
  local policy_file=""

  if [[ -f "${log_file}" ]]; then
    policy_dir="$("${PYTHON_BIN}" - "${log_file}" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"Saved final policy to (.+)", text)
print(matches[-1].strip() if matches else "")
PY
)"
  fi

  if [[ -n "${policy_dir}" ]]; then
    policy_file="${policy_dir}/policy.pkl"
    printf "%q -m scalable_safe_rl.visualize_policy --policy-dir %q --output %q\n" \
      "${PYTHON_BIN}" "${policy_dir}" "${policy_dir}/eval_rollout.html" >> "${POLICY_VIS_COMMANDS_FILE}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${RUN_ID}" "${seed}" "${gpu}" "${ENV_ID}" "${policy_dir}" "${policy_file}" "${log_file}" >> "${POLICIES_TSV}"
}

launch_on_gpu() {
  local gpu="$1"
  shift
  local seed="${SEEDS_ARRAY[${next_seed_idx}]}"
  local seed_dir="${RUN_DIR}/${seed}"
  local log_file="${seed_dir}/seed_${seed}.log"
  local started_at
  started_at="$(date -Is)"
  mkdir -p "${seed_dir}"
  next_seed_idx=$((next_seed_idx + 1))

  echo "Starting seed=${seed} on GPU=${gpu}; log=${log_file}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m scalable_safe_rl.train \
      --config "${CONFIG}" \
      --env_id "${ENV_ID}" \
      --seed "${seed}" \
      --seed-output-dir "${seed_dir}" \
      --track false \
      "${EXTRA_TRAIN_ARGS[@]}" \
      "$@"
  ) > "${log_file}" 2>&1 &

  local pid=$!
  PID_TO_GPU["${pid}"]="${gpu}"
  PID_TO_SEED["${pid}"]="${seed}"
  PID_TO_LOG["${pid}"]="${log_file}"
  PID_TO_STARTED_AT["${pid}"]="${started_at}"
}

for gpu in "${GPUS_ARRAY[@]}"; do
  if (( next_seed_idx >= ${#SEEDS_ARRAY[@]} )); then
    break
  fi
  launch_on_gpu "${gpu}" "$@"
done

while (( ${#PID_TO_GPU[@]} > 0 )); do
  finished_pid=""
  if wait -n -p finished_pid; then
    status=0
  else
    status=$?
  fi

  gpu="${PID_TO_GPU[${finished_pid}]:-}"
  seed="${PID_TO_SEED[${finished_pid}]:-unknown}"
  log_file="${PID_TO_LOG[${finished_pid}]:-}"
  started_at="${PID_TO_STARTED_AT[${finished_pid}]:-}"
  finished_at="$(date -Is)"
  unset "PID_TO_GPU[${finished_pid}]"
  unset "PID_TO_SEED[${finished_pid}]"
  unset "PID_TO_LOG[${finished_pid}]"
  unset "PID_TO_STARTED_AT[${finished_pid}]"

  status_label="success"

  if (( status == 0 )); then
    echo "Finished seed=${seed} on GPU=${gpu}"
  else
    echo "Failed seed=${seed} on GPU=${gpu} with exit code ${status}" >&2
    exit_code="${status}"
    status_label="failed"
  fi

  if [[ -n "${log_file}" && -f "${log_file}" ]]; then
    parse_log_metrics "${log_file}" "${seed}" "${gpu}" || echo "Warning: failed to parse metrics from ${log_file}" >&2
    record_policy_path "${log_file}" "${seed}" "${gpu}" || echo "Warning: failed to record policy path from ${log_file}" >&2
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${RUN_ID}" "${seed}" "${gpu}" "${ENV_ID}" "${METHOD}" "${status_label}" "${status}" \
    "${log_file}" "${RUN_DIR}/${seed}" "${started_at}" "${finished_at}" >> "${RUNS_TSV}"

  if [[ -n "${gpu}" && ${next_seed_idx} -lt ${#SEEDS_ARRAY[@]} ]]; then
    launch_on_gpu "${gpu}" "$@"
  fi
done

if (( exit_code == 0 )); then
  echo "All seed runs finished successfully."
else
  echo "Some seed runs failed. Check logs in ${RUN_DIR}." >&2
fi

echo "Raw logs and extracted metrics are in: ${RUN_DIR}"
echo "Metrics for plotting:"
echo "  ${METRICS_LONG_TSV}"
echo "  ${METRICS_FINAL_TSV}"
echo "Run summary:"
echo "  ${RUNS_TSV}"
echo "Saved policies:"
echo "  ${POLICIES_TSV}"
echo "Policy visualization commands:"
echo "  ${POLICY_VIS_COMMANDS_FILE}"

exit "${exit_code}"
