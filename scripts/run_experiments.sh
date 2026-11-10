#!/usr/bin/env bash
# =============================================================================
#  Re-run CasCL experiments: train each preset, then evaluate every checkpoint.
#
#  Usage:
#    bash scripts/run_experiments.sh [options] <preset|group> [<preset|group> ...] [-- KEY=VALUE ...]
#
#  Examples:
#    bash scripts/run_experiments.sh --list                          # show presets and groups
#    bash scripts/run_experiments.sh cascl                           # one preset
#    bash scripts/run_experiments.sh baselines ablations             # two groups
#    bash scripts/run_experiments.sh all                             # everything
#    bash scripts/run_experiments.sh --data-dir data/example --quick cascl   # smoke test on the example data
#    bash scripts/run_experiments.sh cascl -- train_config.num_train_epochs=2   # extra config overrides
#
#  Options:
#    --data-dir DIR    directory with train.json / test.json / documents.json   (default: data)
#    --config FILE     YAML config                                     (default: configs/qwen3_0.6b_cascl.yml)
#    --models-dir DIR  where runs/checkpoints are written              (default: models)
#    --eval-dir DIR    where evaluation JSON files are written         (default: eval)
#    --base-model ID   base encoder (HF id or local path)              (default: from the config)
#    --gpu ID          GPU index → CUDA_VISIBLE_DEVICES                 (default: 0)
#    --sdpa            use PyTorch SDPA attention instead of flash-attention 2
#    --quick           tiny run (batch 8, 1 epoch, checkpoint every 10 steps) to check the pipeline
#    --no-eval         train only
#    --eval-only       skip training; evaluate the existing checkpoints of the most recent run
#                      of each preset found in --models-dir
#    --last-only       evaluate only the last checkpoint of each run
#    --dry-run         print the commands without running them
#    --list            list presets and groups, then exit
#
#  Environment:
#    PYTHON            python interpreter (default: python)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."            # repo root

PYTHON="${PYTHON:-python}"
CONFIG="configs/qwen3_0.6b_cascl.yml"
DATA_DIR="data"
MODELS_DIR="models"
EVAL_DIR="eval"
GPU="0"
BASE_MODEL=""
ATTN=""
QUICK=0
DO_TRAIN=1
DO_EVAL=1
LAST_ONLY=0
DRY_RUN=0

# -----------------------------------------------------------------------------
#  Presets: name → train.py arguments.
#  "Final setting" = configs/qwen3_0.6b_cascl.yml as is
#  (Qwen3-Embedding-0.6B, batch 64, lr 1e-5, 4 epochs, γ=15, δ₀=0.5, λ₂=0.5).
# -----------------------------------------------------------------------------
declare -A PRESET
declare -a ORDER

add() { PRESET["$1"]="$2"; ORDER+=("$1"); }

# ── CasCL and its variants (final setting) ──
add cascl                          "--loss cascl"
add cascl_supcon                   "--loss cascl_supcon"
add cascl_rectified_a0.05          "--loss cascl_rectified --alpha_min 0.05"
add cascl_rectified_a0.1           "--loss cascl_rectified --alpha_min 0.1"
add cascl_rectified_a0.15          "--loss cascl_rectified --alpha_min 0.15"
add cascl_rectified_a0.2           "--loss cascl_rectified --alpha_min 0.2"
add cascl_two_regime_m0.05         "--loss cascl_two_regime --margin_safe 0.05"
add cascl_two_regime_m0.1          "--loss cascl_two_regime --margin_safe 0.1"
add cascl_two_regime_m0.05_gain_on "--loss cascl_two_regime --margin_safe 0.05 --use_relevance_gain on"
add cascl_two_regime_m0.1_gain_on  "--loss cascl_two_regime --margin_safe 0.1 --use_relevance_gain on"
add cascl_neg_relative             "--loss cascl_neg_relative --alpha_mode relative"
add cascl_ga2                      "--loss cascl --set train_config.gradient_accumulation_steps=2"

# ── Baselines (final setting; loss hyper-parameters from their papers) ──
for b in infonce circle triplet mnrl margin_mse rince cloc; do
  add "baseline_$b" "--loss $b"
done

# ── Ablations of CasCL (final setting) ──
for a in no_infonce no_cascade no_selfpaced no_margins no_grading no_relevance_weight; do
  add "ablation_$a" "--loss cascl_$a"
done

# ── Early experiments (batch 128) ──
BS128="train_config.per_device_train_batch_size=128"
add early_cascl                    "--loss cascl --set $BS128"
add early_infonce_only             "--loss cascl_no_cascade --set $BS128"
for g in 32 64 128; do
  add "early_cascl_delta_warmup_g$g" \
      "--loss cascl --set $BS128 train_config.delta_warmup=true train_config.gamma=$g"
done
EARLY_ABL="$BS128 train_config.lambda2=0.4 train_config.delta_0=0.3 train_config.gamma=64"
add early_ablation_no_infonce          "--loss cascl_no_infonce --set $EARLY_ABL"
add early_ablation_no_relevance_weight "--loss cascl_no_relevance_weight --set $EARLY_ABL"
for b in circle margin_mse mnrl triplet; do
  add "early_baseline_$b" "--loss $b --set $BS128"
done

# ── Groups ──
declare -A GROUP
GROUP[main]="cascl cascl_supcon cascl_rectified_a0.05 cascl_rectified_a0.1 cascl_rectified_a0.15 cascl_rectified_a0.2 cascl_two_regime_m0.05 cascl_two_regime_m0.1 cascl_two_regime_m0.05_gain_on cascl_two_regime_m0.1_gain_on cascl_neg_relative cascl_ga2"
GROUP[baselines]="$(printf '%s ' "${ORDER[@]}" | tr ' ' '\n' | grep '^baseline_' | tr '\n' ' ')"
GROUP[ablations]="$(printf '%s ' "${ORDER[@]}" | tr ' ' '\n' | grep '^ablation_' | tr '\n' ' ')"
GROUP[early]="$(printf '%s ' "${ORDER[@]}" | tr ' ' '\n' | grep '^early_' | tr '\n' ' ')"
GROUP[all]="base ${ORDER[*]}"
# "base" = evaluation of the base model, no training.

list_presets() {
  echo "Presets (name → train.py arguments):"
  echo "  base                                 (no training) evaluate the base model"
  for n in "${ORDER[@]}"; do printf "  %-36s %s\n" "$n" "${PRESET[$n]}"; done
  echo
  echo "Groups:"
  for g in main baselines ablations early all; do printf "  %-10s %s\n" "$g" "${GROUP[$g]}"; done
}

# -----------------------------------------------------------------------------
#  Arguments
# -----------------------------------------------------------------------------
TARGETS=()
EXTRA_SET=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)   DATA_DIR="$2"; shift 2 ;;
    --config)     CONFIG="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --eval-dir)   EVAL_DIR="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --gpu)        GPU="$2"; shift 2 ;;
    --sdpa)       ATTN="sdpa"; shift ;;
    --quick)      QUICK=1; shift ;;
    --no-eval)    DO_EVAL=0; shift ;;
    --eval-only)  DO_TRAIN=0; shift ;;
    --last-only)  LAST_ONLY=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --list)       list_presets; exit 0 ;;
    -h|--help)    sed -n '2,36p' "$0"; exit 0 ;;
    --)           shift; EXTRA_SET=("$@"); break ;;
    -*)           echo "Unknown option: $1" >&2; exit 2 ;;
    *)            TARGETS+=("$1"); shift ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "Nothing to run. Give a preset or a group (see --list)." >&2; exit 2
fi

# Expand groups into presets
RUNS=()
for t in "${TARGETS[@]}"; do
  if [[ -n "${GROUP[$t]:-}" ]]; then
    for p in ${GROUP[$t]}; do RUNS+=("$p"); done
  elif [[ "$t" == "base" || -n "${PRESET[$t]:-}" ]]; then
    RUNS+=("$t")
  else
    echo "Unknown preset or group: $t (see --list)" >&2; exit 2
  fi
done

for f in train.json test.json documents.json; do
  if [[ ! -f "$DATA_DIR/$f" ]]; then
    echo "Missing $DATA_DIR/$f — see README 'Data format', or use --data-dir data/example" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$MODELS_DIR" "$EVAL_DIR"

# Config overrides shared by every run
SET=()
[[ -n "$BASE_MODEL" ]] && SET+=("model_config.base_model_reference=$BASE_MODEL")
[[ -n "$ATTN" ]] && SET+=("model_config.attn_implementation=$ATTN")
if [[ $QUICK -eq 1 ]]; then
  SET+=("train_config.per_device_train_batch_size=8" "train_config.per_device_eval_batch_size=16"
        "train_config.num_train_epochs=1" "train_config.eval_steps=5" "train_config.save_steps=10"
        "train_config.save_total_limit=1" "train_config.logging_steps=1")
fi
SET+=("${EXTRA_SET[@]}")

EVAL_ARGS=(--test_dataset_path "$DATA_DIR/test.json" --document_dataset_path "$DATA_DIR/documents.json")
[[ -n "$ATTN" ]] && EVAL_ARGS+=(--attn_implementation "$ATTN")

run() {
  echo "+ $*"
  [[ $DRY_RUN -eq 1 ]] || "$@"
}

evaluate_checkpoint() {  # $1 = model path, $2 = output json
  if [[ -f "$2" ]]; then echo "  ✓ already evaluated: $2"; return; fi
  run "$PYTHON" cascl_evaluate.py --model_path "$1" "${EVAL_ARGS[@]}" --output_file "$2"
}

evaluate_run() {  # $1 = run directory
  local run_dir="$1" name ckpts
  name="$(basename "$run_dir")"
  mapfile -t ckpts < <(ls -d "$run_dir"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
  if [[ ${#ckpts[@]} -eq 0 ]]; then echo "  ! no checkpoint in $run_dir"; return; fi
  [[ $LAST_ONLY -eq 1 ]] && ckpts=("${ckpts[-1]}")
  for c in "${ckpts[@]}"; do
    evaluate_checkpoint "$c" "$EVAL_DIR/${name}_$(basename "$c").json"
  done
}

RUN_DIR_FILE="$(mktemp)"
trap 'rm -f "$RUN_DIR_FILE"' EXIT

for p in "${RUNS[@]}"; do
  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "  $p"
  echo "════════════════════════════════════════════════════════════════════"

  if [[ "$p" == "base" ]]; then
    model="${BASE_MODEL:-$(grep -E '^\s*base_model_reference:' "$CONFIG" | head -1 | sed -E "s/.*:\s*['\"]?([^'\"#]+)['\"]?.*/\1/" | xargs)}"
    [[ $DO_EVAL -eq 1 ]] && evaluate_checkpoint "$model" "$EVAL_DIR/$(basename "$model")_base.json"
    continue
  fi

  # shellcheck disable=SC2086
  read -r -a ARGS <<< "${PRESET[$p]}"
  # merge the preset's own --set values with the shared ones
  PRESET_SET=()
  TRAIN_ARGS=()
  in_set=0
  for a in "${ARGS[@]}"; do
    if [[ "$a" == "--set" ]]; then in_set=1; continue; fi
    if [[ $in_set -eq 1 && "$a" == *=* ]]; then PRESET_SET+=("$a"); else in_set=0; TRAIN_ARGS+=("$a"); fi
  done
  ALL_SET=("${PRESET_SET[@]}" "${SET[@]}")   # shared overrides (e.g. --quick) win

  CMD=("$PYTHON" train.py --config "$CONFIG" --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR"
       --run_dir_file "$RUN_DIR_FILE" "${TRAIN_ARGS[@]}")
  [[ ${#ALL_SET[@]} -gt 0 ]] && CMD+=(--set "${ALL_SET[@]}")

  if [[ $DO_TRAIN -eq 1 ]]; then
    : > "$RUN_DIR_FILE"
    run "${CMD[@]}"
    run_dir="$(tail -n 1 "$RUN_DIR_FILE" 2>/dev/null || true)"
  else
    # --eval-only: ask train.py for the run name, then match it with any date
    pattern="$("${CMD[@]}" --print_run_name 2>/dev/null | tail -n 1)"
    date_part="$(date +%d%m%Y)"
    glob="${pattern/$date_part/*}"
    run_dir="$(ls -dt "$MODELS_DIR"/$glob 2>/dev/null | head -1 || true)"
  fi

  if [[ $DO_EVAL -eq 1 && $DRY_RUN -eq 0 ]]; then
    if [[ -n "${run_dir:-}" && -d "$run_dir" ]]; then
      evaluate_run "$run_dir"
    else
      echo "  ! no run directory found for $p"
    fi
  fi
done

echo
echo "Done. Evaluation files are in $EVAL_DIR/ — visualise them with: python tools/plot_eval.py --eval_dir $EVAL_DIR"
