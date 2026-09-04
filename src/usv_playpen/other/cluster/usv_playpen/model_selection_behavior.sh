#!/bin/bash
#SBATCH --job-name=model_selection_job
#SBATCH --output=logs/model_selection_%A.out
#SBATCH --error=logs/model_selection_%A.err
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=nsurname@domain.edu
#SBATCH --mail-type=FAIL

# Usage: sbatch model_selection_behavior.sh (onset|params|category|behavioral_response|multinomial|continuous)
ANALYSIS_TYPE=$1

# Operator-edited roots. Every path below DERIVES from these plus $ANALYSIS_TYPE,
# so choosing an analysis cannot leave a path aimed at another analysis' results.
# A fixed 'univariate_multinomial_results.pkl' used to be read even for a
# behavioural-response run, which would have screened on the wrong artifact.
# Any of INPUT_DATA / OUTPUT_DIR / UNIVARIATE_DIR / UNIVARIATE_PATH may be
# exported before sbatch to override the derived value.
EXPERIMENTER_ID="Name"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"
MODELING_ROOT="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling"

OUTPUT_DIR="${OUTPUT_DIR:-$MODELING_ROOT/model_selection_results}"
UNIVARIATE_DIR="${UNIVARIATE_DIR:-$MODELING_ROOT/univariate_results}"

mkdir -p logs

# Set analysis-specific defaults
TARGET_VAR="bout_durations" # Only used for 'params'
PVAL=0.01

case $ANALYSIS_TYPE in
  onset|category|multinomial|continuous)
    echo "Configuring $ANALYSIS_TYPE stepwise selection using consolidated dispatcher..."
    INPUT_DATA="${INPUT_DATA:-$MODELING_ROOT/data/modeling_male_hist4s.pkl}"
    ;;

  behavioral_response)
    # Inverted direction: predicts a BEHAVIOURAL feature and adds the partner's
    # vocal block as the final step. Writes one resumable checkpoint per step.
    #   --anchor           IS honoured (forces the top-screened feature as step 0).
    #   --target_variable  ignored: the response feature is a settings key
    #                      (behavioral_response.response_feature) and must match
    #                      the feature the extraction artifact was built with, so
    #                      it is read from settings rather than the command line.
    #   --pval             ignored: this screen accepts on the paired 1SE rule,
    #                      which has no p-value threshold to bind to.
    echo "Configuring BEHAVIOURAL-RESPONSE stepwise selection (vocal block added last)..."
    # Must be the artifact BehavioralResponsePipeline wrote: the screen's candidate
    # set comes from it, and any candidate missing from the univariate artifact is
    # reported as skipped rather than screened.
    # The extraction artifact is named
    # modeling_behavioral_response_m<idx>_<feature>_<condition>_<ts>.pkl, so it is
    # resolved by prefix rather than guessed; the newest one wins.
    if [ -z "${INPUT_DATA:-}" ]; then
      INPUT_DATA=$(ls -1t "$MODELING_ROOT"/data/modeling_behavioral_response_*.pkl \
                   2>/dev/null | head -1 || true)
    fi
    ;;

  params)
    echo "Configuring BOUT PARAMETER stepwise selection..."
    # You can override the target variable here if needed via a 2nd argument
    TARGET_VAR=${2:-"bout_durations"}
    INPUT_DATA="${INPUT_DATA:-$MODELING_ROOT/data/modeling_male_hist4s.pkl}"
    ;;

  *)
    echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
    echo "Usage: sbatch model_selection.sh [onset|params|category|behavioral_response|multinomial|continuous]"
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"

set -e

# Consolidated artifacts are timestamped -- `univariate_<tag>_<condition>_<ts>.pkl`
# -- so no fixed filename can name one. Take the newest for this analysis and
# exclude the per-feature pickles, which share the prefix.
#
# The exclusion keys on the zero-padded 4-digit feature index that
# main_univariate_dispatcher puts in every per-feature name. It must NOT be
# anchored just after $ANALYSIS_TYPE: the real analysis_tag carries the
# responder and feature too (behavioral_response_m1_speed), so the index lands
# well to the right of the type. Matching `_NNNN_` anywhere is safe here, since
# the timestamp fields are 8 and 6 digits, never 4.
if [ -z "${UNIVARIATE_PATH:-}" ]; then
  UNIVARIATE_PATH=$(ls -1t "$UNIVARIATE_DIR"/univariate_"$ANALYSIS_TYPE"*.pkl 2>/dev/null \
                    | grep -Ev "_[0-9]{4}_" | head -1 || true)
fi
if [ ! -f "${UNIVARIATE_PATH:-}" ]; then
  echo "Error: no consolidated univariate artifact for '$ANALYSIS_TYPE' in $UNIVARIATE_DIR"
  echo "       expected univariate_${ANALYSIS_TYPE}_<condition>_<ts>.pkl"
  echo "       run usv_playpen.modeling.consolidate_univariate_results first"
  exit 1
fi
if [ ! -f "$INPUT_DATA" ]; then
  echo "Error: extraction pickle not found: $INPUT_DATA"
  exit 1
fi

# Echoed so the log records exactly which artifacts the run consumed.
echo "analysis    = $ANALYSIS_TYPE"
echo "input       = $INPUT_DATA"
echo "univariate  = $UNIVARIATE_PATH"

# Environment
source ${USV_PLAYPEN_PATH}.venv/bin/activate

export PYTHONUNBUFFERED=1

# JAX optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=2

echo "node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

# Execute the consolidated dispatcher
python -m usv_playpen.modeling.main_model_selection_dispatcher \
    --analysis_type "$ANALYSIS_TYPE" \
    --univariate_path "$UNIVARIATE_PATH" \
    --input_path "$INPUT_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --target_variable "$TARGET_VAR" \
    --pval "$PVAL" \
    --anchor
