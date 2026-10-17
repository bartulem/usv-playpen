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

# Usage: sbatch model_selection_behavior.sh (onset|params|category|multinomial|continuous)
ANALYSIS_TYPE=$1

# Define core variables
EXPERIMENTER_ID="Name"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"
UNIVARIATE_PATH="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling/cluster/univariate_results_multi_file/univariate_results.pkl"
INPUT_DATA="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling/modeling_manifold_qlvm_category_<cohort>_<YYYYMMDD_HHMMSS>.pkl"  # the analysis's extraction pickle
OUTPUT_DIR="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling/cluster/model_selection_multi_file"

mkdir -p logs

# The screening significance level and the bout-parameter regression target are
# read by the dispatcher from modeling_settings.json (model_params.selection_p_val,
# model_params.model_target_variable); set them there, not here.

case $ANALYSIS_TYPE in
  onset|category|multinomial|continuous)
    echo "Configuring $ANALYSIS_TYPE stepwise selection using consolidated dispatcher..."
    ;;

  params)
    echo "Configuring BOUT PARAMETER stepwise selection (target: model_params.model_target_variable)..."
    ;;

  *)
    echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
    echo "Usage: sbatch model_selection.sh [onset|params|category|multinomial|continuous]"
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"

set -e

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
    --anchor
