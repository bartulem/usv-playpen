#!/bin/bash
#SBATCH --job-name=model_selection_job
#SBATCH --output=logs/model_selection_%A.out
#SBATCH --error=logs/model_selection_%A.err
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mail-user=nsurname@domain.edu
#SBATCH --mail-type=FAIL

# Usage: sbatch model_selection_behavior.sh (onset|params|category|multinomial|continuous)
ANALYSIS_TYPE=$1

# Define core variables
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"
UNIVARIATE_PATH="/mnt/cup/labs/falkner/Name/modeling/univariate_results/univariate_multinomial_results.pkl"
INPUT_DATA="/mnt/cup/labs/falkner/Name/modeling/data/modeling_male_hist4s.pkl"
SETTINGS_FILE="/mnt/cup/labs/falkner/Name/modeling/cluster/settings/modeling_settings.json"
OUTPUT_DIR="/mnt/cup/labs/falkner/Name/modeling/model_selection_results"

mkdir -p logs

# Set analysis-specific defaults
TARGET_VAR="bout_durations" # Only used for 'params'
PVAL=0.01

case $ANALYSIS_TYPE in
  onset|category|multinomial|continuous)
    echo "Configuring $ANALYSIS_TYPE stepwise selection using consolidated dispatcher..."
    ;;

  params)
    echo "Configuring BOUT PARAMETER stepwise selection..."
    # You can override the target variable here if needed via a 2nd argument
    TARGET_VAR=${2:-"bout_durations"}
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
(cd ${USV_PLAYPEN_PATH} && uv sync --extra gpu)

# JAX optimization
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Execute the consolidated dispatcher
python -m usv_playpen.modeling.main_model_selection_dispatcher \
    --analysis_type "$ANALYSIS_TYPE" \
    --univariate_path "$UNIVARIATE_PATH" \
    --input_path "$INPUT_DATA" \
    --settings_path "$SETTINGS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --target_variable "$TARGET_VAR" \
    --pval "$PVAL" \
    --anchor
