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

# Resource Guidelines:
# For Binomial category selection: --mem=512G --cpus-per-task=8
# For Multinomial (JAX) selection: --mem=64G --gres=gpu:1
# For Continuous (JAX) selection:  --mem=256G --gres=gpu:1
# For Binary Coarse (JAX) selection: --mem=64G --gres=gpu:1

# NOTE: Replace 'nsurname' below with your cluster username before submitting.
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"

ANALYSIS_TYPE=$1  # Target: onset, params, category, multinomial, continuous, or binary_coarse

# Define core variables
UNIVARIATE_PATH="/mnt/cup/labs/falkner/Bartul/modeling/univariate_results/univariate_multinomial_results.pkl"
INPUT_DATA="/mnt/cup/labs/falkner/Bartul/modeling/data/glm_male_hist4s.pkl"
SETTINGS_FILE="/mnt/cup/labs/falkner/Bartul/modeling/cluster/settings/modeling_settings.json"
OUTPUT_DIR="/mnt/cup/labs/falkner/Bartul/modeling/model_selection_results"

mkdir -p logs

# Set analysis-specific defaults
TARGET_VAR="bout_durations" # Only used for 'params'
PVAL=0.01

case $ANALYSIS_TYPE in
  onset|category|multinomial|continuous|binary_coarse)
    echo "Configuring $ANALYSIS_TYPE stepwise selection using consolidated dispatcher..."
    ;;

  params)
    echo "Configuring BOUT PARAMETER stepwise selection..."
    # You can override the target variable here if needed via a 2nd argument
    TARGET_VAR=${2:-"bout_durations"}
    ;;

  *)
    echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
    echo "Usage: sbatch model_selection.sh [onset|params|category|multinomial|continuous|binary_coarse]"
    exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"

# Environment
source ${USV_PLAYPEN_PATH}.venv/bin/activate
(cd ${USV_PLAYPEN_PATH} && uv sync)

# JAX optimization
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

set -e

# Execute the consolidated dispatcher
python main_model_selection_dispatcher.py \
    --analysis_type "$ANALYSIS_TYPE" \
    --univariate_path "$UNIVARIATE_PATH" \
    --input_path "$INPUT_DATA" \
    --settings_path "$SETTINGS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --target_variable "$TARGET_VAR" \
    --pval "$PVAL" \
    --anchor
