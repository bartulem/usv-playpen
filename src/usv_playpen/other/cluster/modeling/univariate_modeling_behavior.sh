#!/bin/bash
#SBATCH --job-name=univariate_modeling_job
#SBATCH --output=logs/univariate_modeling_%A_%a.out
#SBATCH --error=logs/univariate_modeling_%A_%a.err
#SBATCH --array=0-26
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mail-user=nsurname@domain.edu
#SBATCH --mail-type=FAIL

# Usage: sbatch univariate_modeling_behavior.sh (onset|params|category|multinomial|continuous)
ANALYSIS_TYPE=$1

# Define your core variables
# Experimenter id for the paths below (edit to yours). These scripts pass explicit
# paths, so EXPERIMENTER_ID is not exported and no TOML re-keying is involved.
EXPERIMENTER_ID="Name"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"
INPUT_DATA="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling/data/modeling_UMAP_manifold_position_female_20260226_150803_hist4s.pkl"
OUTPUT_DIR="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling/univariate_results"

mkdir -p logs

# Validate input type
case $ANALYSIS_TYPE in
  onset|params|category|multinomial|continuous)
    echo "Configuring $ANALYSIS_TYPE analysis using consolidated dispatcher..."
    ;;
  *)
    echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
    echo "Usage: sbatch univariate_modeling.sh [onset|params|category|multinomial|continuous]"
    exit 1
    ;;
esac

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

set -e

# Environment
source ${USV_PLAYPEN_PATH}.venv/bin/activate

export PYTHONUNBUFFERED=1

# JAX specific optimization: prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run the consolidated dispatcher
python -m usv_playpen.modeling.main_univariate_dispatcher \
    --analysis_type "$ANALYSIS_TYPE" \
    --feature_idx "$SLURM_ARRAY_TASK_ID" \
    --input_data "$INPUT_DATA" \
    --output_dir "$OUTPUT_DIR"
