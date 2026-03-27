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

# For univariate onset modeling, use: --time=48:00:00 --mem-per-cpu=16G --cpus-per-task=1
# For univariate params modeling, use: --time=96:00:00 --mem=32G --cpus-per-task=1
# For univariate category (Binomial), use: --time=96:00:00 --mem=64G --cpus-per-task=1
# For univariate multinomial (JAX), use: --time=02:00:00 --mem=32G --cpus-per-task=1 --gpus:1
# For univariate continuous (JAX), recommended: --time=04:00:00 --mem=32G --gpus:1

ANALYSIS_TYPE=$1  # onset, params, category, multinomial, continuous, or binary_coarse

# Define your core variables
INPUT_DATA="/mnt/cup/labs/falkner/Bartul/modeling/data/modeling_UMAP_manifold_position_female_20260226_150803_hist4s.pkl"
SETTINGS_FILE="/mnt/cup/labs/falkner/Bartul/modeling/cluster/settings/modeling_settings.json"
OUTPUT_DIR="/mnt/cup/labs/falkner/Bartul/modeling/univariate_results"

mkdir -p logs

# Validate input type
case $ANALYSIS_TYPE in
  onset|params|category|multinomial|continuous|binary_coarse)
    echo "Configuring $ANALYSIS_TYPE analysis using consolidated dispatcher..."
    ;;
  *)
    echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
    echo "Usage: sbatch univariate_modeling.sh [onset|params|category|multinomial|continuous|binary_coarse]"
    exit 1
    ;;
esac

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Environment
module load anacondapy/2024.02
module load cudatoolkit/12.9
conda activate modeling_env

export PYTHONUNBUFFERED=1

# JAX specific optimization: prevent JAX from pre-allocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

set -e

# Run the consolidated dispatcher
python main_univariate_dispatcher.py \
    --analysis_type "$ANALYSIS_TYPE" \
    --feature_idx "$SLURM_ARRAY_TASK_ID" \
    --input_data "$INPUT_DATA" \
    --settings_file "$SETTINGS_FILE" \
    --output_dir "$OUTPUT_DIR"
