#!/bin/bash
#SBATCH --job-name=model_selection_job
#SBATCH --output=logs/model_selection_%A.out
#SBATCH --error=logs/model_selection_%A.err
#SBATCH --time=96:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mail-user=nsurname@domain.edu
#SBATCH --mail-type=FAIL

# Usage: sbatch model_selection_behavior.sh (onset|params|category|multinomial|continuous)
#
# ONE job runs the whole selection: the candidate x fold fits of every step AND
# the shifted refits of the per-step test, through a forked process pool inside
# this job. The pool's worker count is min(selection_pool_workers, memory /
# (selection_fit_peak_gb_base + selection_fit_peak_gb_per_term * n_terms)); with
# the shipped 0 / 0 the CPUs and the memory are read from this job's
# SLURM_CPUS_PER_TASK and SLURM_MEM_PER_NODE, so --cpus-per-task and --mem above
# are what sets them (96 GB: 6 workers at 3 tensor terms, 2 at 9, at the measured
# pygam peaks of 13 GB and 36 GB). A job that ends mid-step is resumed by
# resubmitting it: finished steps and checkpointed null draws are reused.
ANALYSIS_TYPE=$1

# Define core variables
EXPERIMENTER_ID="Name"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"
UNIVARIATE_PATH="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling/cluster/univariate_results_multi_file/univariate_results.pkl"
INPUT_DATA="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling/modeling_manifold_qlvm_category_<cohort>_<YYYYMMDD_HHMMSS>.pkl"  # the analysis's extraction pickle; its *_traces.h5 sidecar must sit next to it
OUTPUT_DIR="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling/cluster/model_selection_multi_file"

mkdir -p logs

# The level of the per-step shift test and of the univariate screen, the test's
# lag floor and checkpoint interval, the pool knobs, and the bout-parameter
# regression target are read by the dispatcher and the selectors from
# modeling_settings.json (model_params.selection_p_val,
# model_params.selection_step_null_*, model_params.selection_pool_*,
# model_params.selection_fit_peak_gb_*, model_params.model_target_variable);
# set them there, not here.
#
# The search starts from the empty model and tests the cross-validated winner of
# every step, the first included. Passing --anchor to the dispatcher would force
# the top-ranked univariate feature to be the first candidate instead; it is
# still tested, so the template does not pass it.

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
echo "cpus_per_task=${SLURM_CPUS_PER_TASK:-<unset>}  mem_per_node_mb=${SLURM_MEM_PER_NODE:-<unset>}"
nvidia-smi -L || true

# Execute the consolidated dispatcher
python -m usv_playpen.modeling.main_model_selection_dispatcher \
    --analysis_type "$ANALYSIS_TYPE" \
    --univariate_path "$UNIVARIATE_PATH" \
    --input_path "$INPUT_DATA" \
    --output_dir "$OUTPUT_DIR"
