#!/bin/bash
#SBATCH --job-name=neural_modeling_cohort
#SBATCH --output=logs/neural_modeling_cohort_%A_%a.out
#SBATCH --error=logs/neural_modeling_cohort_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-user=nsurname@domain.edu
#SBATCH --mail-type=FAIL

# One unit of the neural modeling cohort per array task, end to end (encoding folds, combine, vocal steps).
#
# Usage:
#   1. plan once (writes <output_dir>/cohort_manifest.json and prints the array range):
#        python -m usv_playpen.neural_modeling.main_neural_cohort_dispatcher plan --output-directory <output_dir>
#   2. submit, capping concurrent tasks with %N:
#        sbatch --array=0-<N-1>%50 neural_modeling_cohort.sh <output_dir>
#      Arrays larger than the cluster's MaxArraySize are submitted in chunks with an offset:
#        INDEX_OFFSET=1000 sbatch --array=0-999%50 neural_modeling_cohort.sh <output_dir>
#
# A task that times out can simply be resubmitted with the same index: folds and sections already written
# under the same settings are kept.
#
# Monitor the run (counts done / running / failed / not started, failed stages, median stage minutes):
#   python -m usv_playpen.neural_modeling.main_neural_cohort_dispatcher status \
#       --manifest <output_dir>/cohort_manifest.json --output-directory <output_dir>
# Per task: logs/neural_modeling_cohort_<job>_<task>.out (timeline) and <output_dir>/logs/<unit_uid>_fold<i>.log.

OUTPUT_DIR=$1
if [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: sbatch --array=0-<N-1> neural_modeling_cohort.sh <output_dir>"
  exit 1
fi

EXPERIMENTER_ID="Name"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"
DATA_ROOT="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/Data"
INDEX_OFFSET=${INDEX_OFFSET:-0}
PARALLEL_FOLDS=${PARALLEL_FOLDS:-3}

mkdir -p logs

set -e

# Environment
source ${USV_PLAYPEN_PATH}.venv/bin/activate

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

echo "node=$(hostname)  task=${SLURM_ARRAY_TASK_ID}  offset=${INDEX_OFFSET}  cpus=${SLURM_CPUS_PER_TASK}"

python -m usv_playpen.neural_modeling.main_neural_cohort_dispatcher run \
    --manifest "$OUTPUT_DIR/cohort_manifest.json" \
    --index-offset "$INDEX_OFFSET" \
    --output-directory "$OUTPUT_DIR" \
    --data-root "$DATA_ROOT" \
    --n-workers "$SLURM_CPUS_PER_TASK" \
    --parallel-folds "$PARALLEL_FOLDS"
