#!/bin/bash
#SBATCH --job-name=univariate_modeling_job
#SBATCH --output=logs/univariate_modeling_%A_%a.out
#SBATCH --error=logs/univariate_modeling_%A_%a.err
#SBATCH --array=0-26
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=nsurname@domain.edu
#SBATCH --mail-type=FAIL

# Usage: sbatch univariate_modeling_behavior.sh (onset|params|category|behavioral_response|multinomial|continuous)
#
# NOTE on --array: the array index is the FEATURE index into the sorted feature
# keys of INPUT_DATA, so the upper bound must equal (number of features - 1) for
# whichever pickle you point at. For 'behavioral_response' the pickle holds both
# mice's egocentric features plus the dyadic set and the vocal trace.
#
# The two directions fail very differently, so check this before submitting:
#   TOO LARGE  is harmless -- surplus tasks print 'FATAL: Index out of bounds'
#              and exit without writing.
#   TOO SMALL  is NOT harmless -- the unswept features simply never appear in the
#              consolidated artifact, and the downstream screen reports them as
#              'skipped' and carries on with a truncated candidate set. The run
#              completes and looks normal. Confirm the count first, e.g.:
#   python -c "import pickle,sys; d=pickle.load(open(sys.argv[1],'rb')); \
#              print(len([k for k in d if not k.startswith('_')]))" $INPUT_DATA
ANALYSIS_TYPE=$1

# Operator-edited roots; INPUT_DATA is chosen per analysis in the case block
# below so a behavioural-response sweep cannot run against, say, the manifold
# extraction pickle. Export INPUT_DATA or OUTPUT_DIR to override.
EXPERIMENTER_ID="Name"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"
MODELING_ROOT="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/modeling"

OUTPUT_DIR="${OUTPUT_DIR:-$MODELING_ROOT/univariate_results}"

mkdir -p logs

# Validate input type
case $ANALYSIS_TYPE in
  behavioral_response)
    echo "Configuring BEHAVIOURAL-RESPONSE univariate sweep..."
    # Timestamped extraction artifact; resolved by prefix, newest wins.
    if [ -z "${INPUT_DATA:-}" ]; then
      INPUT_DATA=$(ls -1t "$MODELING_ROOT"/data/modeling_behavioral_response_*.pkl \
                   2>/dev/null | head -1 || true)
    fi
    ;;

  onset|params|category|multinomial|continuous)
    echo "Configuring $ANALYSIS_TYPE analysis using consolidated dispatcher..."
    INPUT_DATA="${INPUT_DATA:-$MODELING_ROOT/data/modeling_UMAP_manifold_position_female_20260226_150803_hist4s.pkl}"
    ;;
  *)
    echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
    echo "Usage: sbatch univariate_modeling.sh [onset|params|category|behavioral_response|multinomial|continuous]"
    exit 1
    ;;
esac

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

set -e

# Environment
source ${USV_PLAYPEN_PATH}.venv/bin/activate

export PYTHONUNBUFFERED=1

# JAX optimization
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=2

if [ ! -f "$INPUT_DATA" ]; then
  echo "Error: extraction pickle not found: $INPUT_DATA"
  exit 1
fi

echo "analysis = $ANALYSIS_TYPE"
echo "input    = $INPUT_DATA"
echo "node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

# Run the consolidated dispatcher
python -m usv_playpen.modeling.main_univariate_dispatcher \
    --analysis_type "$ANALYSIS_TYPE" \
    --feature_idx "$SLURM_ARRAY_TASK_ID" \
    --input_data "$INPUT_DATA" \
    --output_dir "$OUTPUT_DIR"
