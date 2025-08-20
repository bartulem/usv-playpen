#!/bin/bash

# Usage: bash process_data_step_two_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

WORK_DIR="/mnt/cup/labs/falkner/Name/USV_PLAYPEN/processing"
CPUS_PER_TASK=12
TOTAL_MEMORY="48G"
TIME_RESTRICTION="02:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
CONDA_NAME="anacondapy"
CONDA_DATE="2024.02"
CONDA_NAME_UPPERCASE="${CONDA_NAME^^}"
CONDA_VERSION="$CONDA_NAME/$CONDA_DATE"
USV_PLAYPEN_ENV="pni"
SLEAP_CONDA_ENV="sleap1.3.3"

SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726"
ARENA_SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_092213"
EXP_CODE="BCL2MGFGe"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

SESSION_ID=$(basename "$SESSION_ROOT_DIRECTORY")

JOB_SCRIPT="$WORK_DIR/process_data_step_two_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=processing-two-$SESSION_ID" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/processing-two-%j-$SESSION_ID.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/processing-two-%j-$SESSION_ID.err" >> "$JOB_SCRIPT"
echo "#SBATCH --gpus=1" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem=$TOTAL_MEMORY" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "set -e" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "module load ffmpeg" >> "$JOB_SCRIPT"
echo "module load $CONDA_VERSION" >> "$JOB_SCRIPT"
echo "source /mnt/cup/PNI-facilities/Computing/sw/pkg/Rhel9/$CONDA_NAME_UPPERCASE/$CONDA_DATE/etc/profile.d/conda.sh" >> "$JOB_SCRIPT"
echo "conda activate $USV_PLAYPEN_ENV" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "sleap-to-h5 --root-directory \"$SESSION_ROOT_DIRECTORY\" --env-name $SLEAP_CONDA_ENV" >> "$JOB_SCRIPT"
echo "anipose-triangulate --root-directory \"$SESSION_ROOT_DIRECTORY\" --cal-directory \"$ARENA_SESSION_ROOT_DIRECTORY\" --display-progress --no-arena-points" >> "$JOB_SCRIPT"
echo "anipose-trm --root-directory \"$SESSION_ROOT_DIRECTORY\" --exp-code $EXP_CODE --arena-directory \"$ARENA_SESSION_ROOT_DIRECTORY\" --delete-original" >> "$JOB_SCRIPT"
echo "echo 'All processing steps (step two) completed successfully.'" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"