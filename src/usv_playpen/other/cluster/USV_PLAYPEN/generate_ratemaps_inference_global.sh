#!/bin/bash

# Usage: bash generate_ratemaps_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

CUP_ROOT="Name"
CPUS_PER_TASK=4
TOTAL_MEMORY="48G"
TIME_RESTRICTION="05:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"

SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

SESSION_ID=$(basename "$SESSION_ROOT_DIRECTORY")

WORK_DIR="/mnt/cup/labs/falkner/$CUP_ROOT/USV_PLAYPEN/ratemaps"
JOB_SCRIPT="$WORK_DIR/generate_ratemaps_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=generate-ratemaps-$SESSION_ID" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/generate-ratemaps-%j-$SESSION_ID.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/generate-ratemaps-%j-$SESSION_ID.err" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem=$TOTAL_MEMORY" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "set -e" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "source $USV_PLAYPEN_PATH.venv/bin/activate" >> "$JOB_SCRIPT"
echo "(cd $USV_PLAYPEN_PATH && uv sync)" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "generate-rm --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "generate-rm-figs --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
