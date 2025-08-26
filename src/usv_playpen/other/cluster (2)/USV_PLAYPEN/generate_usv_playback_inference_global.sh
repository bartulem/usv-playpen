#!/bin/bash

# Usage: bash generate_usv_playback_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

CUP_ROOT="Name"
CPUS_PER_TASK=4
TOTAL_MEMORY="24G"
TIME_RESTRICTION="04:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
CONDA_VERSION="anacondapy/2024.02"
USV_PLAYPEN_ENV="pni"

EXP_ID="Name"
NUM_USV_FILES=1

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

WORK_DIR="/mnt/cup/labs/falkner/$CUP_ROOT/USV_PLAYPEN/usv_playback"
JOB_SCRIPT="$WORK_DIR/generate_usv_playback_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=visualize-video" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/visualize-video-%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/visualize-video-%j.err" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem=$TOTAL_MEMORY" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "module load $CONDA_VERSION" >> "$JOB_SCRIPT"
echo "source /mnt/cup/PNI-facilities/Computing/sw/pkg/Rhel9/ANACONDAPY/2024.02/etc/profile.d/conda.sh" >> "$JOB_SCRIPT"
echo "conda activate $USV_PLAYPEN_ENV" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "generate-usv-playback --exp_id $EXP_ID --num_usv_files $NUM_USV_FILES" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
