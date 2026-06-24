#!/bin/bash

# Usage: bash generate_usv_playback_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

# Experimenter id keying the experimenter-owned work/resource/model paths below
# and the `--exp-id` passed to the function (session/arena roots stay as
# entered). Match the `experimenter` key in this checkout's
# behavioral_experiments_settings.toml (read to fill {experimenter}).
EXPERIMENTER_ID="Name"
CPUS_PER_TASK=4
TOTAL_MEMORY="24G"
TIME_RESTRICTION="02:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen"

NUM_USV_FILES=1

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

WORK_DIR="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/USV_PLAYPEN/usv_playback"
JOB_SCRIPT="$WORK_DIR/generate_usv_playback_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=usv-playback" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/usv-playback-%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/usv-playback-%j.err" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem=$TOTAL_MEMORY" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "source $USV_PLAYPEN_PATH/.venv/bin/activate" >> "$JOB_SCRIPT"
echo "(cd $USV_PLAYPEN_PATH && uv sync --extra gpu)" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "generate-usv-playback --exp-id \"$EXPERIMENTER_ID\" --num-usv-files $NUM_USV_FILES" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
