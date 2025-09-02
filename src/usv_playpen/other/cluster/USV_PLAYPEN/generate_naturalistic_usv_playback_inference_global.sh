#!/bin/bash

# Usage: bash generate_naturalistic_usv_playback_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

CUP_ROOT="Name"
CPUS_PER_TASK=4
TOTAL_MEMORY="16G"
TIME_RESTRICTION="02:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/mnt/cup/labs/falkner/NAME/spock/usv-playpen/"

EXP_ID="Name"
NUM_USV_FILES=1
SAMPLING_RATE=250
SNIPPETS_DIR_PREFIX="female"
TOTAL_PLAYBACK_TIME=1080

INTER_SEQ_INTERVAL_DIST='{"2.5": 0.125, "5": 0.5, "7.5": 0.25, "10": 0.125}'
USV_SEQ_LENGTH_DIST='{"5": 0.5, "10": 0.25, "20": 0.125, "40": 0.0625, "80": 0.0625}'
INTER_USV_INTERVAL_DIST='{"0.02": 0.02, "0.04": 0.33, "0.06": 0.45, "0.08": 0.1, "0.1": 0.025, "0.15": 0.045, "0.2": 0.025, "0.25": 0.005}'

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

WORK_DIR="/mnt/cup/labs/falkner/$CUP_ROOT/USV_PLAYPEN/naturalistic_usv_playback"
JOB_SCRIPT="$WORK_DIR/generate_naturalistic_usv_playback_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=gen-natural-usv-${EXP_ID}" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/gen-natural-usv-%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/gen-natural-usv-%j.err" >> "$JOB_SCRIPT"
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
echo "generate-naturalistic-usv-playback \\
    --exp-id \"$EXP_ID\" \\
    --num-naturalistic-usv-files $NUM_USV_FILES \\
    --naturalistic-wav-sampling-rate $SAMPLING_RATE \\
    --naturalistic-playback-snippets-dir-prefix \"$SNIPPETS_DIR_PREFIX\" \\
    --total-playback-time $TOTAL_PLAYBACK_TIME \\
    --inter-seq-interval-dist '$INTER_SEQ_INTERVAL_DIST' \\
    --usv-seq-length-dist '$USV_SEQ_LENGTH_DIST' \\
    --inter-usv-interval-dist '$INTER_USV_INTERVAL_DIST'" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"