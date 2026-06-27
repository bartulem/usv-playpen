#!/bin/bash

# Usage: bash generate_naturalistic_usv_playback_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

# Experimenter id keying the experimenter-owned work/resource/model paths below
# and the `--exp-id` passed to the function (session/arena roots stay as
# entered). Match the `experimenter` key in this checkout's
# behavioral_experiments_settings.toml (experimenter-scoped *_settings.json paths are re-keyed to it automatically).
EXPERIMENTER_ID="Name"
CPUS_PER_TASK=4
TOTAL_MEMORY="16G"
TIME_RESTRICTION="02:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen"

NUM_USV_FILES=1
SAMPLING_RATE=250
SNIPPETS_DIR_PREFIX="female"
TOTAL_PLAYBACK_TIME=1080

# Inter-USV / inter-sequence interval distributions are no longer passed here:
# they are reconstructed at generation time from the per-sex Student-t mixture
# in the HDF5 interval archive (configured via naturalistic_iui_archive_h5 in
# analyses_settings.json).

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

WORK_DIR="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/USV_PLAYPEN/naturalistic_usv_playback"
JOB_SCRIPT="$WORK_DIR/generate_naturalistic_usv_playback_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=gen-natural-usv-${EXPERIMENTER_ID}" >> "$JOB_SCRIPT"
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
echo "source $USV_PLAYPEN_PATH/.venv/bin/activate" >> "$JOB_SCRIPT"
echo "(cd $USV_PLAYPEN_PATH && uv sync --extra gpu)" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "generate-naturalistic-usv-playback \\
    --exp-id \"$EXPERIMENTER_ID\" \\
    --num-naturalistic-usv-files $NUM_USV_FILES \\
    --naturalistic-wav-sampling-rate $SAMPLING_RATE \\
    --naturalistic-playback-snippets-dir-prefix \"$SNIPPETS_DIR_PREFIX\" \\
    --total-playback-time $TOTAL_PLAYBACK_TIME" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
