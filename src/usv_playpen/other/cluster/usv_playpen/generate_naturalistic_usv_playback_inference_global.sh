#!/bin/bash

# Usage: bash generate_naturalistic_usv_playback_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

# Experimenter id keying the experimenter-owned work/resource/model paths below
# (session/arena roots stay as entered). It is exported into the SLURM job and
# overrides the checkout's behavioral_experiments_settings.toml, so the
# experimenter-scoped *_settings.json paths are re-keyed to it automatically --
# no TOML edit needed.
EXPERIMENTER_ID="Name"
CPUS_PER_TASK=4
TOTAL_MEMORY="16G"
TIME_RESTRICTION="02:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen"

NUM_USV_FILES=1
PLAYBACK_SEX="female"
TOTAL_PLAYBACK_TIME=1080

# Real bouts are replayed from a per-sex naturalistic USV repository H5, configured
# via male_repository_h5 / female_repository_h5 in analyses_settings.json; only the
# sex (and the per-sex repository path) is selected -- no interval model here.

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
echo "export EXPERIMENTER_ID=\"$EXPERIMENTER_ID\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "generate-naturalistic-usv-playback \\
    --exp-id \"$EXPERIMENTER_ID\" \\
    --num-naturalistic-usv-files $NUM_USV_FILES \\
    --playback-sex \"$PLAYBACK_SEX\" \\
    --total-playback-time $TOTAL_PLAYBACK_TIME" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
