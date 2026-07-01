#!/bin/bash

# Usage: bash process_data_step_four_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

# Experimenter id keying the experimenter-owned work/resource/model paths below
# (session/arena roots stay as entered). It is exported into the SLURM job and
# overrides the checkout's behavioral_experiments_settings.toml, so the
# experimenter-scoped *_settings.json paths are re-keyed to it automatically --
# no TOML edit needed.
EXPERIMENTER_ID="Name"
WORK_DIR="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/USV_PLAYPEN/processing"
CPUS_PER_TASK=8
TOTAL_MEMORY="64G"
TIME_RESTRICTION="03:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen"

SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

SESSION_ID=$(basename "$SESSION_ROOT_DIRECTORY")
JOB_SCRIPT="$WORK_DIR/process_data_step_four_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=usv-spectro-$SESSION_ID" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/usv-spectro-%j-$SESSION_ID.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/usv-spectro-%j-$SESSION_ID.err" >> "$JOB_SCRIPT"
echo "#SBATCH --gpus=1" >> "$JOB_SCRIPT"
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
echo "generate-usv-spectrograms --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "generate-usv-masks --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "generate-usv-acoustic-features --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "infer-qlvm-latents --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "echo 'All processing steps (step four) completed successfully.'" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
