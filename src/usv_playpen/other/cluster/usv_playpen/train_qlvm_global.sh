#!/bin/bash

# Usage: bash train_qlvm_global.sh

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
TIME_RESTRICTION="06:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen"

# Comma-separated list of session root directories the cohort is built from.
SESSION_ROOT_DIRECTORIES="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726,/mnt/cup/labs/falkner/Bartul/Data/20230126_142000"
# Where build-qlvm-training-set writes the .npz training set (train-qlvm reads it back).
DATASET_DIRECTORY="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/spectrograms/qlvm/training_set"
# Where train-qlvm writes the checkpoint + decoder weights.
MODEL_OUTPUT_DIRECTORY="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/spectrograms/qlvm"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

JOB_SCRIPT="$WORK_DIR/train_qlvm_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=train-qlvm" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/train-qlvm-%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/train-qlvm-%j.err" >> "$JOB_SCRIPT"
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
echo "build-qlvm-training-set --root-directories \"$SESSION_ROOT_DIRECTORIES\" --output-directory \"$DATASET_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "train-qlvm --dataset-directory \"$DATASET_DIRECTORY\" --output-directory \"$MODEL_OUTPUT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "echo 'QLVM training (build set + train) completed successfully.'" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
