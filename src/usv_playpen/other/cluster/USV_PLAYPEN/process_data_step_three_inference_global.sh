#!/bin/bash

# Usage: bash process_data_step_three_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

WORK_DIR="/mnt/cup/labs/falkner/Name/USV_PLAYPEN/processing"
CPUS_PER_TASK=6
TOTAL_MEMORY="24G"
TIME_RESTRICTION="04:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH = "/usr/people/nsurname/usv-playpen/"

VCL_CONDA_ENV="vcl-ssl"
SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726"
ARENA_SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_092213"
VCL_VERSION="vcl-ssl" # 'vcl' or 'vcl-ssl'
VCL_MODEL_DIR="/mnt/cup/labs/falkner/Bartul/sound_localization/conformer_smol_4.00"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

SESSION_ID=$(basename "$SESSION_ROOT_DIRECTORY")
JOB_SCRIPT="$WORK_DIR/process_data_step_three_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=vcl-assign-$SESSION_ID" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/vcl-assign-%j-$SESSION_ID.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/vcl-assign-%j-$SESSION_ID.err" >> "$JOB_SCRIPT"
echo "#SBATCH --gpus=1" >> "$JOB_SCRIPT"
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
echo "prepare-vcl-assign --root-directory \"$SESSION_ROOT_DIRECTORY\" --arena-directory \"$ARENA_SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "vcl-assign \\
    --root-directory \"$SESSION_ROOT_DIRECTORY\" \\
    --vcl-version \"$VCL_VERSION\" \\
    --env-name \"$VCL_CONDA_ENV\" \\
    --model-dir \"$VCL_MODEL_DIR\"" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"