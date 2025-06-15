#!/bin/bash

# Usage: bash process_data_step_one_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

CUP_ROOT="Name"
CPUS_PER_TASK=24
TOTAL_MEMORY="128G"
TIME_RESTRICTION="06:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
CONDA_VERSION="anacondapy/2024.02"
USV_PLAYPEN_ENV="pni"

SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

SESSION_ID=$(basename "$SESSION_ROOT_DIRECTORY")

HPSS_GLOBAL_JOB_LIST="/mnt/cup/labs/falkner/$CUP_ROOT/HPSS/job_list.txt"
HPSS_GLOBAL_SHELL_SCRIPT="/mnt/cup/labs/falkner/$CUP_ROOT/HPSS/hpss_inference_global.sh"
DAS_GLOBAL_JOB_LIST="/mnt/cup/labs/falkner/$CUP_ROOT/DAS/job_list.txt"
DAS_GLOBAL_SHELL_SCRIPT="/mnt/cup/labs/falkner/$CUP_ROOT/DAS/das_inference_global.sh"

WORK_DIR="/mnt/cup/labs/falkner/$CUP_ROOT/USV_PLAYPEN/processing"
JOB_SCRIPT="$WORK_DIR/process_data_step_one_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=processing-one-$SESSION_ID" >> "$JOB_SCRIPT"
echo "#SBATCH --output=logs/processing-one-%j-$SESSION_ID.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=logs/processing-one-%j-$SESSION_ID.err" >> "$JOB_SCRIPT"
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
echo "conda activate $USV_PLAYPEN_ENV" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "concatenate-video-files --root_directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "rectify-video-fps --root_directory \"$SESSION_ROOT_DIRECTORY\" --conduct-concat" >> "$JOB_SCRIPT"
echo "multichannel-to-single-ch --root_directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "crop-wav-files --root_directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "av-sync-check --root_directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "echo $SESSION_ID > $HPSS_GLOBAL_JOB_LIST" >> "$JOB_SCRIPT"
echo "bash $HPSS_GLOBAL_SHELL_SCRIPT" >> "$JOB_SCRIPT"
echo "bp-filter-audio --root_directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "concatenate-audio-files --root_directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "echo $SESSION_ID > $DAS_GLOBAL_JOB_LIST" >> "$JOB_SCRIPT"
echo "bash $DAS_GLOBAL_SHELL_SCRIPT" >> "$JOB_SCRIPT"
echo "das-summarize --root_directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"