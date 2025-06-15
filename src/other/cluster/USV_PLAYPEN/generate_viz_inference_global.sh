#!/bin/bash

# Usage: bash generate_viz_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

CUP_ROOT="Name"
CPUS_PER_TASK=24
TOTAL_MEMORY="96G"
TIME_RESTRICTION="24:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
CONDA_VERSION="anacondapy/2024.02"
USV_PLAYPEN_ENV="pni"

EXP_ID="Name"
SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726"
ARENA_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_091231"
VIDEO_START_TIME=0.0
VIDEO_DURATION=1199.0

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

SESSION_ID=$(basename "$SESSION_ROOT_DIRECTORY")

WORK_DIR="/mnt/cup/labs/falkner/$CUP_ROOT/USV_PLAYPEN/visualizations"
JOB_SCRIPT="$WORK_DIR/generate_viz_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=visualize-video-$SESSION_ID" >> "$JOB_SCRIPT"
echo "#SBATCH --output=logs/visualize-video-%j-$SESSION_ID.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=logs/visualize-video-%j-$SESSION_ID.err" >> "$JOB_SCRIPT"
echo "#SBATCH --gpus=1" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem=$TOTAL_MEMORY" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "module load ffmpeg" >> "$JOB_SCRIPT"
echo "module load $CONDA_VERSION" >> "$JOB_SCRIPT"
echo "conda activate $USV_PLAYPEN_ENV" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "generate-viz --root_directory \"$SESSION_ROOT_DIRECTORY\" --arena_directory \"$ARENA_DIRECTORY\" --exp_id $EXP_ID --animate_bool --video_start_time $VIDEO_START_TIME --video_duration $VIDEO_DURATION" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
