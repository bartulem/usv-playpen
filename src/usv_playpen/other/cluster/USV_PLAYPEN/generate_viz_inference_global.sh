#!/bin/bash

# Usage: bash generate_viz_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

WORK_DIR="/mnt/cup/labs/falkner/Name/USV_PLAYPEN/visualizations"
CPUS_PER_TASK=6
TOTAL_MEMORY="8G"
TIME_RESTRICTION="24:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/mnt/cup/labs/falkner/NAME/spock/usv-playpen/"

EXP_ID="Name"
SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726"
ARENA_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_091231"
VIDEO_START_TIME=0.0
VIDEO_DURATION=1199.0
ANIMATION_CODEC="h264_nvenc"
ANIMATION_CODEC_PRESET="p5"
ANIMATION_CODEC_TUNE="hq"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

SESSION_ID=$(basename "$SESSION_ROOT_DIRECTORY")

JOB_SCRIPT="$WORK_DIR/generate_viz_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=visualize-video-$SESSION_ID" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/visualize-video-%j-$SESSION_ID.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/visualize-video-%j-$SESSION_ID.err" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem=$TOTAL_MEMORY" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "source $USV_PLAYPEN_PATH.venv/bin/activate" >> "$JOB_SCRIPT"
echo "(cd $USV_PLAYPEN_PATH && uv sync)" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "generate-viz \\
    --root-directory \"$SESSION_ROOT_DIRECTORY\" \\
    --arena-directory \"$ARENA_DIRECTORY\" \\
    --exp-id \"$EXP_ID\" \\
    --animate \\
    --video-start-time $VIDEO_START_TIME \\
    --video-duration $VIDEO_DURATION \\
    --animation-codec \"$ANIMATION_CODEC\" \\
    --animation-codec-preset \"$ANIMATION_CODEC_PRESET\" \\
    --animation-codec-tune \"$ANIMATION_CODEC_TUNE\"" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
