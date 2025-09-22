#!/bin/bash

# Usage: bash process_data_step_one_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

WORK_DIR="/mnt/cup/labs/falkner/Name/USV_PLAYPEN/processing"
HPSS_WORK_DIR="/mnt/cup/labs/falkner/Name/HPSS"
DAS_WORK_DIR="/mnt/cup/labs/falkner/Name/DAS"

CPUS_PER_TASK=24
TOTAL_MEMORY="128G"
TIME_RESTRICTION="03:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen/"

SESSION_ROOT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

SESSION_ID=$(basename "$SESSION_ROOT_DIRECTORY")

HPSS_GLOBAL_JOB_LIST="$HPSS_WORK_DIR/job_list.txt"
HPSS_GLOBAL_SHELL_SCRIPT="$HPSS_WORK_DIR/hpss_inference_global.sh"
DAS_GLOBAL_JOB_LIST="$DAS_WORK_DIR/job_list.txt"
DAS_GLOBAL_SHELL_SCRIPT="$DAS_WORK_DIR/das_inference_global.sh"

JOB_SCRIPT="$WORK_DIR/process_data_step_one_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=processing-one-$SESSION_ID" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/processing-one-%j-$SESSION_ID.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/processing-one-%j-$SESSION_ID.err" >> "$JOB_SCRIPT"
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
echo "concatenate-video-files --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "rectify-video-fps --root-directory \"$SESSION_ROOT_DIRECTORY\" --conduct-concat" >> "$JOB_SCRIPT"
echo "multichannel-to-single-ch --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "crop-wav-files --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "echo $SESSION_ID > \"$HPSS_GLOBAL_JOB_LIST\"" >> "$JOB_SCRIPT"
echo "HPSS_SUB_OUT=\$(bash \"$HPSS_GLOBAL_SHELL_SCRIPT\")" >> "$JOB_SCRIPT"
echo "echo \"Raw submission output for HPSS: '\$HPSS_SUB_OUT'\"" >> "$JOB_SCRIPT"
echo "HPSS_JOB_IDS=\$(echo \"\$HPSS_SUB_OUT\" | grep 'Submitted batch job' | awk '{print \$NF}')" >> "$JOB_SCRIPT"
echo "echo \"Extracted HPSS Job IDs: '\$HPSS_JOB_IDS'\"" >> "$JOB_SCRIPT"
echo "if [ -z \"\$HPSS_JOB_IDS\" ]; then" >> "$JOB_SCRIPT"
echo "    echo 'Error: Failed to capture any HPSS Job IDs.'" >> "$JOB_SCRIPT"
echo "    exit 1" >> "$JOB_SCRIPT"
echo "fi" >> "$JOB_SCRIPT"
echo "echo \"Waiting for all HPSS jobs to complete...\"" >> "$JOB_SCRIPT"
echo "for JOB_ID in \$HPSS_JOB_IDS; do" >> "$JOB_SCRIPT"
echo "    echo \"  - Waiting for HPSS job \$JOB_ID...\"" >> "$JOB_SCRIPT"
echo "    while squeue -h -j \"\$JOB_ID\" &>/dev/null; do sleep 30; done" >> "$JOB_SCRIPT"
echo "    echo \"  - HPSS job \$JOB_ID completed.\"" >> "$JOB_SCRIPT"
echo "done" >> "$JOB_SCRIPT"
echo "echo \"All HPSS jobs are assumed to have completed successfully.\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "bp-filter-audio --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "concatenate-audio-files --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "echo $SESSION_ID > \"$DAS_GLOBAL_JOB_LIST\"" >> "$JOB_SCRIPT"
echo "DAS_SUB_OUT=\$(bash \"$DAS_GLOBAL_SHELL_SCRIPT\")" >> "$JOB_SCRIPT"
echo "echo \"Raw submission output for DAS: '\$DAS_SUB_OUT'\"" >> "$JOB_SCRIPT"
echo "DAS_JOB_IDS=\$(echo \"\$DAS_SUB_OUT\" | grep 'Submitted batch job' | awk '{print \$NF}')" >> "$JOB_SCRIPT"
echo "echo \"Extracted DAS Job IDs: '\$DAS_JOB_IDS'\"" >> "$JOB_SCRIPT"
echo "if [ -z \"\$DAS_JOB_IDS\" ]; then" >> "$JOB_SCRIPT"
echo "    echo 'Error: Failed to capture any DAS Job IDs.'" >> "$JOB_SCRIPT"
echo "    exit 1" >> "$JOB_SCRIPT"
echo "fi" >> "$JOB_SCRIPT"
echo "echo \"Waiting for all DAS jobs to complete...\"" >> "$JOB_SCRIPT"
echo "for JOB_ID in \$DAS_JOB_IDS; do" >> "$JOB_SCRIPT"
echo "    echo \"  - Waiting for DAS job \$JOB_ID...\"" >> "$JOB_SCRIPT"
echo "    while squeue -h -j \"\$JOB_ID\" &>/dev/null; do sleep 30; done" >> "$JOB_SCRIPT"
echo "    echo \"  - DAS job \$JOB_ID completed.\"" >> "$JOB_SCRIPT"
echo "done" >> "$JOB_SCRIPT"
echo "echo \"All DAS jobs are assumed to have completed successfully.\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "das-summarize --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "av-sync-check --root-directory \"$SESSION_ROOT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "echo 'All processing steps (step one) completed successfully.'" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
