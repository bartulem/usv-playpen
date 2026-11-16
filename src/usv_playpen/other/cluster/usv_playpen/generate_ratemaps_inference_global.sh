#!/bin/bash

# Usage: bash generate_ratemaps_inference_global.sh
#
# Submits one SLURM array task per session listed in SESSION_LIST_FILE (one
# session root directory per line; blank lines and lines starting with '#' are
# skipped). Each task runs generate-rm and then generate-rm-figs on its session.

# Select hyper-parameters

# Experimenter id keying the experimenter-owned work/resource/model paths below
# (session/arena roots stay as entered). It is exported into the SLURM job and
# overrides the checkout's behavioral_experiments_settings.toml, so the
# experimenter-scoped *_settings.json paths are re-keyed to it automatically --
# no TOML edit needed.
EXPERIMENTER_ID="Name"
CPUS_PER_TASK=2
TOTAL_MEMORY="16G"
TIME_RESTRICTION="18:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen"

# Most array tasks allowed to run at once (empty for no limit).
MAX_CONCURRENT_TASKS=""

WORK_DIR="/mnt/cup/labs/falkner/$EXPERIMENTER_ID/USV_PLAYPEN/ratemaps"
SESSION_LIST_FILE="$WORK_DIR/ratemaps_sessions_list.txt"

# Create the job script

JOB_SCRIPT="$WORK_DIR/generate_ratemaps_inference_settings.sh"
ARRAY_LIST_FILE="$WORK_DIR/ratemaps_array_sessions.txt"

if [ ! -f "$SESSION_LIST_FILE" ]; then
    echo "Session list not found: $SESSION_LIST_FILE"
    exit 1
fi

mkdir -p "$WORK_DIR/logs"

# A clean copy of the list (no blanks, comments or carriage returns), so the
# array index maps onto a session line one to one.
grep -v '^[[:space:]]*\(#\|$\)' "$SESSION_LIST_FILE" | tr -d '\r' > "$ARRAY_LIST_FILE"
NUM_SESSIONS=$(grep -c . "$ARRAY_LIST_FILE")

if [ "$NUM_SESSIONS" -eq 0 ]; then
    echo "No sessions in $SESSION_LIST_FILE"
    exit 1
fi

while IFS= read -r session_root; do
    if [ ! -d "$session_root" ]; then
        echo "Session directory not found: $session_root"
        exit 1
    fi
done < "$ARRAY_LIST_FILE"

cat << EOF > "$JOB_SCRIPT"
#!/bin/bash
#SBATCH --job-name=generate-ratemaps
#SBATCH --output=$WORK_DIR/logs/generate-ratemaps-%A_%a.out
#SBATCH --error=$WORK_DIR/logs/generate-ratemaps-%A_%a.err
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$TOTAL_MEMORY
#SBATCH --time=$TIME_RESTRICTION
#SBATCH --mail-type=$EMAIL_TYPE
#SBATCH --mail-user=$EMAIL_ADDRESS

set -e

SESSION_ROOT_DIRECTORY=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "$ARRAY_LIST_FILE")
echo "Array task \$SLURM_ARRAY_TASK_ID: \$SESSION_ROOT_DIRECTORY"

source $USV_PLAYPEN_PATH/.venv/bin/activate
export EXPERIMENTER_ID="$EXPERIMENTER_ID"

generate-rm --root-directory "\$SESSION_ROOT_DIRECTORY"
generate-rm-figs --root-directory "\$SESSION_ROOT_DIRECTORY"
EOF

# Run the job array

ARRAY_RANGE="1-$NUM_SESSIONS"
if [ -n "$MAX_CONCURRENT_TASKS" ]; then
    ARRAY_RANGE="$ARRAY_RANGE%$MAX_CONCURRENT_TASKS"
fi

echo "Submitting $NUM_SESSIONS ratemap task(s) from $SESSION_LIST_FILE"
sbatch --array="$ARRAY_RANGE" "$JOB_SCRIPT"
