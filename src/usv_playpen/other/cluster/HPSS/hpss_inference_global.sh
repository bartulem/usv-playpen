#!/bin/bash

# Usage: bash hpss_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

CUP_ROOT_DIR="falkner/Name/Data"
WORK_DIR="/mnt/cup/labs/falkner/Name/HPSS"
AUDIO_CH_NUM=24
CPUS_PER_TASK=4
MEMORY_PER_CPU="40G"
TIME_RESTRICTION="02:02:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="NONE"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

JOB_SCRIPT="$WORK_DIR/hpss_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=hpss" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/hpss_%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/hpss_%j.err" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem-per-cpu=$MEMORY_PER_CPU" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "echo \"Working on harmonic=percussive source separation in \$1, channel \$3.\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "source $USV_PLAYPEN_PATH/.venv/bin/activate" >> "$JOB_SCRIPT"
echo "(cd $USV_PLAYPEN_PATH && uv sync --extra gpu)" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "mkdir -p \"/mnt/cup/labs/\$2/\$1/audio/hpss\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "export NUMBA_CACHE_DIR=/tmp/numba_cache_\$SLURM_JOB_ID" >> "$JOB_SCRIPT"
echo "python $WORK_DIR/hpss.py \"\$1\" \"\$2\" \"\$3\"" >> "$JOB_SCRIPT"
echo "rm -rf /tmp/numba_cache_\$SLURM_JOB_ID" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# ---------------- CREATE JOB ARRAY ---------------- #

ARRAY_ARGS_FILE="$WORK_DIR/job_list.txt"

DIR_NUM=$(grep -c "^" "$ARRAY_ARGS_FILE")
NUM_JOBS=$((DIR_NUM*"$AUDIO_CH_NUM"))

echo "Jobs: $NUM_JOBS"

for i in $(seq 1 "$DIR_NUM");
do
    for j in $(seq 0 $(("$AUDIO_CH_NUM"-1)));
    do
        session_id=$(sed -n "$i p" $ARRAY_ARGS_FILE)
        session_id=$(echo "$session_id" | tr -d '\r\n')

        sbatch "$JOB_SCRIPT" "$session_id" "$CUP_ROOT_DIR" "$j"
    done
done
