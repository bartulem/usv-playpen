#!/bin/bash

# Usage: bash sleap_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

SLEAP_ROOT="Name"
MOUSE_COUNT=2
BATCH_SIZE=2
CPUS_PER_TASK=2
MEMORY_PER_CPU="96G"
TIME_RESTRICTION="05:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
CONDA_VERSION="anacondapy/2024.02"
SLEAP_ENV="sleap1.3.3"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

WORK_DIR="/mnt/cup/labs/falkner/$SLEAP_ROOT/SLEAP/inference"
JOB_SCRIPT="$WORK_DIR/sleap_inference_settings.sh"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=sleap-inference-topdown" >> "$JOB_SCRIPT"
echo "#SBATCH --output=logs/infer-topdown_%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=logs/infer-topdown_%j.err" >> "$JOB_SCRIPT"
echo "#SBATCH --gpus=1" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem-per-cpu=$MEMORY_PER_CPU" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "text_file=\"\$1\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "linenum=\$SLURM_ARRAY_TASK_ID" >> "$JOB_SCRIPT"
echo "linetxt=\$(sed -n \"\$linenum p\" \$text_file)" >> "$JOB_SCRIPT"
echo "linetxt=\$(echo \$linetxt | tr -d '\r\n')" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "echo \"\$linetxt\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "delim=' ' read -r -a linearray <<< \"\$linetxt\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "centroid_model=\"\${linearray[0]}\"" >> "$JOB_SCRIPT"
echo "centered_instance_model=\"\${linearray[1]}\"" >> "$JOB_SCRIPT"
echo "video_path=\"\${linearray[2]}\"" >> "$JOB_SCRIPT"
echo "save_path=\"\${linearray[3]}\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "module load $CONDA_VERSION" >> "$JOB_SCRIPT"
echo "conda activate $SLEAP_ENV" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "sleap-track \"\$video_path\" -m \"\$centroid_model\" -m \"\$centered_instance_model\" -o \"\$save_path\" --verbosity rich --tracking.tracker flow --peak_threshold 0.25 --tracking.post_connect_single_breaks 1 --tracking.similarity instance --tracking.clean_instance_count \$2 --batch_size \$3" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# ---------------- CREATE JOB ARRAY ---------------- #

ARRAY_ARGS_FILE="$WORK_DIR/job_list.txt"

NUM_ARRAY_JOBS="$(cat $ARRAY_ARGS_FILE | wc -l)"

echo "Jobs: $NUM_ARRAY_JOBS"

sbatch -a 1-"$NUM_ARRAY_JOBS" "$JOB_SCRIPT" "$ARRAY_ARGS_FILE" "$MOUSE_COUNT" "$BATCH_SIZE"
