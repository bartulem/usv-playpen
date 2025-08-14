#!/bin/bash

# Usage: bash sleap_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

WORK_DIR="/mnt/cup/labs/falkner/Name/SLEAP/inference"
CPUS_PER_TASK=2
MEMORY_PER_CPU="24G"
TIME_RESTRICTION="05:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
CONDA_NAME="anacondapy"
CONDA_DATE="2024.02"
CONDA_NAME_UPPERCASE="${CONDA_NAME^^}"
CONDA_VERSION="$CONDA_NAME/$CONDA_DATE"
SLEAP_ENV="sleap1.3.3"

SLEAP_BATCH_SIZE=2
SLEAP_TRACKER="flow"
SLEAP_TRACKING_SIMILARITY="instance"
SLEAP_TRACKING_WINDOW=5
SLEAP_CONNECT_SINGLE_BREAKS=1
SLEAP_PEAK_THRESHOLD=0.25
SLEAP_IOU_THRESHOLD=0.3
SLEAP_TARGET_INSTANCE_COUNT=2
SLEAP_PRECULL_TO_TARGET=2

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

JOB_SCRIPT="$WORK_DIR/sleap_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=sleap-inference-topdown" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/infer-topdown_%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/infer-topdown_%j.err" >> "$JOB_SCRIPT"
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
echo "source /mnt/cup/PNI-facilities/Computing/sw/pkg/Rhel9/$CONDA_NAME_UPPERCASE/$CONDA_DATE/etc/profile.d/conda.sh" >> "$JOB_SCRIPT"
echo "conda activate $SLEAP_ENV" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "sleap-track \"\$video_path\" -m \"\$centroid_model\" -m \"\$centered_instance_model\" -o \"\$save_path\" --verbosity rich --tracking.tracker \$3 --tracking.target_instance_count \$8 --tracking.pre_cull_to_target \$9 --tracking.pre_cull_iou_threshold \${10} --tracking.track_window \$7 --peak_threshold \$5 --tracking.post_connect_single_breaks \$6 --tracking.similarity \$4 --batch_size \$2" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# ---------------- CREATE JOB ARRAY ---------------- #

ARRAY_ARGS_FILE="$WORK_DIR/job_list.txt"

NUM_ARRAY_JOBS="$(cat $ARRAY_ARGS_FILE | wc -l)"

echo "Jobs: $NUM_ARRAY_JOBS"

sbatch -a 1-"$NUM_ARRAY_JOBS" "$JOB_SCRIPT" "$ARRAY_ARGS_FILE" "$SLEAP_BATCH_SIZE" "$SLEAP_TRACKER" "$SLEAP_TRACKING_SIMILARITY" "$SLEAP_PEAK_THRESHOLD" "$SLEAP_CONNECT_SINGLE_BREAKS" "$SLEAP_TRACKING_WINDOW" "$SLEAP_TARGET_INSTANCE_COUNT" "$SLEAP_PRECULL_TO_TARGET" "$SLEAP_IOU_THRESHOLD"
