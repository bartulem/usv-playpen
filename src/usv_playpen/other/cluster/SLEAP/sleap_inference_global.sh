#!/bin/bash

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

SLEAP_ROOT="Name"
CPUS_PER_TASK=2
MEMORY_PER_CPU="24G"
TIME_RESTRICTION="05:00:00"
EMAIL_ADDRESS="nsurname@princeton.edu"
EMAIL_TYPE="ALL"

# SLEAP parameters
SLEAP_BATCH_SIZE=2
SLEAP_MAX_INSTANCES=2

SLEAP_FILTER_MIN_INSTANCE_SCORE=0.5
SLEAP_FILTER_MIN_MEAN_NODE_SCORE=0.62
SLEAP_FILTER_MIN_VISIBLE_NODES=7
SLEAP_FILTER_OVERLAPPING_METHOD="oks"
SLEAP_FILTER_OVERLAPPING_THRESHOLD=0.5
SLEAP_TRACKING_WINDOW_SIZE=150

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

WORK_DIR="/mnt/cup/labs/falkner/$SLEAP_ROOT/SLEAP/inference"
JOB_SCRIPT="$WORK_DIR/sleap_inference_settings.sh"
ARRAY_ARGS_FILE="$WORK_DIR/job_list.txt"

mkdir -p "$WORK_DIR/logs"

cat << EOF > "$JOB_SCRIPT"
#!/bin/bash
#SBATCH --job-name=sleap-inference
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem-per-cpu=$MEMORY_PER_CPU
#SBATCH --time=$TIME_RESTRICTION
#SBATCH --mail-type=$EMAIL_TYPE
#SBATCH --mail-user=$EMAIL_ADDRESS

# Parse the input file for this specific array task
text_file="\$1"
linenum=\$SLURM_ARRAY_TASK_ID
linetxt=\$(sed -n "\${linenum}p" "\$text_file" | tr -d '\r\n')

delim=' ' read -r -a linearray <<< "\$linetxt"
centroid_model="\${linearray[0]}"
centered_instance_model="\${linearray[1]}"
video_path="\${linearray[2]}"
save_path="\${linearray[3]}"

module load cudatoolkit/11.8.0 cudnn/11.x/8.9.7.29

# Make the persistently installed 'sleap-nn' tool resolvable.
# Requires a one-time 'uv tool install sleap-nn' (see repo docs).
export PATH="\$HOME/.local/bin:\$PATH"

echo "Running inference on: \$video_path"

sleap-nn track \\
    -i "\$video_path" \\
    -m "\$centroid_model" \\
    -m "\$centered_instance_model" \\
    -o "\$save_path" \\
    --batch_size $SLEAP_BATCH_SIZE \\
    --max_instances $SLEAP_MAX_INSTANCES \\
    --filter_min_instance_score $SLEAP_FILTER_MIN_INSTANCE_SCORE \\
    --filter_min_mean_node_score $SLEAP_FILTER_MIN_MEAN_NODE_SCORE \\
    --filter_min_visible_nodes $SLEAP_FILTER_MIN_VISIBLE_NODES \\
    --filter_overlapping \\
    --filter_overlapping_method $SLEAP_FILTER_OVERLAPPING_METHOD \\
    --filter_overlapping_threshold $SLEAP_FILTER_OVERLAPPING_THRESHOLD \\
    --tracking \\
    --tracking_window_size $SLEAP_TRACKING_WINDOW_SIZE
EOF

# -------------------------------------------------- #
# ---------------- RUN JOB ARRAY ------------------- #

NUM_ARRAY_JOBS=$(grep -c "^" "$ARRAY_ARGS_FILE")

echo "Submitting $NUM_ARRAY_JOBS jobs."

sbatch -a 1-"$NUM_ARRAY_JOBS" "$JOB_SCRIPT" "$ARRAY_ARGS_FILE"
