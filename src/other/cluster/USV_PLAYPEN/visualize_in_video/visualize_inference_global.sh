#!/bin/bash

# Usage: bash visualize_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

CUP_ROOT="Name"
CPUS_PER_TASK=8
MEMORY_PER_CPU="48G"
TIME_RESTRICTION="12:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
CONDA_VERSION="anacondapy/2024.02"
USV_PLAYPEN_ENV="pni"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

WORK_DIR="/mnt/cup/labs/falkner/$CUP_ROOT/USV_PLAYPEN/visualizations"
JOB_SCRIPT="$WORK_DIR/visualize_inference_settings.sh"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=visualize-video" >> "$JOB_SCRIPT"
echo "#SBATCH --output=logs/visualize-video_%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=logs/visualize-video_%j.err" >> "$JOB_SCRIPT"
echo "#SBATCH --gpus=1" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem-per-cpu=$MEMORY_PER_CPU" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "module load $CONDA_VERSION" >> "$JOB_SCRIPT"
echo "conda activate $USV_PLAYPEN_ENV" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "video-viz" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
