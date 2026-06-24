#!/bin/bash

# Usage: bash train_masks_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

WORK_DIR="/mnt/cup/labs/falkner/Name/USV_PLAYPEN/processing"
CPUS_PER_TASK=8
TOTAL_MEMORY="64G"
TIME_RESTRICTION="06:00:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="ALL"
USV_PLAYPEN_PATH="/usr/people/nsurname/usv-playpen"

# Comma-separated list of session root directories the cohort is built from.
SESSION_ROOT_DIRECTORIES="/mnt/cup/labs/falkner/Bartul/Data/20230124_094726,/mnt/cup/labs/falkner/Bartul/Data/20230126_142000"
# Where export-yolo-dataset writes the Ultralytics dataset (train-masks reads it back).
DATASET_DIRECTORY="/mnt/cup/labs/falkner/Bartul/spectrograms/sam/yolo_dataset"
# Where train-masks writes the Ultralytics run + copied best.pt.
MODEL_OUTPUT_DIRECTORY="/mnt/cup/labs/falkner/Bartul/spectrograms/sam"

# Box-label source for the YOLO training set:
#   "cc"     -- AUTO: connected-component pseudo-labels, zero manual annotation (default).
#   "manual" -- LABELED: hand-verified YOLO-format {spec_id}.txt files taken from
#               MANUAL_LABELS_DIRECTORY (spectrograms with no file get an empty label).
#   "merge"  -- cc pseudo-labels, overridden by a manual {spec_id}.txt where one exists.
LABEL_SOURCE="cc"
# Only used when LABEL_SOURCE is "manual" or "merge" (ignored for the "cc" auto path).
MANUAL_LABELS_DIRECTORY="/mnt/cup/labs/falkner/Bartul/spectrograms/sam/manual_labels"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

JOB_SCRIPT="$WORK_DIR/train_masks_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=train-masks" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/train-masks-%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/train-masks-%j.err" >> "$JOB_SCRIPT"
echo "#SBATCH --gpus=1" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem=$TOTAL_MEMORY" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "set -e" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "source $USV_PLAYPEN_PATH/.venv/bin/activate" >> "$JOB_SCRIPT"
echo "(cd $USV_PLAYPEN_PATH && uv sync --extra gpu)" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
EXPORT_DATASET_CMD="export-yolo-dataset --root-directories \"$SESSION_ROOT_DIRECTORIES\" --output-directory \"$DATASET_DIRECTORY\" --label-source \"$LABEL_SOURCE\""
if [ "$LABEL_SOURCE" != "cc" ]; then
    EXPORT_DATASET_CMD="$EXPORT_DATASET_CMD --manual-labels-directory \"$MANUAL_LABELS_DIRECTORY\""
fi
echo "$EXPORT_DATASET_CMD" >> "$JOB_SCRIPT"
echo "train-masks --dataset-directory \"$DATASET_DIRECTORY\" --output-directory \"$MODEL_OUTPUT_DIRECTORY\"" >> "$JOB_SCRIPT"
echo "echo 'YOLO mask-detector training (export dataset + train) completed successfully.'" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# --------------------- RUN JOB -------------------- #

sbatch "$JOB_SCRIPT"
