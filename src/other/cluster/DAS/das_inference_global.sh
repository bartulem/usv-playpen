#!/bin/bash

# Usage: bash das_inference_global.sh

# -------------------------------------------------- #
# ------------- SELECT HYPER-PARAMETERS ------------ #

CUP_ROOT_DIR="falkner/Name/Data"
WORK_DIR="/mnt/cup/labs/falkner/Name/DAS"
AUDIO_CH_NUM=24
CPUS_PER_TASK=2
MEMORY_PER_CPU="64G"
TIME_RESTRICTION="01:02:00"
EMAIL_ADDRESS="nsurname@domain.edu"
EMAIL_TYPE="NONE"
CONDA_NAME="anacondapy"
CONDA_DATE="2024.02"
CONDA_NAME_UPPERCASE="${CONDA_NAME^^}"
CONDA_VERSION="$CONDA_NAME/$CONDA_DATE"
DAS_ENV="das"
DAS_CONFIDENCE_THRESHOLD=0.5
DAS_SEGMENT_MIN_LEN=0.015
DAS_SEGMENT_FILL_GAP=0.015
DAS_SAVE_FORMAT="csv"

# -------------------------------------------------- #
# ---------------- CREATE JOB SCRIPT --------------- #

JOB_SCRIPT="$WORK_DIR/das_inference_settings.sh"

mkdir -p "$WORK_DIR/logs"

touch "$JOB_SCRIPT"
echo "#!/bin/bash" > "$JOB_SCRIPT"
echo "#SBATCH --job-name=das_inference" >> "$JOB_SCRIPT"
echo "#SBATCH --output=$WORK_DIR/logs/das_inference_%j.out" >> "$JOB_SCRIPT"
echo "#SBATCH --error=$WORK_DIR/logs/das_inference_%j.err" >> "$JOB_SCRIPT"
echo "#SBATCH --gpus=1" >> "$JOB_SCRIPT"
echo "#SBATCH --cpus-per-task=$CPUS_PER_TASK" >> "$JOB_SCRIPT"
echo "#SBATCH --mem-per-cpu=$MEMORY_PER_CPU" >> "$JOB_SCRIPT"
echo "#SBATCH --time=$TIME_RESTRICTION" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-type=$EMAIL_TYPE" >> "$JOB_SCRIPT"
echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "das_confidence_threshold=$DAS_CONFIDENCE_THRESHOLD" >> "$JOB_SCRIPT"
echo "das_segment_min_len=$DAS_SEGMENT_MIN_LEN" >> "$JOB_SCRIPT"
echo "das_segment_fill_gap=$DAS_SEGMENT_FILL_GAP" >> "$JOB_SCRIPT"
echo "das_save_format=\"$DAS_SAVE_FORMAT\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "das_model_base=\"$4/model_2024-03-25/20240325_073951\"" >> "$JOB_SCRIPT"
echo "inference_file=\$(ls /mnt/cup/labs/\$2/\$1/audio/hpss_filtered/*.wav | sort | sed -n \"\$3 p\")" >> "$JOB_SCRIPT"
echo "inference_file_base_name=\"\${inference_file%.*}\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "echo \"Performing DAS inference on \$inference_file.\"" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "module load $CONDA_VERSION" >> "$JOB_SCRIPT"
echo "source /mnt/cup/PNI-facilities/Computing/sw/pkg/Rhel9/$CONDA_NAME_UPPERCASE/$CONDA_DATE/etc/profile.d/conda.sh" >> "$JOB_SCRIPT"
echo "conda activate $DAS_ENV" >> "$JOB_SCRIPT"
echo "" >> "$JOB_SCRIPT"
echo "das predict \"\$inference_file\" \"\$das_model_base\" --segment-thres \"\$das_confidence_threshold\" --segment-minlen \"\$das_segment_min_len\" --segment-fillgap \"\$das_segment_fill_gap\" --save-format \"\$das_save_format\" && mv \"\${inference_file_base_name}_annotations.\${das_save_format}\" \"/mnt/cup/labs/\$2/\$1/audio/das_annotations\"" >> "$JOB_SCRIPT"

# -------------------------------------------------- #
# ---------------- CREATE JOB ARRAY ---------------- #

ARRAY_ARGS_FILE="$WORK_DIR/job_list.txt"

DIR_NUM=$(cat $ARRAY_ARGS_FILE | wc -l)
NUM_JOBS=$((DIR_NUM*"$AUDIO_CH_NUM"))

echo "Jobs: $NUM_JOBS"

for i in $(seq 1 "$DIR_NUM");
do
    session_id=$(sed -n "$i p" $ARRAY_ARGS_FILE)
    session_id=$(echo "$session_id" | tr -d '\r\n')
	mkdir -p "/mnt/cup/labs/$CUP_ROOT_DIR/$session_id/audio/das_annotations"
	
    for j in $(seq 1 $AUDIO_CH_NUM);
    do  
        sbatch "$JOB_SCRIPT" "$session_id" "$CUP_ROOT_DIR" "$j" "$WORK_DIR"
    done
done

