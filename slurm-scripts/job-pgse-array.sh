#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J pgse-k6
#SBATCH -p nodes           # Ensure this is the correct partition
#SBATCH -o slurm-pgse-k6-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=4:00:00
#SBATCH --array=0-9

# Optionally set OMP_NUM_THREADS if using OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WORKERS_PER_NODE=39

## Array of label names
LABELS=("AMC" "AMK" "AMX" "CAZ" "CHL" "CIP" "FEP" "GEN" "MEM" "TGC")
EA_MAXES=(64 64 64 64 64 4 64 64 16 999)
EA_MINS=(0 0.03 0 0 0.03 0 0.03 0.125 0.004 0.125)

# Select label based on the array task ID
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}
EA_MAX=${EA_MAXES[$SLURM_ARRAY_TASK_ID]}
EA_MIN=${EA_MINS[$SLURM_ARRAY_TASK_ID]}

date
echo "This code is running on"
hostname
echo "Starting processing for label $LABEL on host $HOSTNAME"

conda activate genome

# Run the Python script with label-specific files
pgse-train \
        --label-file "../volatile/cgr_labels_new/cgr_label_${LABEL}.csv" \
        --data-dir "../volatile/cgr/" \
        --pre-kfold-info-file "../volatile/cgr_labels_new/cgr_label_${LABEL}_kfold.json" \
        --save-file "../volatile/var/${LABEL}-k6.save" \
        --export-file "../volatile/var/result-${LABEL}-k6" \
        --workers $WORKERS_PER_NODE \
        --features 10000 \
        --dist 0 \
        --k 6 \
        --target 70 \
        --ext 2 \
        --lr 0.001 \
        --num-rounds 6000 \
        --folds 0 \
        --ea-max $EA_MAX \
        --ea-min $EA_MIN

echo "Finished processing for label $LABEL - goodbye from $HOSTNAME"
