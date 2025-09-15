#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J pgse
#SBATCH -p nodes           # Ensure this is the correct partition
#SBATCH -o slurm-pgse-param-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00

# Optionally set OMP_NUM_THREADS if using OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WORKERS_PER_NODE=39

date
echo "This code is running on "
hostname
echo "Starting running on host $HOSTNAME"

conda activate genome

# Change this to the your paths
pgse-train \
        --label-file "../volatile/cgr_labels/cgr_label_CAZ.csv" \
        --data-dir "../volatile/cgr/" \
        --save-file "../volatile/var/CAZ.save" \
        --export-file "../volatile/var/result-CAZ" \
        --workers $WORKERS_PER_NODE \
        --features 10000 \
        --dist 0 \
        --k 10 \
        --target 70 \
        --ext 2 \
        --lr 0.001 \
        --num-rounds 6000 \
        --folds 5

echo "Finished running - goodbye from $HOSTNAME"
