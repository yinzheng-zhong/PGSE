#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J main-rec
#SBATCH -p nodes           # Ensure this is the correct partition
#SBATCH -o slurm-pgse.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=3-00:00:00

# Optionally set OMP_NUM_THREADS if using OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

date
echo "This code is running on "
hostname
echo "Starting running on host $HOSTNAME"

conda activate genome

# Change this to the your paths
python3 main-pgse.py \
        --label-file "../volatile/e_coli_mic_label.csv" \
        --data-dir "../volatile/e_coli_mic/" \
        --save-file "../volatile/var/pgse-70.save" \
        --export-file "../volatile/var/pgse-result.txt" \
        --workers 76  \
        --features 10000 \

echo "Finished running - goodbye from $HOSTNAME"
