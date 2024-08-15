#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J main-rec
#SBATCH -p lowpriority,nodes           # Ensure this is the correct partition
#SBATCH -o slurm-rec-dist.out
#SBATCH -N 4                  # Number of nodes
#SBATCH --ntasks-per-node=1   # Number of tasks per node
#SBATCH --cpus-per-task=40 # CPUs per task
#SBATCH --time=72:00:00

# Optionally set OMP_NUM_THREADS if using OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate the conda environment first
conda activate genome

# Get the list of nodes
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# shellcheck disable=SC2206
NODELIST=($NODES)
HEAD_NODE=${NODELIST[0]}
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)
HEAD_PORT=8000  # Using the original port number

echo "NODELIST: $NODES"
echo "HEAD_NODE: $HEAD_NODE"
echo "HEAD_ADDR: $HEAD_ADDR"
echo "HEAD_PORT: $HEAD_PORT"
echo "SLURM_NODEID: $SLURM_NODEID"

echo "Starting HEAD at $HEAD_NODE"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --node-ip-address="$HEAD_ADDR" --port=$HEAD_PORT \
    --num-cpus 38 --block &

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
# print the number of worker nodes
echo "Number of worker nodes: $worker_num"

for ((i = 1; i <= worker_num; i++)); do
    node_i=${NODELIST[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$HEAD_ADDR":"$HEAD_PORT" \
        --num-cpus 38 --block &
    sleep 5
done

python3 main-recursive.py --worker $((SLURM_CPUS_PER_TASK - 2)) --save-file "rec-70-1.txt" --features 1500 --dist 1
echo "Finished running - goodbye from $HOSTNAME"

# Stop Ray
ray stop
