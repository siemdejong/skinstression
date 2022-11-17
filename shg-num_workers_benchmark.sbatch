#!/bin/bash
#SBATCH --job-name=shg
#SBATCH --output=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.out
#SBATCH --error=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.err
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --mem-per-gpu 5G
#SBATCH --cpus-per-gpu 16

# Please make sure config.dist contains the same allocation specifications.

source /scistor/guest/sjg203/.bashrc

# Uncomment if debugging parallel processes.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

srun python src/main.py