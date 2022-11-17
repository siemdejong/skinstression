#!/bin/bash
#SBATCH --job-name=shg
#SBATCH --output=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.out
#SBATCH --error=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.err
#SBATCH --nodes 4
#SBATCH --gpus-per-node 2
#SBATCH --mem-per-gpu 5G
#SBATCH --cpus-per-gpu 10
#SBATCH --mail-user=siem.de.jong@student.vu.nl
#SBATCH --mail-type=ALL

# Please make sure config.dist contains the same allocation specifications.

source /scistor/guest/sjg203/.bashrc

# Uncomment if debugging parallel processes.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

export HYDRA_FULL_ERROR=1

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(srun --nodes 1 --ntasks 1 -w "$head_node" hostname --ip-address)
head_node_port=29500
rdzv_backend=c10d

echo Head node IP: $head_node_ip:$head_node_port

srun python src/main.py -m hydra/launcher=torchrun \
    hydra.launcher.min_nodes=$SLURM_JOB_NUM_NODES \
    hydra.launcher.max_nodes=$SLURM_JOB_NUM_NODES \
    hydra.launcher.nproc_per_node=$SLURM_GPUS_PER_NODE \
    hydra.launcher.rdzv_id=$SLURM_JOB_ID \
    hydra.launcher.rdzv_endpoint=$head_node_ip:$head_node_port \
    hydra.launcher.rdzv_backend=$rdzv_backend
