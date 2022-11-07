#!/bin/bash
#SBATCH --job-name=shg
#SBATCH --output=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.out
#SBATCH --error=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.err
#SBATCH --nodes=3
#SBATCH --gpus-per-node 2
#SBATCH --mem-per-gpu 5G
#SBATCH --cpus-per-gpu 13
#SBATCH --mail-user=siem.de.jong@student.vu.nl
#SBATCH --mail-type=ALL

# Make sure hydra.dist contains the same allocation specification.

source /scistor/guest/sjg203/.bashrc

# Uncomment if debugging parallel processes.
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

export HYDRA_FULL_ERROR=0
export NCCL_ASYNC_ERROR_HANDLING=1

export MASTER_ADDR=$(hostname)
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()');
export MASTER_PORT=$PORT


for (( i=0; i<$SLURM_NNODES; i++ ))
do
    echo "r$i | Starting training process. "
    export SLURM_RANK=$i
    srun --nodes 1 python src/main.py &  # Start an Optuna process in the background.
    sleep 10  # To reduce the chance that Optuna processes share one hydra output dir.
done

wait

echo "All Optuna processes exited."