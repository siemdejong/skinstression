#!/bin/bash
#SBATCH --job-name=shg
#SBATCH --output=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.out
#SBATCH --error=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.err
#SBATCH --nodes=3
#SBATCH --gpus-per-node 2
#SBATCH --mem-per-gpu 5G
#SBATCH --cpus-per-gpu 10
#SBATCH --mail-user=siem.de.jong@student.vu.nl
#SBATCH --mail-type=ALL

source /scistor/guest/sjg203/.bashrc

export HYDRA_FULL_ERROR=0
export NCCL_ASYNC_ERROR_HANDLING=1


for (( i=0; i<$SLURM_NNODES; i++ ))
do
    echo "r$i | Starting Optuna process. "
    srun --nodes 1 python src/main.py &  # Start an Optuna process in the background.
    sleep 10  # To reduce the chance that Optuna processes share one hydra output dir.
done

wait

echo "All Optuna processes exited."