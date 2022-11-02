#!/bin/bash
#SBATCH --job-name=shg
#SBATCH --output=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.out
#SBATCH --error=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
##SBATCH --cpus-per-gpu=10
#SBATCH --mail-user=siem.de.jong@student.vu.nl
#SBATCH --mail-type=ALL

source /scistor/guest/sjg203/.bashrc

export HYDRA_FULL_ERROR=0

echo "r$SLURM_NODEID: Launching python script"

srun \
    --nodes 1
    --gpus-per-node 2
    --cpus-per-gpu 10
    --mem-per-gpu 5G
    python src/main.py
