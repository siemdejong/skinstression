#!/bin/bash
#SBATCH --job-name=shg
#SBATCH --output=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.out
#SBATCH --error=/scistor/guest/sjg203/projects/shg-strain-stress/slurm/%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=16
#SBATCH --mail-user=siem.de.jong@student.vu.nl
#SBATCH --mail-type=ALL

source /scistor/guest/sjg203/.bashrc

# if some error happens in the initialation of parallel process then you can
# get the debug info. This can easily increase the size of out.txt.
export NCCL_DEBUG=INFO  # comment it if you are not debugging distributed parallel setup

export NCCL_DEBUG_SUBSYS=ALL # comment it if you are not debugging distributed parallel setup

# Find the ip-address of one of the node. Treat it as master
ip1=`hostname -I | awk '{print $2}'`
echo $ip1

# Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_ADDR=$(hostname)

echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID Launching python script"

srun python train.py --nodes=2 --ngpus 2 --ip_adress $ip1
