#!/bin/sh
#SBATCH --job-name=setup_slurm     # Specify the job name
#SBATCH --output=setup_slurm.out   # Output file name
#SBATCH --partition=normal               # Partition name (queue name)
#SBATCH --gres=gpu:1                  # Number of GPUs required. Adjust as needed.
# Activate the environment
source /opt/conda/bin/activate alpha
# Set the default CUDA location
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# List of extension folders
EXTENSION_FOLDERS=("Pointnet2_PyTorch")
cd ~/ATTPCPoinTr/extensions/Pointnet2_PyTorch/
python setup.py install --user
