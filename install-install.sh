#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "BW-ptr-train"               # name
#SBATCH --output "install-out.log"      # output file
#SBATCH --error "install-err.log"       # error message file

# resource request info 
#SBATCH --mem-per-cpu 32g            # Memory to request per CPU
#SBATCH --gpus 1                    # GPUs to request in total

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user bewagner@davidson.edu

## Script to Execute:
# change working directory to pipenv managed directory

source /opt/conda/bin/activate piptorch
cd ~/summer2023/ATTPCPoinTr

# execute python script in virtal env.
sh ./install.sh