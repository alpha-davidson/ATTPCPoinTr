#!/bin/bash
### Job Parameters:
# basic info
#SBATCH --job-name "chamf-setup"               # name
#SBATCH --output "setup-out.log"      # output file
#SBATCH --error "setup-err.log"       # error message file

# resource request info 
#SBATCH --mem-per-cpu 32g            # Memory to request per CPU
#SBATCH --gpus 1                    # GPUs to request in total

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user bewagner@davidson.edu

## Script to Execute:
# change working directory to pipenv managed directory

cd ~/summer2023/PoinTr/extensions/gridding_loss

# execute python script in virtal env.
source /opt/conda/bin/activate piptorch
python setup.py install --user