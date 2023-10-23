#!/bin/bash
#SBATCH --job-name=my_job_name
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=my_job_output_%j.txt

# Load any necessary modules and activate your conda environment
#module load cuda/10.0
source /opt/conda/bin/activate alpha
#source activate /home/DAVIDSON/bewagner/.conda/envs/summer2023test/ 

# Now run your script
sh scripts/train.sh 0 --config ATTPCPoinTr/ATTPCPoinTr/cfgs/Mg22_Ne20pp_models/HalfCutSnowflake.yaml --exp_name half_cut_test

