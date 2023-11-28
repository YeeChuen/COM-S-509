#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename

#SBATCH --time=48:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node 
#SBATCH --mem=64G   # maximum memory per node
#SBATCH --job-name="LR_model"
#SBATCH --output="LR_model_training_%j" # job standard output file (%j replaced by job id)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# make sure anaconda is activated
module load miniconda3/22.11.1-hydt3qz
source activate viral_escape

# go to the example directory
cd /work/ratul1/chuen/temp/COM-S-573/LR_model

# execute
python main.py
