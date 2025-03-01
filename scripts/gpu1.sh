#!/bin/bash -e

#SBATCH --job-name=enhance # create a short name for your job
#SBATCH --output=./log/%A.out # create a output file
#SBATCH --error=./log/%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=64GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.ngannh9@vinai.io
eval "$(conda shell.bash hook)"
conda deactivate
conda deactivate
conda activate /lustre/scratch/client/vinai/users/ngannh9/diffuser/env/diffuser
# ./cmmd.sh
coco/fid.sh