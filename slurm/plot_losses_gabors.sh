#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --gres=cn:12GB:2
#SBATCH --mem=12GB
#SBATCH --array=0
#SBATCH --time=4:00:00
#SBATCH --job-name=dpc_loss_gab
#SBATCH -o /home/mila/g/gillonco/scratch/dpc_losses_gabors_%j.txt

# 1. Load modules
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Launch your job
EXIT=0
set -x # echo commands to console

python plot_losses.py --direc $SCRATCH/dpc

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code
