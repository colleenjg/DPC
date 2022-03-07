#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=1:30:00
#SBATCH -o /home/mila/g/gillonco/scratch/losses_%j.txt

# 1. Set directory in which to load and plot results (also update output directory above)
SAVE_DIR=/home/mila/g/gillonco/scratch

if [[ ! -d $SAVE_DIR ]]; then
    echo -e "Save directory '$SAVE_DIR' does not exist. Please set SAVE_DIR "\
            "to the path of an existing directory."
    exit 1
fi


# 2. Load modules
module purge
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 3. Launch your job
EXIT=0
set -x # echo commands to console

python plot_losses.py --direc $SAVE_DIR/test_gabors

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code
