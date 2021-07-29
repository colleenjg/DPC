#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=cn:48GB:2
#SBATCH --mem=48GB
#SBATCH --array=0
#SBATCH --time=8:00:00
#SBATCH --job-name=dpc
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_%j.out


# 1. Load modules
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Set some variables
if [[ $SEED == "" ]]; then
    SEED=100
fi

if [[ $DATASET == "" ]]; then
    DATASET=k400
fi

if [[ $NUM_EPOCHS == "" ]]; then
    NUM_EPOCHS=100
fi

if [[ $PRETRAINED == 1 ]]; then
    MODEL="lc-rnn"
    TRAIN_WHAT=ft
    PRETRAINED="--pretrained "$SCRATCH"/dpc/pretrained/k400_128_r18_dpc-rnn/model/k400_128_r18_dpc-rnn.pth.tar"
else
    MODEL="dpc-rnn"
    TRAIN_WHAT=all
    PRETRAINED=""
fi


# 3. Train model
EXIT=0

set -x # echo commands to console

python train_model.py \
    --output_dir $SCRATCH/dpc/models \
    --net resnet18 \
    --img_dim 128 \
    --batch_size 64 \
    --num_workers 8 \
    --dataset $DATASET \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --train_what $TRAIN_WHAT \
    --seed $SEED \
    $PRETRAINED \

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console
  

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code

