#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48GB
#SBATCH --array=0
#SBATCH --time=5:00:00
#SBATCH --job-name=dpc_gab
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_gabors_%j.txt


# 1. Load modules
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Set some variables
if [[ $SEED == "" ]]; then
    SEED=100
fi

SEQ_LEN=6
NUM_SEQ=8
GAB_IMG_LEN=3

BATCH_SIZE=64
DATASET=Gabors
ROLL_ARG="--roll"
TRAIN_LEN=500
U_PROB=0.5
TRANSF_ARG="--no_transforms"

# 3. Set the pretrain path, if applicable
if [[ $PRETRAINED == 1 ]]; then
    MODEL=dpc-rnn
    TRAIN_WHAT=all
    PRETRAINED_ARG="--pretrained "$SCRATCH"/dpc/models/mousesim_left-128_r18_dpc-rnn_bs10_lr0.0001_nseq25_pred3_len5_train-all_seed100/model/model_best_epoch889.pth.tar"
    NUM_EPOCHS=80
    UNEXP_EPOCH=40
else
    MODEL=dpc-rnn
    TRAIN_WHAT=all
    PRETRAINED_ARG=""
    NUM_EPOCHS=100
    UNEXP_EPOCH=50
fi

# 3. Train model
EXIT=0

set -x # echo commands to console

python train_model.py \
    --output_dir $SCRATCH/dpc/models \
    --net resnet18 \
    --img_dim 128 \
    --num_workers 8 \
    --seed $SEED \
    --seq_len $SEQ_LEN \
    --num_seq $NUM_SEQ \
    --gab_img_len $GAB_IMG_LEN \
    --batch_size $BATCH_SIZE \
    --dataset $DATASET \
    $ROLL_ARG \
    --train_len $TRAIN_LEN \
    --U_prob $U_PROB \
    $TRANSF_ARG \
    --model $MODEL \
    --train_what $TRAIN_WHAT \
    $PRETRAINED_ARG \
    --num_epochs $NUM_EPOCHS \
    --unexp_epoch $UNEXP_EPOCH \

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console
  

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code

