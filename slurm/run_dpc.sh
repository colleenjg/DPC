#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48GB
#SBATCH --array=0-7
#SBATCH --time=8:00:00
#SBATCH --job-name=dpc
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_%A_%a.out


# 1. Load modules
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Array ID hyperparameters
PRETRAINEDS=( no yes ) # 2
DATASETS=( MouseSim HMDB51 UCF101 Kinetics400 ) # 4

# set values for task ID
# inner loop
PRETRAINEDS_LEN=${#PRETRAINEDS[@]}
PRETRAINEDS_IDX=$(( $SLURM_ARRAY_TASK_ID % $PRETRAINEDS_LEN ))
PRETRAINED=${PRETRAINEDS[$PRETRAINEDS_IDX]}

# outer loop
DATASETS_LEN=${#DATASETS[@]}
DATASETS_IDX=$(( $SLURM_ARRAY_TASK_ID / $PRETRAINEDS_LEN % $DATASETS_LEN ))
DATASET=${DATASETS[$DATASETS_IDX]}


# 3. Set some variables
if [[ $SEED == "" ]]; then
    SEED=100
fi

if [[ $NUM_EPOCHS == "" ]]; then
    NUM_EPOCHS=100
fi

BATCH_SIZE=64
if [[ $DATASET == "MouseSim" ]]; then
    NUM_EPOCHS=1000
    BATCH_SIZE=10
    LR_ARG="--lr 0.0001"
    NUM_SEQ_ARG="--num_seq 25"
fi


if [[ $PRETRAINED == yes ]]; then
    MODEL="lc-rnn"
    TRAIN_WHAT=ft
    PRETRAINED_ARG="--pretrained "$SCRATCH"/dpc/pretrained/k400_128_r18_dpc-rnn/model/k400_128_r18_dpc-rnn.pth.tar"
else
    MODEL="dpc-rnn"
    TRAIN_WHAT=all
    PRETRAINED_ARG=""
fi


# 4. Train model
EXIT=0

set -x # echo commands to console

python train_model.py \
    --output_dir $SCRATCH/dpc/models \
    --net resnet18 \
    --img_dim 128 \
    --num_workers 8 \
    --batch_size $BATCH_SIZE \
    --dataset $DATASET \
    --model $MODEL \
    --num_epochs $NUM_EPOCHS \
    --train_what $TRAIN_WHAT \
    $LR_ARG \
    $NUM_SEQ_ARG \
    --seed $SEED \
    $PRETRAINED_ARG \

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console
  

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code

