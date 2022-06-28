#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48GB
#SBATCH --array=0-5
#SBATCH --time=5:00:00
#SBATCH --job-name=dpc_gab
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_gabors_%A_%a.txt


# 1. Load modules
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Fixed hyperparameters
FIXED_DATASET=Gabors
FIXED_MODEL="dpc-rnn"
FIXED_NET=resnet18
FIXED_TRAIN_WHAT=all
FIXED_IMG_DIM=128
FIXED_TRANSF_ARG="--no_transforms"
FIXED_BATCH_SIZE=32
FIXED_TRAIN_LEN=500
FIXED_GAB_IMG_LEN=3
FIXED_U_PROB=0.1
FIXED_NUM_WORKERS=8

echo -e "\nFIXED HYPERPARAMETERS\n"\
"dataset: $FIXED_DATASET\n"\
"model: $FIXED_MODEL\n"\
"net: $FIXED_NET\n"\
"train what: $FIXED_TRAIN_WHAT\n"\
"image dimensions: $FIXED_IMG_DIM\n"\
"transform argument: $FIXED_TRANSF_ARG\n"\
"batch size: $FIXED_BATCH_SIZE\n"\
"training dataset length: $FIXED_TRAIN_LEN\n"\
"Gabor image length: $FIXED_GAB_IMG_LEN\n"\
"U probability: $FIXED_U_PROB\n"\
"number of workers: $FIXED_NUM_WORKERS\n"\


# 3. Externally set parameters
if [[ $SEED == "" ]]; then
    SEED_ARG=""
else
    SEED_ARG="--seed "$SEED
fi

echo -e "EXTERNALLY SET HYPERPARAMETERS\n"\
"seed argument: $SEED_ARG\n"\


# 4. Array ID hyperparameters
PRETRAINEDS=( no yes ) # 2
SIMPLES=( no semi full ) # 3

# set values for task ID
# inner loop
PRETRAINEDS_LEN=${#PRETRAINEDS[@]}
PRETRAINEDS_IDX=$(( $SLURM_ARRAY_TASK_ID % $PRETRAINEDS_LEN ))
PRETRAINED=${PRETRAINEDS[$PRETRAINEDS_IDX]}

# outer loop
SIMPLES_LEN=${#SIMPLES[@]}
SIMPLES_IDX=$(( $SLURM_ARRAY_TASK_ID / $PRETRAINEDS_LEN % $SIMPLES_LEN ))
SIMPLE=${SIMPLES[$SIMPLES_IDX]}

# Set the pretrain path, if applicable
if [[ $PRETRAINED == yes ]]; then
    PRETRAINED_ARG="--pretrained "$SCRATCH"/dpc/models/mousesim_left-128_r18_dpc-rnn_bs10_lr0.0001_nseq25_pred3_len5_train-all_seed100/model/model_best_epoch975.pth.tar"
    UNEXP_EPOCH=20
else
    PRETRAINED_ARG=""
    UNEXP_EPOCH=30
fi

# Set the sequence parameters
if [[ $SIMPLE == full ]]; then
    SEQ_LEN=$FIXED_GAB_IMG_LEN
    PRED_STEP=1
    NUM_SEQ=5
    ROLL_ARG=""
else
    if [[ $SIMPLE == semi ]]; then
        SEQ_LEN=$FIXED_GAB_IMG_LEN
        PRED_STEP=1
    else
        SEQ_LEN=6
        PRED_STEP=3 
    fi

    NUM_SEQ=$(expr $PRED_STEP + 5)
    ROLL_ARG="--roll"
fi

NUM_EPOCHS=$(expr $UNEXP_EPOCH \* 2 )


echo -e "\nARRAY ID HYPERPARAMETERS\n"\
"unexpected epoch: $UNEXP_EPOCH\n"\
"sequence length: $SEQ_LEN\n"\
"number of steps to predict: $PRED_STEP\n"\
"number of sequences per batch item: $NUM_SEQ\n"\
"roll argument: $ROLL_ARG\n"\
"number of epochs: $NUM_EPOCHS\n"\
"pretrained argument: $PRETRAINED_ARG\n"\


# 5. Train model
EXIT=0

set -x # echo commands to console

python train_model.py \
    --output_dir $SCRATCH/dpc/models \
    --dataset $FIXED_DATASET \
    --model $FIXED_MODEL \
    --net $FIXED_NET \
    --train_what $FIXED_TRAIN_WHAT \
    --img_dim $FIXED_IMG_DIM \
    $FIXED_TRANSF_ARG \
    --batch_size $FIXED_BATCH_SIZE \
    --train_len $FIXED_TRAIN_LEN \
    --gab_img_len $FIXED_GAB_IMG_LEN \
    --U_prob $FIXED_U_PROB \
    --num_workers $FIXED_NUM_WORKERS \
    $SEED_ARG \
    --unexp_epoch $UNEXP_EPOCH \
    --seq_len $SEQ_LEN \
    --pred_step $PRED_STEP \
    --num_seq $NUM_SEQ \
    $ROLL_ARG \
    --num_epochs $NUM_EPOCHS \
    $PRETRAINED_ARG \

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console
  

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code

