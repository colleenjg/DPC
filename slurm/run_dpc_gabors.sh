#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48GB
#SBATCH --array=0-5
#SBATCH --time=5:00:00
#SBATCH --job-name=dpc_gab
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_gabors_%A_%a.out


# Current longest: ?

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
FIXED_AUGM_ARG="--no_augm"
FIXED_BATCH_SIZE=32
FIXED_GAB_IMG_LEN=5
FIXED_SEQ_LEN="$FIXED_GAB_IMG_LEN"
FIXED_U_PROB=0.08
FIXED_U_POSSIZE_ARG="--diff_U_possizes"
FIXED_NUM_WORKERS=8


echo -e "\nFIXED HYPERPARAMETERS\n" \
"dataset: $FIXED_DATASET\n" \
"model: $FIXED_MODEL\n" \
"net: $FIXED_NET\n" \
"train what: $FIXED_TRAIN_WHAT\n" \
"image dimensions: $FIXED_IMG_DIM\n" \
"augmentation argument: $FIXED_AUGM_ARG\n" \
"batch size: $FIXED_BATCH_SIZE\n" \
"Gabor image length: $FIXED_GAB_IMG_LEN\n" \
"sequence length: $FIXED_SEQ_LEN\n" \
"U probability: $FIXED_U_PROB\n" \
"U position/size argument: $FIXED_U_POSSIZE_ARG\n" \
"number of workers: $FIXED_NUM_WORKERS\n" \


# 3. Externally set parameters
if [[ "$SEED" == "" ]]; then
    SEED_ARG=""
else
    SEED_ARG="--seed $SEED"
fi

echo -e "EXTERNALLY SET HYPERPARAMETERS\n" \
"seed argument: $SEED_ARG\n" \


# 4. Array ID hyperparameters
PRETRAINEDS=( "MouseSim" "Kinetics400" "no" ) # 3
ROLL_ARGS=( "" "--roll" ) # 2

# set values for task ID
# inner loop
PRETRAINEDS_LEN=${#PRETRAINEDS[@]}
PRETRAINEDS_IDX=$(( SLURM_ARRAY_TASK_ID % PRETRAINEDS_LEN ))
PRETRAINED=${PRETRAINEDS[$PRETRAINEDS_IDX]}

# outer loop
ROLL_ARGS_LEN=${#ROLL_ARGS[@]}
ROLL_ARGS_IDX=$(( SLURM_ARRAY_TASK_ID / PRETRAINEDS_LEN % ROLL_ARGS_LEN ))
ROLL_ARG=${ROLL_ARGS[$ROLL_ARGS_IDX]}


# Set the pretrain path, if applicable
if [[ "$PRETRAINED" == "MouseSim" ]]; then
    PRETRAINED_ARG="--pretrained ${SCRATCH}/dpc/pretrained/mousesim_left-128_r18_dpc-rnn/model/mousesim_left_best_epoch853.pth.tar"
elif [[ "$PRETRAINED" == "Kinetics400" ]]; then
    PRETRAINED_ARG="--pretrained ${SCRATCH}/dpc/pretrained/k400_128_r18_dpc-rnn/model/k400_128_r18_dpc-rnn.pth.tar"
else
    PRETRAINED_ARG=""
fi

# Set the sequence parameters
if [[ "$ROLL_ARG" == "" ]]; then
    PRED_STEP=1 # predict D/U only
    NUM_SEQ_IN=4 # gray will never appear
    TRAIN_LEN=1024
else
    PRED_STEP=2
    NUM_SEQ_IN=5
    TRAIN_LEN=4096
fi

if [[ "$PRETRAINED_ARG" == "" ]]; then
    UNEXP_EPOCH=20
else
    UNEXP_EPOCH=10
fi

NUM_EPOCHS=$(( UNEXP_EPOCH + 5 ))

if [[ "$ROLL_ARG" ]]; then
    if [[ $SUFFIX_ARG ]]; then
        SUFFIX_ARG="${SUFFIX_ARG}_roll"
    else
        SUFFIX_ARG="--suffix roll"
    fi
fi


echo -e "\nARRAY ID HYPERPARAMETERS\n" \
"roll argument: $ROLL_ARG\n" \
"number of steps to predict: $PRED_STEP\n" \
"number of consecutive blocks to use as input: $NUM_SEQ_IN\n" \
"training dataset length: $TRAIN_LEN\n" \
"unexpected epoch: $UNEXP_EPOCH\n" \
"number of epochs: $NUM_EPOCHS\n" \
"suffix argument: $SUFFIX_ARG\n" \
"pretrained argument: $PRETRAINED_ARG\n" \


# 5. Train model
EXIT=0

set -x # echo commands to console

python run_model.py \
    --output_dir "$SCRATCH/dpc/gabor_models" \
    --dataset "$FIXED_DATASET" \
    --model "$FIXED_MODEL" \
    --net "$FIXED_NET" \
    --train_what "$FIXED_TRAIN_WHAT" \
    --img_dim "$FIXED_IMG_DIM" \
    $FIXED_AUGM_ARG \
    --batch_size "$FIXED_BATCH_SIZE" \
    --gab_img_len "$FIXED_GAB_IMG_LEN" \
    --seq_len "$FIXED_SEQ_LEN" \
    --U_prob "$FIXED_U_PROB" \
    $FIXED_U_POSSIZE_ARG \
    --num_workers "$FIXED_NUM_WORKERS" \
    $SEED_ARG \
    $ROLL_ARG \
    --pred_step "$PRED_STEP" \
    --num_seq_in "$NUM_SEQ_IN" \
    --train_len "$TRAIN_LEN" \
    --unexp_epoch "$UNEXP_EPOCH" \
    --num_epochs "$NUM_EPOCHS" \
    $SUFFIX_ARG \
    $PRETRAINED_ARG \
    --log_test_cmd \

code="$?"
if [[ "$code" -gt "$EXIT" ]]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console
  

if [[ "$EXIT" -ne 0 ]]; then exit "$EXIT"; fi # exit with highest exit code

