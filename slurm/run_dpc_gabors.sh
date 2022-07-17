#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48GB
#SBATCH --array=0-23
#SBATCH --time=5:00:00
#SBATCH --job-name=dpc_gab
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_gabors_%A_%a.txt


# 1. Load modules
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Fixed hyperparameters
FIXED_DATASET=Gabors
FIXED_NET=resnet18
FIXED_TRAIN_WHAT=all
FIXED_IMG_DIM=128
FIXED_AUGM_ARG="--no_augm"
FIXED_BATCH_SIZE=32
FIXED_TRAIN_LEN=500
FIXED_GAB_IMG_LEN=3
FIXED_U_PROB=0.1
FIXED_NUM_WORKERS=8

echo -e "\nFIXED HYPERPARAMETERS\n"\
"dataset: $FIXED_DATASET\n"\
"net: $FIXED_NET\n"\
"train what: $FIXED_TRAIN_WHAT\n"\
"image dimensions: $FIXED_IMG_DIM\n"\
"augmentation argument: $FIXED_AUGM_ARG\n"\
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

U_POSSIZE_ARGS=( "" "--diff_U_possizes" ) # 2
SUFFIX_ARGS=( "" "--suffix diff_U_possizes" )

MODELS=( "dpc-rnn" "lc-rnn" ) # 2

# set values for task ID
# inner loop
PRETRAINEDS_LEN=${#PRETRAINEDS[@]}
PRETRAINEDS_IDX=$(( $SLURM_ARRAY_TASK_ID % $PRETRAINEDS_LEN ))
PRETRAINED=${PRETRAINEDS[$PRETRAINEDS_IDX]}

# middle-inner loop
SIMPLES_LEN=${#SIMPLES[@]}
SIMPLES_IDX=$(( $SLURM_ARRAY_TASK_ID / $PRETRAINEDS_LEN % $SIMPLES_LEN ))
SIMPLE=${SIMPLES[$SIMPLES_IDX]}

# middle-outer loop
U_POSSIZE_ARGS_LEN=${#U_POSSIZE_ARGS[@]}
U_POSSIZE_ARGS_IDX=$(( $SLURM_ARRAY_TASK_ID / ( $PRETRAINEDS_LEN * $SIMPLES_LEN ) % $U_POSSIZE_ARGS_LEN ))
U_POSSIZE_ARG=${U_POSSIZE_ARGS[$U_POSSIZE_ARGS_IDX]}
SUFFIX_ARG=${SUFFIX_ARGS[$U_POSSIZE_ARGS_IDX]}

# outer loop
MODELS_LEN=${#MODELS[@]}
MODELS_IDX=$(( $SLURM_ARRAY_TASK_ID / ( $PRETRAINEDS_LEN * $SIMPLES_LEN * $U_POSSIZE_ARGS_LEN ) % $MODELS_LEN ))
MODEL=${MODELS[$MODELS_IDX]}


# Set the pretrain path, if applicable
if [[ $PRETRAINED == yes ]]; then
    PRETRAINED_ARG="--pretrained "$SCRATCH"/dpc/pretrained/mousesim_left-128_r18_dpc-rnn/model/model_best_epoch715.pth.tar"
    UNEXP_EPOCH=20
else
    PRETRAINED_ARG=""
    UNEXP_EPOCH=30
fi

# Set the sequence parameters
if [[ $SIMPLE == full ]]; then
    SEQ_LEN=$FIXED_GAB_IMG_LEN
    PRED_STEP=1
    NUM_SEQ=4 # gray will never appear
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

if [[ $ROLL_ARG ]]; then
    if [[ $SUFFIX_ARG ]]; then
        SUFFIX_ARG=$SUFFIX_ARG"_roll"
    else
        SUFFIX_ARG="--suffix roll"
    fi
fi


echo -e "\nARRAY ID HYPERPARAMETERS\n"\
"model: $MODEL\n"\
"U position/size argument: $U_POSSIZE_ARG\n"\
"unexpected epoch: $UNEXP_EPOCH\n"\
"sequence length: $SEQ_LEN\n"\
"number of steps to predict: $PRED_STEP\n"\
"number of consecutive sequences per batch item: $NUM_SEQ\n"\
"roll argument: $ROLL_ARG\n"\
"number of epochs: $NUM_EPOCHS\n"\
"suffix argument: $SUFFIX_ARG\n"\
"pretrained argument: $PRETRAINED_ARG\n"\


# 5. Train model
EXIT=0

set -x # echo commands to console

python run_model.py \
    --output_dir $SCRATCH/dpc/models \
    --dataset $FIXED_DATASET \
    --net $FIXED_NET \
    --train_what $FIXED_TRAIN_WHAT \
    --img_dim $FIXED_IMG_DIM \
    $FIXED_AUGM_ARG \
    --batch_size $FIXED_BATCH_SIZE \
    --train_len $FIXED_TRAIN_LEN \
    --gab_img_len $FIXED_GAB_IMG_LEN \
    --U_prob $FIXED_U_PROB \
    --num_workers $FIXED_NUM_WORKERS \
    $SEED_ARG \
    --model $MODEL \
    $U_POSSIZE_ARG \
    --unexp_epoch $UNEXP_EPOCH \
    --seq_len $SEQ_LEN \
    --pred_step $PRED_STEP \
    --num_seq $NUM_SEQ \
    $ROLL_ARG \
    --num_epochs $NUM_EPOCHS \
    $SUFFIX_ARG \
    $PRETRAINED_ARG \

code="$?"
if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console


# 5. Print instructions for testing model
if [[ $MODEL == "lc-rnn" ]]; then
    echo -e "To test the model, run:\n"\
    "python run_model.py"\
    "--model $MODEL"\
    "--dataset $FIXED_DATASET"\
    "--img_dim $FIXED_IMG_DIM"\
    "--train_len $FIXED_TRAIN_LEN"\
    "--gab_img_len $FIXED_GAB_IMG_LEN"\
    "--U_prob $FIXED_U_PROB"\
    "$SEED_ARG"\
    "$U_POSSIZE_ARG"\
    "--seq_len $SEQ_LEN"\
    "--num_seq $NUM_SEQ"\
    "$ROLL_ARG"\
    "--unexp_epoch 0"\
    "--test $SCRATCH/dpc/models/gabors..."
fi
  

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code

