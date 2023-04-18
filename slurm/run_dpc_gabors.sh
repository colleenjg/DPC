#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48GB
#SBATCH --array=0-5
#SBATCH --time=8:00:00
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
FIXED_PRED_STEP=1
FIXED_U_PROB=0.08
FIXED_U_POSSIZE_ARG="--diff_U_possizes"
FIXED_GAB_ANALYSIS_ARG="--analysis"
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
"number of steps to predict: $FIXED_PRED_STEP\n" \
"U probability: $FIXED_U_PROB\n" \
"U position/size argument: $FIXED_U_POSSIZE_ARG\n" \
"Gabor analysis argument: $FIXED_GAB_ANALYSIS_ARG\n" \
"number of workers: $FIXED_NUM_WORKERS\n" \


# 3. Externally set parameters
if [[ "$SEED" == "" ]]; then
    SEED=100
fi

# 4. Array ID hyperparameters
ROLL_ARGS=( "--roll" "" ) # 2
PRETRAINEDS=( "MouseSim" "Kinetics400" "no" ) # 3

# set values for task ID
# inner loop
ROLL_ARGS_LEN=${#ROLL_ARGS[@]}
ROLL_ARGS_IDX=$(( SLURM_ARRAY_TASK_ID % ROLL_ARGS_LEN ))
ROLL_ARG=${ROLL_ARGS[$ROLL_ARGS_IDX]}

# outer loop
PRETRAINEDS_LEN=${#PRETRAINEDS[@]}
PRETRAINEDS_IDX=$(( SLURM_ARRAY_TASK_ID / ROLL_ARGS_LEN % PRETRAINEDS_LEN ))
PRETRAINED=${PRETRAINEDS[$PRETRAINEDS_IDX]}


# Set the sequence parameters
if [[ "$ROLL_ARG" == "" ]]; then
    NUM_SEQ_IN=3 # predict D/U only (gray will never appear)
    TRAIN_LEN=1024
else
    NUM_SEQ_IN=4
    TRAIN_LEN=4096
fi

if [[ "$PRETRAINED" == "MouseSim" ]]; then
    PRETRAINED_ARG="set in loop"
elif [[ "$PRETRAINED" == "Kinetics400" ]]; then
    PRETRAINED_ARG="--pretrained ${SCRATCH}/dpc/pretrained/k400_128_r18_dpc-rnn/model/k400_128_r18_dpc-rnn.pth.tar"
else
    PRETRAINED_ARG=""
fi

if [[ "$PRETRAINED_ARG" == "" ]]; then
    UNEXP_EPOCH=12
else
    UNEXP_EPOCH=6
fi

if [[ "$ROLL_ARG" ]]; then
    if [[ $SUFFIX_ARG ]]; then
        SUFFIX_ARG="${SUFFIX_ARG}_roll"
    else
        SUFFIX_ARG="--suffix roll"
    fi
fi

NUM_EPOCHS=$(( UNEXP_EPOCH + 4 ))


echo -e "\nARRAY ID HYPERPARAMETERS\n" \
"roll argument: $ROLL_ARG\n" \
"number of consecutive blocks to use as input: $NUM_SEQ_IN\n" \
"training dataset length: $TRAIN_LEN\n" \
"unexpected epoch: $UNEXP_EPOCH\n" \
"number of epochs: $NUM_EPOCHS\n" \
"suffix argument: $SUFFIX_ARG\n" \
"pretrained argument: $PRETRAINED_ARG\n" \

# 5. Pretrained paths
MOUSESIM_SEEDS=(  100 101 102 103 ) # 4
MOUSESIM_EPOCHS=( 770 761 950 891 ) # 4

# 6. Train model
EXIT=0

for i in {0..3}; do

    LOOP_SEED=$(( SEED + i )) 

    echo -e "LOOP HYPERPARAMETERS\n" \
    "loop argument: $LOOP_SEED\n" \

    # Set the pretrain path, if applicable
    if [[ "$PRETRAINED" == "MouseSim" ]]; then
        MOUSESIM_SEED=${MOUSESIM_SEEDS[$i]}
        MOUSESIM_EPOCH=${MOUSESIM_EPOCHS[$i]}
        PRETRAINED_ARG="--pretrained ${SCRATCH}/dpc/pretrained/mousesim_right-128_r18_dpc-rnn_seed${MOUSESIM_SEED}/model/mousesim_right_best_epoch${MOUSESIM_EPOCH}.pth.tar"
        echo -e "pretrained argument: $PRETRAINED_ARG\n"
    fi

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
        --pred_step "$FIXED_PRED_STEP" \
        --U_prob "$FIXED_U_PROB" \
        $FIXED_U_POSSIZE_ARG \
        $FIXED_GAB_ANALYSIS_ARG \
        --num_workers "$FIXED_NUM_WORKERS" \
        $ROLL_ARG \
        --num_seq_in "$NUM_SEQ_IN" \
        --train_len "$TRAIN_LEN" \
        --unexp_epoch "$UNEXP_EPOCH" \
        --num_epochs "$NUM_EPOCHS" \
        --seed "$LOOP_SEED" \
        $SUFFIX_ARG \
        $PRETRAINED_ARG \


    code="$?"
    if [[ "$code" -gt "$EXIT" ]]; then EXIT="$code"; fi # collect exit code

    set +x # stop echoing commands to console

done


if [[ "$EXIT" -ne 0 ]]; then exit "$EXIT"; fi # exit with highest exit code

