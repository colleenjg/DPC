#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=48GB
#SBATCH --array=0-7
#SBATCH --time=6:00:00
#SBATCH --job-name=dpc
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_%A_%a.out


# Current longest: 4h30

# 1. Load modules
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Array ID hyperparameters
DATASETS=( "MouseSim" "HMDB51" "UCF101" "Kinetics400" ) # 4
PRETRAINEDS=( "no" "yes" ) # 2

# set values for task ID
# inner loop
DATASETS_LEN="${#DATASETS[@]}"
DATASETS_IDX=$(( SLURM_ARRAY_TASK_ID % DATASETS_LEN ))
DATASET=${DATASETS[$DATASETS_IDX]}

# outer loop
PRETRAINEDS_LEN=${#PRETRAINEDS[@]}
PRETRAINEDS_IDX=$(( SLURM_ARRAY_TASK_ID / DATASETS_LEN % PRETRAINEDS_LEN ))
PRETRAINED=${PRETRAINEDS[$PRETRAINEDS_IDX]}


# 3. Set some variables
if [[ $SEED == "" ]]; then
    SEED=100
fi

if [[ $NUM_EPOCHS == "" ]]; then
    NUM_EPOCHS=100
fi

IMG_DIM=128
BATCH_SIZE=64
NUM_SEQ=8
DS_ARG=""
if [[ $DATASET == "Kinetics400" && $IMG_DIM -gt 140 ]]; then
    K400_BIG_ARG="-b" # for copying data
elif [[ $DATASET == "MouseSim" ]]; then
    EYE_ARG="--eye left"
    EYE_COPY_ARG="-e left" # for copying data
    DS_ARG="--ucf_hmdb_ms_ds 3" # from 50 Hz -> 16.7 Hz
    NUM_EPOCHS=1000
    BATCH_SIZE=28
    LR_ARG="--lr 0.0001"
    NUM_SEQ=18
fi

if [[ $PRETRAINED == yes ]]; then
    MODEL="lc-rnn"
    TRAIN_WHAT="ft"
    PRETRAINED_ARG="--pretrained ${SCRATCH}/dpc/pretrained/k400_128_r18_dpc-rnn/model/k400_128_r18_dpc-rnn.pth.tar"
else
    MODEL="dpc-rnn"
    TRAIN_WHAT="all"
    PRETRAINED_ARG=""
fi


# 4. Copy data to $SLURM_TMPDIR
EXIT=0

set -x # echo commands to console
bash slurm/copy_frame_tars.sh -s "${SCRATCH}/dpc/datasets" -t "$SLURM_TMPDIR" -d "$DATASET" $EYE_COPY_ARG $K400_BIG_ARG

code=${?}
if [[ "$code" -gt "$EXIT" ]]; then 
    EXIT="$code"
    echo "Copying data to ${SLURM_TMPDIR} failed."
    exit ${EXIT}
fi 


# 5. Train model

python run_model.py \
    --output_dir "${SCRATCH}/dpc/models" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    $EYE_ARG \
    $DS_ARG \
    --img_dim "$IMG_DIM" \
    --batch_size "$BATCH_SIZE" \
    --num_seq "$NUM_SEQ" \
    --num_epochs "$NUM_EPOCHS" \
    --train_what "$TRAIN_WHAT" \
    $LR_ARG \
    --seed "$SEED" \
    $PRETRAINED_ARG \
    --num_workers 8 \
    --log_test_cmd \
    --temp_data_dir "$SLURM_TMPDIR" \

code=${?}
if [[ "$code" -gt "$EXIT" ]]; then EXIT="$code"; fi # collect exit code

set +x # stop echoing commands to console



if [[ "$EXIT" -ne 0 ]]; then exit "$EXIT"; fi # exit with highest exit code

