#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --gres=cn:12GB:2
#SBATCH --mem=12GB
#SBATCH --array=0
#SBATCH --time=4:00:00
#SBATCH --job-name=dpc_gab
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_gabors_%j.txt


# 1. Load modules
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Set the pretrain path, if applicable
if [[ $PRETRAINED == 1 ]]; then
    MODEL=lc-rnn
    TRAIN_WHAT=ft
    PRETRAINED="--pretrained "$SCRATCH"/dpc/pretrained/k400_128_r18_dpc-rnn/model/k400_128_r18_dpc-rnn.pth.tar"
else
    MODEL=dpc-rnn
    TRAIN_WHAT=all
    PRETRAINED=""
fi


# 3. Set fixed hyperparameters
SE=5
N_SEQ=4
U_PROB=0.1
SEED=2


# 4. Identify seeds to run through
BASE_SEED=1
if [[ -n "$N_PER" ]]; then
    N_PER=6 # default value
fi

START_SEED=$((SLURM_ARRAY_TASK_ID * N_PER + BASE_SEED))
STOP_SEED=$((START_SEED + N_PER - 1)) # inclusive
SEEDS=($( seq $START_SEED 1 $STOP_SEED ))


# 5. Results are saved under $SCRATCH/dpc
EXIT=0

for SEED in "${SEEDS[@]}"
do
    set -x # echo commands to console

    python train_model.py \
        --output_dir $SCRATCH/dpc \
        --net resnet18 \
        --dataset gabors \
        --img_dim 128 \
        --batch_size 10 \
        --num_epochs 25 \
        --log_freq 1 \
        --pred_step 1 \
        --unexp_epoch 5 \
        --roll \
        --num_workers 8 \
        --model $MODEL \
        --train_what $TRAIN_WHAT \
        --seq_len $SE \
        --num_seq $N_SEQ \
        --U_prob $U_PROB \
        --seed $SEED \
        $PRETRAINED \

    code="$?"
    if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

    set +x # stop echoing commands to console

    # 6. Check if test is being run
    if [[ $TEST == 1 ]]; then # only run 1 iteration
        break
    fi
done

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code

