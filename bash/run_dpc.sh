#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0,1
#SBATCH --mem=10G
#SBATCH --array=0-4
#SBATCH --time=4:00:00
#SBATCH --output=/home/mila/g/gillonco/scratch/dpc_%j.txt


# 1. Load modules
module purge
module load anaconda/3
module load cuda/10.2/cudnn/7.6

conda activate ssl


# 2. Optionally, copy pre-trained model weights (downloaded from DPC repo) 
# from SCRATCH to SLURM_TMPDIR 
if [[ $PRETRAIN == 1 ]]; then
    echo "Copying pre-trained model to slurm temporary directory."
    PRETRAIN_STR="pretrained_"
    PRETRAIN="--pretrain $SLURM_TMPDIR/k400_128_r18_dpc-rnn.pth.tar"
    cp $SCRATCH/dpc/pretrained/k400_128_r18_dpc-rnn.pth.tar $SLURM_TMPDIR/

    code="$?"
    if [ "$code" -ne 0 ]; then exit "$code"; fi # exit, if failed
fi


# 3. Set fixed hyperparameters
SE=5
N_SEQ=4
U_PROB=0.1
SEED=2


# 4. Identify seeds to run through
BASE_SEED = 1
if [[ -n "$N_PER" ]]; then
    N_PER=6 # default value
fi

START_SEED=$((SLURM_ARRAY_TASK_ID * N_PER + BASE_SEED))
STOP_SEED=$((START_SEED + N_PER - 1)) # inclusive
SEEDS=($( seq $START_SEED 1 $STOP_SEED ))


# 5. Results are saved in $SLURM_TMPDIR
EXIT=0

for SEED in "${SEEDS[@]}"
do
    set -x # echo commands to console

    python train_model.py \
        --output_dir $SLURM_TMPDIR \
        $PRETRAIN \
        --net resnet18 \
        --model "dpc-rnn" \
        --dataset gabors \
        --img_dim 128 \
        --seq_len $SE \
        --num_seq $N_SEQ \
        --train_what ft \
        --batch_size 10 \
        --num_epochs 25 \
        --log_freq 1 \
        --seed $SEED \
        --pred_step 1 \
        --unexp_epoch 5 \
        --roll \
        --U_prob $U_PROB \

    code="$?"
    if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi # collect exit code

    set +x # stop echoing commands to console

    # 6. Copy results and model to scratch
    cd $SLURM_TMPDIR

    LOCAL_DIR=$SCRATCH/test_gabors/$PRETRAIN_STR"noblanks_numseq"$N_SEQ"_Elast_bothED_batch10"
    mkdir -p $LOCAL_DIR
    for EXT in ".pth.tar" "*.json" "*.svg" "*.md" "tensorboard"
    do
        find . -type f -wholename $EXT | cpio -pdm $LOCAL_DIR
    done

    code="$?"
    if [ "$code" -gt "$EXIT" ]; then EXIT="$code"; fi

    # 7. Check if test run
    if [[ $TEST == 1 ]]; then # only run 1 iteration
        break
    fi
done

if [ "$EXIT" -ne 0 ]; then exit "$EXIT"; fi # exit with highest exit code

