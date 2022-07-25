#!/bin/bash

EXIT_SRC=110
EXIT_EYE=210

while getopts s:t:d:e:b opt; do
    case "$opt" in
        s) SRC_DIR=$OPTARG;;
        t) TARG_DIR=$OPTARG;;
        d) DATASET=$OPTARG;;
        e) EYE=$OPTARG;;
        b) K400_BIG=1;;
        *) echo "usage: $0 [-s, -t, -d, -e, -b]" >&2 # direct to stderr
       exit 1 ;;
    esac
done


# 1. Identify the source and target directories.

FRAMES_TARG_DIR="frames"
if [[ $DATASET == "MouseSim" ]]; then
    if [[ $EYE == "left" || $EYE == "right" ]]; then
        FRAMES_TARG_DIR="frames_$EYE"
    elif [[ $EYE != "both" ]]; then
        echo "$EYE must be 'left', 'right' or 'both', not ${EYE}."
        exit $EXIT_EYE
    fi
elif [[ $DATASET == "Kinetics400" && $K400_BIG == 1 ]]; then
    FRAMES_TARG_DIR="frames_256"
fi
FRAMES_SRC_DIR="${FRAMES_TARG_DIR}_tar"

FULL_SRC_DIR="${SRC_DIR}/${DATASET}/${FRAMES_SRC_DIR}"
FULL_TARG_DIR="${TARG_DIR}/${DATASET}/${FRAMES_TARG_DIR}"

if [[ ! -d $SRC_DIR ]]; then
    echo "Inferred full source directory does not exist: ${FULL_SRC_DIR}."
    exit $EXIT_SRC
fi

mkdir -p "$FULL_TARG_DIR"


# 2. Copy data.

echo "Copying data from ${FULL_SRC_DIR} to ${FULL_TARG_DIR}."

PIDS=()
cd "$FULL_SRC_DIR" || exit
for dir in *; do
    cp -r "$dir" "${FULL_TARG_DIR}/${dir}" & PIDS+=($!)
done

EXIT=0
for p in "${PIDS[@]}"; do
    wait "$p"
    code=$?
    if [[ "$code" -gt "$EXIT" ]]; then 
        echo "Copying failed."
        EXIT="$code"; 
    fi
done

if [[ "$EXIT" -ne 0 ]]; then exit "$EXIT"; fi # exit with highest exit code


# 3. Untar data.

echo "Untarring data in $FULL_TARG_DIR."

PIDS=()
cd "${FULL_TARG_DIR}" || exit
for dir in *; do
    tar -xf "$dir" & PIDS+=($!)
done


EXIT=0
for p in "${PIDS[@]}"; do
    wait "$p"
    code=$?
    if [[ "$code" -gt "$EXIT" ]]; then 
        echo "Copying failed."
        EXIT="$code"; 
    fi
done

if [[ "$EXIT" -ne 0 ]]; then exit "$EXIT"; fi # exit with highest exit code

# 4. End

echo "Successfully copied and untarred data."

