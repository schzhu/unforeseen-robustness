#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G --time=48:00:00

DATASET_TAR=$1
DATASET_SRC=$2
TRANSFORM=$3
CONFIG=$4
LOG_ID=$5

srun python -u main.py \
  --dataset_tar "${DATASET_TAR}" \
  --dataset_src "${DATASET_SRC}" \
  --transform "${TRANSFORM}" \
  --config "${CONFIG}" \
  --log_id "${LOG_ID}" \

wait
