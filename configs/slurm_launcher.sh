#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=8 --mem-per-cpu=4G --time=48:00:00

DATASET_TAR=$1
DATASET_SRC=$2
TRANSFORM=$3
CONFIG=$4
W_SRC=$5
W_XI=$6
TRANSLATOR_NAME="${7:-none}"

srun python -u main.py \
  --dataset_tar "${DATASET_TAR}" \
  --dataset_src "${DATASET_SRC}" \
  --transform "${TRANSFORM}" \
  --config "${CONFIG}" \
  --w_src "${W_SRC}" \
  --w_xi "${W_XI}" \
  --translator_name "${TRANSLATOR_NAME}" \

wait
