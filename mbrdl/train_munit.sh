#!/bin/bash

# sbatch --gres=gpu:1 --qos=high --cpus-per-task=16 --mem=128G --time=36:00:00 --account=furongh train_munit.sh
# sbatch --partition=scavenger --qos=scavenger --gres=gpu:1  --cpus-per-task=16 --mem=128G --time=48:00:00 --account=scavenger train_munit.sh

# This file can be used to train models of natural variation for the
# SVHN, GTSRB, and CURE-TSR datasets.

# Variables needed to select dataset and source of natural variation
export DATASET='stl10'
export DATA_DIR='work_dirs/datasets'
export SOURCE='rotate'
export SZ=32

# Path to MUNIT configuration file.  Edit this file to change the number of iterations, 
# how frequently checkpoints are saved, and other properties of MUNIT.
# The parameter `style_dim` corresponds to the dimension of `delta` in our work.
export CONFIG_PATH=core/models/munit/munit.yaml

# Output images and checkpoints will be saved to this path.
export OUTPUT_PATH='work_dirs/mbrdl_output/stl10_rotate'

python3 core/train_munit.py \
    --config $CONFIG_PATH --dataset $DATASET --source-of-nat-var $SOURCE \
    --output_path $OUTPUT_PATH --data-size $SZ --train-data-dir $DATA_DIR
    