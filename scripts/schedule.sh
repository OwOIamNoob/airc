#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=610e7d7ec08264ac19d257213565223003941861
python src/train.py trainer.max_epochs=30 logger=wandb

