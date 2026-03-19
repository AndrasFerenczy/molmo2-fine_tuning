#!/bin/bash

WANDB_API_KEY=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 \
torchrun --nproc-per-node=1 /home/agf64/molmo2-fine_tuning/launch_scripts/sft.py \
  /home/agf64/molmo2-fine_tuning/fine-tuning/Molmo2-4B/step2000 videocapqa \
  --save_folder=/home/agf64/molmo2-fine_tuning/results \
  --wandb.name=molmo2-videocapqa-sft --wandb.entity=agf64-cornell-university --wandb.project=molmo2 \
  --num_workers=1 --prefetch_factor=2