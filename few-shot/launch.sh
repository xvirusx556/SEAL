#!/bin/bash
#SBATCH --job-name=launch
#SBATCH --output=logs/%x_%j.log
#SBATCH --gres=gpu:1

# export HOME=<your_home_directory>
source ~/.bashrc

cd ~/few-shot
conda activate seal_env

python eval-self-edits-baseline.py \
  --experiment_folder="~/tti/eval_base_model" \
  --pretrained_checkpoint=meta-llama/Llama-3.2-1B-Instruct \
  --lora_checkpoints_folder="~/few-shot/loras/self-edit/eval_RL_iteration_1_8_epoch" \
  --temperature=0 \
  --n_sample=1 \
  --data_file="~/few-shot/data/arc-agi_evaluation_challenges_filtered_1B_eval_set.json" \
  --solution_file="~/few-shot/data/arc-agi_evaluation_solutions_filtered_1B_eval_set.json" \
  --max_lora_rank=128 \
  --include_n=1 \
  --new_format \
  --num_examples=9 \
