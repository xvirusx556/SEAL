#!/bin/bash
#SBATCH --job-name=request
#SBATCH --output=logs/%A_mkdt_oai.log
#SBATCH --gres=gpu:0

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# --------------------------------------------------------------------- #
python knowledge-incorporation/src/data_generation/make_squad_data_openai.py \
    --dataset_in knowledge-incorporation/data/squad_val.json \
    --dataset_out knowledge-incorporation/data/synthetic_data/eval/gpt4_1_val.json \
    --n 200

echo "Job finished."
