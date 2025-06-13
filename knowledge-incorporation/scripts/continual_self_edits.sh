#!/bin/bash
#SBATCH --job-name=continual
#SBATCH --output=logs/%A_continual.log
#SBATCH --gres=gpu:2

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# -------- User-editable ---------------------------------------------- #
INDEX=0  # Index for this job, used to differentiate runs

MODEL_NAME="Qwen/Qwen2.5-7B"   # initialized model. Use the last RL checkpoint
DATASET="knowledge-incorporation/data/squad_val.json"  # evaluation dataset
OUTPUT_DIR="knowledge-incorporation/results/continual_self_edits/run${INDEX}"
mkdir -p "${OUTPUT_DIR}"

# LoRA / tuning hyper-parameters (matches: r α drop ep lr bs ga)
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0
FINETUNE_EPOCHS=10
FINETUNE_LR=1e-3
BATCH_SIZE=1
GRAD_ACC=1

# Infrastructure layout
VLLM_SERVER_GPUS="0"       # GPU(s) for vLLM server (comma-sep)
PY_DRIVER_GPU="1"          # GPU on which the continual self-edit script runs
PORT=$((8001 + INDEX))     # vLLM HTTP port (unique per node)
ZMQ_PORT=$((5555 + INDEX)) # ZMQ port if driver spawns an inner server
SEED=$((42 + INDEX))

MAX_TOKENS=8192            # self-edit generation cap
TEMPERATURE=1.0            # self-edit sampling temperature
top_p=0.95                 # self-edit top-p

N_SEQUENCES=8              # number of sequence to average over
N_DATAPOINTS=8             # datapoints per sequence
# --------------------------------------------------------------------- #

set -a
source .env
set +a

export CUDA_VISIBLE_DEVICES=${PY_DRIVER_GPU},${VLLM_SERVER_GPUS}

# -------- Launch Driver ---------------------------------------------- #
echo "Starting continual self-edits driver on GPU ${PY_DRIVER_GPU}"
python3 -u -m knowledge-incorporation.src.continual.continual_self_edits \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --finetune_epochs ${FINETUNE_EPOCHS} \
    --finetune_lr ${FINETUNE_LR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --n_sequences ${N_SEQUENCES} \
    --n_datapoints ${N_DATAPOINTS} \
    --output_dir "${OUTPUT_DIR}" \
    --gpus "${VLLM_SERVER_GPUS},${PY_DRIVER_GPU}" \
    --vllm_port ${PORT} \
    --zmq_port ${ZMQ_PORT} \
    --temperature ${TEMPERATURE} \
    --top_p ${top_p} \
    --max_tokens ${MAX_TOKENS} \
    --seed ${SEED}

echo "Job finished."
