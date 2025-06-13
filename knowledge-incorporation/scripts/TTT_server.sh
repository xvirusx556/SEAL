#!/bin/bash
#SBATCH --job-name=server
#SBATCH --output=logs/%A_server.log
#SBATCH --gres=gpu:2

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# -------- User-editable ---------------------------------------------- #
MODEL_NAME="Qwen/Qwen2.5-7B"  # HF model name or path to RL checkpoint (e.g. models/iter1)
VLLM_SERVER_GPUS="0"
INNER_LOOP_GPU="1"
PORT=8001
ZMQ_PORT=5555

MAX_SEQ_LENGTH=2048  # Max sequence length for training
EVAL_MAX_TOKENS=64   # Max generated tokens for evaluation completions
EVAL_TEMPERATURE=0.0
EVAL_TOP_P=1.0

MAX_LORA_RANK=32     # Max LoRA rank that will be used
# --------------------------------------------------------------------- #
echo "Launching TTT server on $(hostname)..."

set -a
source .env
set +a

VLLM_HOST=$(hostname -i)
VLLM_API_URL="http://${VLLM_HOST}:${PORT}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

echo "Launching vLLM on GPUs ${VLLM_SERVER_GPUS}"
CUDA_VISIBLE_DEVICES=${VLLM_SERVER_GPUS} vllm serve "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port ${PORT} \
    --max-model-len ${MAX_SEQ_LENGTH} \
    --enable-lora \
    --max-lora-rank ${MAX_LORA_RANK} \
    --trust-remote-code \
    > "logs/${SLURM_JOB_ID}_vllm_server.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM..."
until curl --silent --fail ${VLLM_API_URL}/health >/dev/null; do sleep 3; done
echo "    vLLM ready at ${VLLM_API_URL}"

echo "Starting Inner Loop server on GPU ${INNER_LOOP_GPU}..."
CUDA_VISIBLE_DEVICES=${INNER_LOOP_GPU} python3 -m knowledge-incorporation.src.inner.TTT_server \
    --vllm_api_url "${VLLM_API_URL}" \
    --model "${MODEL_NAME}" \
    --zmq_port ${ZMQ_PORT} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --eval_max_tokens ${EVAL_MAX_TOKENS} \
    --eval_temperature ${EVAL_TEMPERATURE} \
    --eval_top_p ${EVAL_TOP_P} \
    > logs/${SLURM_JOB_ID}_TTT_server.log 2>&1 &

ZMQ_PID=$!
echo "    Inner Loop Server started with PID ${ZMQ_PID}."
echo "Ready to accept requests on port ${ZMQ_PORT}."

trap "echo 'Shutting down...'; kill ${ZMQ_PID} ${VLLM_PID}" EXIT
wait

echo "Job finished."
