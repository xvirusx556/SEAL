#!/bin/bash
#SBATCH --job-name=query
#SBATCH --output=logs/%A_query_server.log
#SBATCH --gres=gpu:0

# -------- Environment ------------------------------------------------ #
# export HOME=<your_home_directory>
source ~/.bashrc
conda activate seal_env
cd ~/SEAL

# -------- Static Config ---------------------------------------------- #
# SERVER_HOST="<TTT server IP>"  # set to TTT server IP
ZMQ_PORT=5555

OUTPUT_DIR="knowledge-incorporation/results/query_server"
mkdir -p "${OUTPUT_DIR}"

# -------- Experiment Grid -------------------------------------------- #
# Columns: exp_name dataset  k  evalT  r  α  drop  ep  lr  bs  ga  n_articles
EXPERIMENTS=(
    "rank_iter0 knowledge-incorporation/data/synthetic_data/train/iter0_train.json  5  3  32  64  0  10  1e-3  1  1 50"
    # "eval_baseline data/synthetic_data/eval/base_val.json  1  1  32  64  0  10  1e-3  1  1 200"
)

SPLIT_NEWLINES=true  # whether to split newlines into separate training documents

# -------- Loop & Launch ---------------------------------------------- #
for EXP in "${EXPERIMENTS[@]}"; do
    read -r EXP_NAME DATASET K_COMPLETIONS EVAL_TIMES \
            LORA_RANK LORA_ALPHA LORA_DROPOUT \
            FINETUNE_EPOCHS FINETUNE_LR \
            BATCH_SIZE GRAD_ACC N_ARTICLES <<< "${EXP}"

    TAG=$(basename "${DATASET%.json}")_k${K_COMPLETIONS}_$((RANDOM))
    LOG_FILE="logs/${SLURM_JOB_ID}_query_${TAG}.log"

    echo "Query-server run: ${TAG}"
    python3 -u -m knowledge-incorporation.src.query.query_server \
        --exp_name ${EXP_NAME} \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        --server_host "${SERVER_HOST}" \
        --zmq_port "${ZMQ_PORT}" \
        --n_articles "${N_ARTICLES}" \
        --k_completions "${K_COMPLETIONS}" \
        --eval_times "${EVAL_TIMES}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --finetune_epochs "${FINETUNE_EPOCHS}" \
        --finetune_lr "${FINETUNE_LR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACC}" \
        ${SPLIT_NEWLINES:+--split_newlines} \
        >> "${LOG_FILE}" 2>&1
done

echo "Job finished."
