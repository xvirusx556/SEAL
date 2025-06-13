# knowledge-incorporation/src/inner/TTT_server.py
"""
Inner-loop Test-Time Training (TTT) server used by SEAL's outer-loop drivers
(`query_server.py`, `CPT.py`, `continual_self_edits.py`) to rapidly fine-tune 
a temporary LoRA adapter on a handful of synthetic sequences and immediately 
evaluate it on corresponding SQuAD questions, without the sequences in context.

The server is stateless across requests: every JSON message describes a complete round consisting of
1. a mini-dataset of train_sequences (for LoRA fine-tuning),
2. a list of eval_questions (for accuracy measurement), and
3. hyper-parameters controlling both steps.

It then replies with baseline-vs-adapter accuracies, generated answers, and per-question booleans indicating correctness.

JSON schema
    Request -->
    {
        "train_sequences": [str],
        "eval_questions":  [{title, context, question, answer}],
        "lora_rank": int,
        ...
    }
    Response <--
    {
        "baseline_accuracy": float,
        "adapter_accuracy":  float,
        "adapter_gain":      float,
        ...
    }
"""
import argparse, gc, logging, os, shutil, time
from pathlib import Path
from typing import Dict, List, Any
import torch
import zmq
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import random, numpy as np, torch, time as _time
from ..utils import (
    set_vllm_api_url,
    load_adapter,
    unload_adapter,
    generate,
    format_answer_prompts,
    format_grade_prompts,
    grade_with_gpt4,
)

# ---------------------------  CONFIG & LOGGING  ----------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger()


def accuracy_and_texts(
    questions: List[Dict[str, str]],
    answer_model_ref: str,
    sampling: Dict[str, Any],
    stop_ids: List[int],
    instruct_model: bool,
) -> tuple[float, List[str], List[bool]]:
    ans_out = generate(
        format_answer_prompts(questions, instruct_model=instruct_model), answer_model_ref, sampling, stop_ids
    ) or []
    preds = [o.get("text", "") for o in ans_out]
    LOG.debug("Formatted answer prompts:", format_answer_prompts(questions, instruct_model=instruct_model))
    LOG.debug("answer_model_ref:", answer_model_ref)
    LOG.debug("sampling:", sampling)
    LOG.debug("stop_ids:", stop_ids)
    LOG.debug("preds:", preds)

    verdicts: List[bool] = [False] * len(preds)
    q_sub, p_sub, idx_sub = [], [], []

    for i, (q, p) in enumerate(zip(questions, preds)):
        if p.strip():
            q_sub.append(q)
            p_sub.append(p)
            idx_sub.append(i)

    if q_sub:
        grade_prompts = format_grade_prompts(q_sub, p_sub)
        graded = grade_with_gpt4(grade_prompts)
        for i, v in zip(idx_sub, graded):
            verdicts[i] = v
    LOG.debug("verdicts:", verdicts)
    acc = sum(verdicts) / len(questions) if questions else 0.0
    return acc, preds, verdicts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zmq_port", type=int, default=5555, help="ZMQ port to listen on")
    p.add_argument("--vllm_api_url", required=True, help="e.g. http://localhost:8001")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model name")
    p.add_argument("--instruct_model", action="store_true", help="Using Qwen Instruct model")
    p.add_argument("--max_seq_length", type=int, default=2048, help="Max training seq len")
    p.add_argument("--eval_temperature", type=float, default=0.0, help="Eval sampling temperature")
    p.add_argument("--eval_top_p", type=float, default=1.0, help="Eval nucleus sampling (top-p)")
    p.add_argument("--eval_max_tokens", type=int, default=64, help="Eval max tokens to generate")
    p.add_argument("--keep_adapter_dir",  action="store_true",
                   help="Skip tmp-dir deletion so outer driver can reuse the LoRA. This causes high disk usage and is only used in continual_self_edits.py or for debugging.")
    args = p.parse_args()

    # initialize vLLM API
    set_vllm_api_url(args.vllm_api_url)

    LOG.info("Loading base model %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.instruct_model:
        stop_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    else:
        stop_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

    # ---------- ZMQ REP socket ---------------------------------------- #
    ctx, sock = zmq.Context(), None
    try:
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{args.zmq_port}")
        LOG.info("ZMQ listening at tcp://*:%d", args.zmq_port)
        step = 0
        while True:
            LOG.info("Waiting for request...")
            msg = sock.recv_json()
            LOG.info("Received request: %s", msg)

            if msg.get("cmd") == "shutdown":
                sock.send_json({"status": "bye"})   # reply
                break                               # exit the while-loop

            recv_start = time.time()
            try:
                LOG.debug("RX %d %s", step, msg.keys())
                seed = (int(_time.time() * 1e6) + step) & 0xFFFFFFFF
                random.seed(seed); np.random.seed(seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                LOG.info("Step %d  using seed %d", step, seed)

                train_sequences   = msg.get("train_sequences")
                questions         = msg.get("eval_questions", [])
                lora_rank         = msg.get("lora_rank", 32)
                lora_alpha        = msg.get("lora_alpha", 64)
                lora_dropout      = msg.get("lora_dropout", 0)
                finetune_epochs   = msg.get("finetune_epochs", 10)
                finetune_lr       = msg.get("finetune_lr", 1e-3)
                batch_size        = msg.get("batch_size", 1)
                gradient_accumulation_steps = msg.get("gradient_accumulation_steps", 1)
                end_mask_substring = msg.get("end_mask_substring")
                skip_training     = bool(msg.get("skip_training", False))

                sampling_cfg = {
                    "n": 1,
                    "temperature": args.eval_temperature,
                    "top_p": args.eval_top_p,
                    "max_tokens": args.eval_max_tokens,
                }

                # ---------- baseline ------------------------------------------------ #
                base_acc, base_texts, base_ok = accuracy_and_texts(
                    questions,
                    answer_model_ref=args.model,
                    sampling=sampling_cfg,
                    stop_ids=stop_ids,
                    instruct_model=args.instruct_model,
                )

                if skip_training or not train_sequences:
                    reply = {
                        "baseline_accuracy": round(base_acc, 4),
                        "adapter_accuracy" : round(base_acc, 4),
                        "adapter_gain"     : 0.0,
                        "baseline_texts"   : base_texts,
                        "adapter_texts"    : base_texts,
                        "baseline_correct" : base_ok,
                        "adapter_correct"  : base_ok,
                        "gains"            : [0]*len(base_ok),
                    }
                    sock.send_json(reply)
                    LOG.info("Step %d  BASE-ONLY  acc %.3f  (%.2fs)",
                            step, base_acc, time.time()-recv_start)
                    step += 1
                    continue

                # ---------- prepare LoRA fine-tune dataset -------------------------- #
                tmp_tag = f"inner_TTT_{step}"
                tmp_dir = Path(f"models/tmp_{args.zmq_port}_{tmp_tag}")
                os.makedirs(tmp_dir, exist_ok=True)

                rows = []
                sub_ids = (
                    tokenizer.encode(end_mask_substring, add_special_tokens=False)
                    if end_mask_substring else []
                )

                for idx, seq in enumerate(train_sequences):
                    tok = tokenizer(
                        seq,
                        truncation=True,
                        max_length=args.max_seq_length,
                        padding="max_length",
                    )
                    labels = tok["input_ids"].copy()
                    if sub_ids:
                        M = len(sub_ids)
                        for i in range(len(labels) - M + 1):
                            if labels[i : i + M] == sub_ids:
                                for j in range(i + M):
                                    labels[j] = -100
                                # ---------- DEBUG LOG (first 5 only) ---------------
                                if idx < 5:
                                    # insert a visual marker after the masked span
                                    marker_pos = tokenizer.decode(tok["input_ids"][: i + M])
                                    debug_str  = seq.replace(
                                        marker_pos,
                                        marker_pos + "<<<MASK_END>>>",
                                        1
                                    )
                                    LOG.info("TRAIN[%d] %s", idx, debug_str)
                                # ---------------------------------------------------
                                break
                    if idx < 3 and not sub_ids:          # no masking substring given
                        LOG.info("TRAIN[%d] %s", idx, seq)

                    rows.append(
                        {
                            "input_ids": tok["input_ids"],
                            "attention_mask": tok["attention_mask"],
                            "labels": labels,
                        }
                    )

                ds = HFDataset.from_list(rows)
                collator = DataCollatorWithPadding(tokenizer)

                lora_cfg = LoraConfig(
                    r=lora_rank, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout, task_type="CAUSAL_LM"
                )
                lora_model = get_peft_model(base_model, lora_cfg)

                trainer = Trainer(
                    model=lora_model,
                    args=TrainingArguments(
                        output_dir=str(tmp_dir),
                        per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        num_train_epochs=finetune_epochs,
                        learning_rate=finetune_lr,
                        logging_steps=1, save_strategy="no", report_to="none",
                        remove_unused_columns=False, fp16=False,
                        bf16=torch.cuda.is_available()
                        and torch.cuda.is_bf16_supported(),
                        seed=seed,
                    ),
                    train_dataset=ds,
                    data_collator=collator,
                )
                trainer.train()
                adapter_path = tmp_dir / "final_adapter"
                lora_model.save_pretrained(str(adapter_path))

                # ---------- evaluation with adapter ------------------------------- #
                adapter_name = tmp_tag
                load_adapter(str(adapter_path), adapter_name)

                adapter_acc, adapter_texts, adapter_ok = accuracy_and_texts(
                    questions,
                    answer_model_ref=adapter_name,
                    sampling=sampling_cfg,
                    stop_ids=stop_ids,
                    instruct_model=args.instruct_model,
                )

                gains = [
                    1  if a and not b else
                    -1 if b and not a else
                    0
                    for b, a in zip(base_ok, adapter_ok)
                ]

                unload_adapter(adapter_name)
                if not args.keep_adapter_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                gc.collect();  torch.cuda.empty_cache()

                reply = {
                    "baseline_accuracy": round(base_acc, 4),
                    "adapter_accuracy": round(adapter_acc, 4),
                    "adapter_gain": round(adapter_acc - base_acc, 4),
                    "baseline_texts": base_texts,
                    "adapter_texts": adapter_texts,
                    "baseline_correct": base_ok,
                    "adapter_correct": adapter_ok,
                    "gains": gains,
                }
                LOG.info(
                    "Step %d  base %.3f  adapter %.3f  Δ %.3f  (%.2fs)",
                    step,
                    base_acc,
                    adapter_acc,
                    adapter_acc - base_acc,
                    time.time() - recv_start,
                )
            except Exception as e:
                LOG.exception("Error processing request.")
                reply = {"error": f"{type(e).__name__}: {e}"}
            finally:
                LOG.info("Sending reply...")
                sock.send_json(reply)
                LOG.info("Reply sent, step %d complete.", step)
                step += 1
    finally:
        if sock:
            sock.close()
        ctx.term()

if __name__ == "__main__":
    main()
