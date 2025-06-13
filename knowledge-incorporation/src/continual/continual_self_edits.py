# knowledge-incorporation/src/continual/continual_self_edits.py
"""
Incremental self-editing experiment with progressive LoRA merges

This script builds a lower-triangular accuracy matrix for K datapoints
d_0 ... d_{K-1} drawn from a SQuAD-style dataset, repeating the entire
process S times to obtain good estimates of mean and standard-deviation

* Outer loop (S sequences): each repeat draws its own subsequence of
  --n_datapoints items (without replacement, but the full dataset may
  be reused across sequences).
* Inner loop (K steps): at step k we
  1. self-edit on datapoint k (generate implications),
  2. finetune one LoRA adapter on that completion plus the raw
     passage (split_newlines=True),
  3. merge the adapter into the base weights, creating a new base
     for all subsequent steps,
  4. evaluate the freshly merged model on the questions from datapoints
     d_0 ... d_k and collect accuracies.

The final output for each sequence is two (K+1) x K top-row + lower-triangular matrices:
  - A mean-accuracy matrix
  - A standard-deviation matrix

Row 0 corresponds to the not-yet-finetuned base model evaluated on all K
datapoints. Rows 1 through K correspond to evaluations after each
self-edit-and-merge step, with row r containing results on datapoints
d_0 through d_{r-1}.

All artifacts are dumped to --output_dir.
"""
import argparse
import json
import os
import random
import statistics as _stats
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple
import requests
import torch
import zmq
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import (
    set_vllm_api_url,
    build_train_sequences,
)
from ..data_generation.make_squad_data import make_prompt

# Silence transformers warning spam inside forked processes
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

###############################################################################
#                               Infra helpers                                 #
###############################################################################

def _banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80 + "\n", flush=True)

def _spawn_vllm(model: str, host: str, port: int, gpus: str, log_dir: Path, tag: str, lora_rank, max_model_len: int) -> subprocess.Popen:
    """Launch vLLM (LoRA-enabled) and wait for /health."""
    cmd = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--enable-lora",
        "--max-lora-rank",
        str(lora_rank),
        "--trust-remote-code",
    ]
    _banner(f"[vLLM] launching on GPU(s) {gpus} → :{port}\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm_{tag}.log"
    proc = subprocess.Popen(cmd, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)

    health = f"http://{host}:{port}/health"
    for _ in range(600):
        if proc.poll() is not None:
            sys.exit(f"[vLLM] crashed (exit {proc.returncode})")
        try:
            if requests.get(health, timeout=1).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    sys.exit("[vLLM] failed to start within timeout")


def _spawn_inner_server(vllm_api: str, model: str, zmq_port: int, gpu: str, log_dir: Path, tag: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "knowledge-incorporation.src.inner.TTT_server",
        "--vllm_api_url",
        vllm_api,
        "--model",
        model,
        "--zmq_port",
        str(zmq_port),
        "--keep_adapter_dir",  # keep the adapter dir for merging later. It will be removed after merging
    ]
    _banner(f"[Inner] launching on GPU {gpu}, ZMQ :{zmq_port}\n$ {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = log_dir / f"inner_{tag}.log"
    proc = subprocess.Popen(cmd, env=env,
                            stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    return proc


def _connect_zmq(port: int):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{port}")
    return ctx, sock


def _send_round(sock, train_seqs: List[str], questions: List[Dict[str, str]], args):
    sock.send_json(
        {
            "train_sequences": train_seqs,
            "eval_questions": questions,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "end_mask_substring": args.end_mask_substring,
        }
    )
    return sock.recv_json()

###############################################################################
#                             LoRA → merge                                    #
###############################################################################

def _merge_lora(base_path: str, adapter_path: Path, out_dir: Path) -> str:
    """Merge adapter_path into base_path and save to out_dir (returns str)."""
    _banner(f"[Merge] base={base_path} + adapter={adapter_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_path).save_pretrained(out_dir)
    torch.cuda.empty_cache()
    print(f"[Merge] saved → {out_dir}\n")
    return str(out_dir)

###############################################################################
#                          One sequence (outer loop)                          #
###############################################################################

def run_one_sequence(seq_idx: int, items: List[Dict[str, Any]], args) -> Tuple[List[List[float]], List[List[float]]]:
    """Run the incremental loop over items and return mean/std matrices."""
    K = len(items)
    R = K + 1
    mat_vals: List[List[List[float]]] = [[[] for _ in range(K)] for _ in range(R)]

    current_model_path = args.model  # evolves after each merge

    # -------- 0) Base-model row  (row 0 in mat_vals) ------------------
    base_tag      = f"seq{seq_idx}_base"
    logs_step_dir = Path(args.output_dir) / "logs"

    vllm = _spawn_vllm(current_model_path, "127.0.0.1", args.vllm_port, 
                       args.vllm_gpus, logs_step_dir, base_tag, args.lora_rank, 2048)
    vllm_api = f"http://127.0.0.1:{args.vllm_port}"
    set_vllm_api_url(vllm_api)

    inner  = _spawn_inner_server(vllm_api, current_model_path,
                                 args.zmq_port, args.inner_gpu,
                                 logs_step_dir, base_tag)
    ctx, sock = _connect_zmq(args.zmq_port)

    for i, item in enumerate(items):
        eval_q = [
            {
                "title":    item["title"],
                "context":  item["context"],
                "question": f"Topic: {item['title']}\n{q['question']}",
                "answer":   q["answer"],
            }
            for q in item["questions"]
        ]

        # send with empty train_sequences → no fine-tuning
        rep_out  = _send_round(sock, [], eval_q, args)
        correct  = rep_out["adapter_correct"]
        acc      = sum(correct) / len(correct)
        mat_vals[0][i].append(acc)          # row-0, col-i
        print(f"    [base] d{i}: {acc:.3f}")

    # clean up
    sock.send_json({"cmd": "shutdown"}); sock.recv_json()
    sock.close(); ctx.term()
    inner.terminate(); vllm.terminate()
    vllm.wait()
    torch.cuda.empty_cache()

    # Pre-compute question spans for convenience
    q_spans: List[Tuple[int, int]] = []
    cum = 0
    for it in items:
        n_q = len(it["questions"])
        q_spans.append((cum, cum + n_q))
        cum += n_q
    agg_questions: List[Dict[str, str]] = []

    max_model_len = args.max_tokens + 2048  # for vLLM
    for k, item in enumerate(items):
        print(f"[Seq {seq_idx}] Step {k}/{K-1} - {item['title']}")

        # ---------------- 1) spin up infra --------------------------------
        step_tag = f"seq{seq_idx}_step{k}"
        logs_step_dir = Path(args.output_dir) / "logs"
        vllm = _spawn_vllm(current_model_path, "127.0.0.1", args.vllm_port, args.vllm_gpus, logs_step_dir, step_tag, args.lora_rank, max_model_len)
        vllm_api = f"http://127.0.0.1:{args.vllm_port}"
        set_vllm_api_url(vllm_api)
        inner = _spawn_inner_server(vllm_api, current_model_path, args.zmq_port, args.inner_gpu, logs_step_dir, step_tag)
        zmq_ctx, zmq_sock = _connect_zmq(args.zmq_port)

        # ---------------- 2) self-edit completion --------------------------
        prompt = make_prompt(item["title"], item["context"], instruct_model=False)
        comp_resp = requests.post(
            f"{vllm_api}/v1/completions",
            json={
                "model": current_model_path,
                "prompt": [prompt],
                "n": 1,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            timeout=600,
        )
        comp_resp.raise_for_status()
        completion = comp_resp.json()["choices"][0]["text"].strip()

        train_sequences = build_train_sequences(
            completion or item["context"], item["context"], item["title"], split_newlines=True
        )

        # ---------------- 3) extend evaluation question list --------------
        new_q = [
            {
                "title": item["title"],
                "context": item["context"],
                "question": f"Topic: {item['title']}\n{q['question']}",
                "answer": q["answer"],
            }
            for q in item["questions"]
        ]
        agg_questions.extend(new_q)

        # ---------------- 4) tune-and-eval  -------------------------------
        rep_out = _send_round(zmq_sock, train_sequences, agg_questions, args)
        correct = rep_out["adapter_correct"]
        for i in range(k + 1):
            s, e = q_spans[i]
            acc = sum(correct[s:e]) / (e - s)
            mat_vals[k+1][i].append(acc)
        print([f"d{i}:{_stats.mean(mat_vals[k+1][i]):.3f}" for i in range(k + 1)])

        # ---------------- 5) grab adapter & merge into base ---------------
        adapter_path = Path(f"models/tmp_{args.zmq_port}_inner_TTT_0/final_adapter")
        print("[Merge] adapter path:", adapter_path)
        if not adapter_path.exists():
            print("[!] adapter not found - skipping merge, keeping previous base")
        else:
            merged_dir = Path(args.output_dir) / f"merged_seq{seq_idx}_step{k}"
            prev_model_path = current_model_path
            current_model_path = _merge_lora(current_model_path, adapter_path, merged_dir)

            # Clean up previous merge dir if it's not the original base
            if k > 0 and Path(prev_model_path).is_dir() and str(prev_model_path).startswith(str(Path(args.output_dir))):
                try:
                    shutil.rmtree(prev_model_path)
                    print(f"[Cleanup] removed previous merge dir {prev_model_path}")
                except Exception as exc:
                    print(f"[Cleanup] failed to remove {prev_model_path}: {exc}")

            # NEW: Also remove last step of previous sequence, if this is first step of current
            if k == 0 and seq_idx > 0:
                prev_final_merge = Path(args.output_dir) / f"merged_seq{seq_idx - 1}_step{K - 1}"
                if prev_final_merge.exists():
                    try:
                        shutil.rmtree(prev_final_merge)
                        print(f"[Cleanup] removed prior sequence's final merge dir {prev_final_merge}")
                    except Exception as exc:
                        print(f"[Cleanup] failed to remove {prev_final_merge}: {exc}")

        # ---------------- 6) graceful shutdown ----------------------------
        try:
            zmq_sock.send_json({"cmd": "shutdown"}); 
            msg = zmq_sock.recv_json()
            print(msg)
        except Exception:
            pass
        zmq_sock.close(); zmq_ctx.term()
        inner.terminate(); vllm.terminate()
        vllm.wait()
        torch.cuda.empty_cache()

    # ---------------- 7) clean up remaining directories ------------------
    last_merge = Path(current_model_path)
    if last_merge.is_dir() and str(last_merge).startswith(str(Path(args.output_dir))):
        try:
            shutil.rmtree(last_merge)
            print(f"[Cleanup] removed final merge dir {last_merge}")
        except Exception as exc:
            print(f"[Cleanup] failed to remove final merge dir {last_merge}: {exc}")

    # delete the inner-server's "models/tmp_*" folder:
    tmp_dir = Path(f"models/tmp_{args.zmq_port}_inner_TTT_0")
    if tmp_dir.exists():
        try:
            shutil.rmtree(tmp_dir)
            print(f"[Cleanup] removed temporary adapter dir {tmp_dir}")
        except Exception as exc:
            print(f"[Cleanup] failed to remove temporary adapter dir {tmp_dir}: {exc}")

    # ---------------- 8) aggregate mean/std over reps --------------------
    mean_mat: List[List[float]] = [[0.0] * K for _ in range(R)]
    std_mat: List[List[float]]  = [[0.0] * K for _ in range(R)]
    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            vals = mat_vals[r][i]
            if vals:                       # base row has all K cells,
                mean_mat[r][i] = _stats.mean(vals)
                std_mat[r][i]  = _stats.stdev(vals) if len(vals) > 1 else 0.0
    print("mean matrix:\n", json.dumps(mean_mat, indent=2))
    print("std matrix:\n", json.dumps(std_mat, indent=2))
    print("finished - matrices computed\n")
    return mean_mat, std_mat

###############################################################################
#                                   CLI                                       #
###############################################################################

def parse_args():
    p = argparse.ArgumentParser()

    # Dataset & sampling
    p.add_argument("--dataset", required=True)
    p.add_argument("--n_sequences", type=int, default=1, help="How many independent sequences")
    p.add_argument("--n_datapoints", type=int, default=8, help="Datapoints per sequence")

    # Model & placement
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--gpus", default="0,1", help="Comma-separated list → first for vLLM, second for inner")
    p.add_argument("--vllm_port", type=int, default=8001)
    p.add_argument("--zmq_port", type=int, default=5555)

    # Generation params for self-edit completion
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=8192)

    # Inner loop hyper-params (pass-through)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--end_mask_substring", default="")

    p.add_argument("--output_dir", default="knowledge-incorporation/results/continual_self_edits")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

###############################################################################
#                             Top-level driver                                #
###############################################################################

def main():
    args = parse_args()
    _banner("[Args] " + json.dumps(vars(args), indent=2))
    random.seed(args.seed)

    gpus = args.gpus.split(",")
    if len(gpus) < 2:
        sys.exit("[!] --gpus must list at least two IDs (vLLM,inner)")
    args.vllm_gpus, args.inner_gpu = gpus[0], gpus[1]

    full_data: List[Dict[str, Any]] = json.load(Path(args.dataset).open())
    if args.n_datapoints > len(full_data):
        sys.exit("[!] n_datapoints exceeds dataset size")

    seq_matrices_mean: List[List[List[float]]] = []
    seq_matrices_std:  List[List[List[float]]] = []

    for seq_idx in range(args.n_sequences):
        items = random.sample(full_data, args.n_datapoints)
        mean_mat, std_mat = run_one_sequence(seq_idx, items, args)
        seq_matrices_mean.append(mean_mat)
        seq_matrices_std.append(std_mat)

    # -------- aggregate across sequences (simple arithmetic mean) --------
    K = args.n_datapoints
    R = K + 1
    agg_mean = [[0.0] * K for _ in range(R)]
    agg_std  = [[0.0] * K for _ in range(R)]
    for r in range(R):
        cols = K if r == 0 else r
        for i in range(cols):
            vals = [seq_matrices_mean[s][r][i] for s in range(args.n_sequences)]
            agg_mean[r][i] = _stats.mean(vals)
            agg_std[r][i] = _stats.stdev(vals) if len(vals) > 1 else 0.0

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "mean_over_sequences": agg_mean,
            "std_over_sequences": agg_std,
            "n_sequences": args.n_sequences,
            "n_datapoints": args.n_datapoints,
            "dataset": args.dataset,
            "base_model": args.model,
        },
        (out_dir / f"summary_{int(time.time())}.json").open("w"),
        indent=2,
    )
    print("\nfinished - summary written to", out_dir)


if __name__ == "__main__":
    main()
