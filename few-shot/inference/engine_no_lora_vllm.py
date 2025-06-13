from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer
from vllm import LLM, EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def get_sampling_params(
    tokenizer: PreTrainedTokenizer,
    num_tokens: int,
    max_tokens: int,
    temperature: float = 0.0,
    n: int = 1,
) -> SamplingParams:
    max_new_tokens = max_tokens - num_tokens
    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=n,
        stop=[tokenizer.eos_token, "<|eot_id|>"],
        # best_of=10,
        # use_beam_search=True,
    )


def initialize_engine(
    model: str,
    enforce_eager: bool = False,
    enable_lora: bool = True,
    max_lora_rank: int = 64,
    quantization: Optional[str] = None,
    lora_repo: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
) -> LLMEngine:
    """Initialize the LLMEngine."""

    llm = LLM(
        model=model,
        enable_lora=False,
        max_lora_rank=max_lora_rank,
        max_model_len=8192,
    )

    return llm

def process_requests(
    engine: LLMEngine, test_prompts: List[Tuple[str, SamplingParams, Optional[LoRARequest], str]]
) -> Dict[str, List[str]]:
    """Continuously process a list of prompts and handle the outputs."""
    all_outputs: Dict[str, List[str]] = {}
    while test_prompts:
        prompt, sampling_param, lora_request, idx = test_prompts.pop(0)
        find_start = prompt.find("<|begin_of_text|>") + len("<|begin_of_text|>")
        prompt = prompt[find_start:]
        request_outputs = engine.generate(prompt, sampling_param)

        for request_output in request_outputs:
            if request_output.finished:
                texts = [output.text for output in request_output.outputs]
                all_outputs[idx] = texts
    return all_outputs
